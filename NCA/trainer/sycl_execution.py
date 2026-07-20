"""Two-tile data-parallel execution for SYCL NCA training."""

from __future__ import annotations

import copy

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from Common.model.boundary import hard_boundary, model_boundary, no_boundary
from NCA.trainer.sycl_shard_map import filter_shard_map
from NCA.trainer.training_execution import TrainingExecution


class SyclTwoTileExecution(TrainingExecution):
    """Run one outer-batch leaf per PVC tile with replicated parameters.

    Global state arrays have shape ``[2, N, C, H, W]`` and are sharded on
    their leading tile axis. Inside :func:`jax.shard_map`, each tile receives
    ``[1, N, C, H, W]`` and the NCA operates on ``[N, C, H, W]``.
    """

    AXIS_NAME = "nca_sycl_tiles"
    TILE_COUNT = 2

    def __init__(self, trainer):
        super().__init__(trainer)
        self.devices = None
        self.boundary_specs = None
        self.loss_masks = None
        self.cached_losses = None
        self.data_augmenters = None
        self.mesh = None
        self.replicated_sharding = None
        self.tile_sharding = None

    def _ensure_devices(self):
        if self.devices is not None:
            return
        self.devices = [
            device for device in jax.local_devices() if device.platform == "sycl"
        ]
        if len(self.devices) != self.TILE_COUNT:
            raise RuntimeError(
                "trainer.sharding=2 requires exactly two visible SYCL tiles; "
                f"JAX reports {self.devices}"
            )
        self.mesh = Mesh(np.asarray(self.devices), (self.AXIS_NAME,))
        self.replicated_sharding = NamedSharding(self.mesh, P())
        self.tile_sharding = NamedSharding(self.mesh, P(self.AXIS_NAME))

    @staticmethod
    def _array_pmean(tree, axis_name):
        """Average every array leaf over ``axis_name``; preserve static leaves."""
        return jtu.tree_map(
            lambda value: jax.lax.pmean(value, axis_name)
            if hasattr(value, "shape") and hasattr(value, "dtype")
            else value,
            tree,
        )

    @staticmethod
    def _remove_local_tile_axis(tree, name):
        """Remove shard_map's size-one physical tile dimension.

        The state/target/key boundary contains only array leaves. Each mapped
        leaf retains its rank and has local leading shape ``[1, ...]``.
        """

        def remove(value):
            if not hasattr(value, "shape"):
                raise TypeError(
                    f"Tile-local {name} must contain only array leaves; got "
                    f"{type(value).__name__}"
                )
            if value.ndim == 0 or value.shape[0] != 1:
                raise ValueError(
                    "Each shard_map tile must receive exactly one outer-B "
                    f"{name} leaf; got shape {value.shape}"
                )
            return value[0]

        return jtu.tree_map(remove, tree)

    @staticmethod
    def _add_local_tile_axis(tree, name):
        """Map array leaves ``[...]`` to ``[1, ...]`` for shard-map output."""
        def add(value):
            if not hasattr(value, "shape"):
                raise TypeError(
                    f"Tile-local {name} must contain only array leaves; got "
                    f"{type(value).__name__}"
                )
            return value[None]

        return jtu.tree_map(add, tree)

    def transform_loss(self, compute_loss):
        """Shard a local loss before the caller applies reverse-mode AD.

        ``compute_loss`` consumes one tile-local outer-B PyTree whose state
        leaves have shape ``[N, C, H, W]``. The returned loss is scalar and
        replicated; auxiliary state and per-example losses regain a leading
        tile axis for the global trainer.
        """
        self._ensure_devices()

        def tile_local_loss(nca_diff, nca_static, x, y, t, key):
            local_x = self._remove_local_tile_axis(x, "state")
            local_y = self._remove_local_tile_axis(y, "target")
            local_key = self._remove_local_tile_axis(key, "PRNG key")
            mean_loss, auxiliary = compute_loss(
                nca_diff, nca_static, local_x, local_y, t, local_key
            )
            (
                x_latent,
                x_processed,
                losses,
                regulariser_losses,
            ) = auxiliary
            return mean_loss, (
                self._add_local_tile_axis(x_latent, "state"),
                self._add_local_tile_axis(x_processed, "processed state"),
                self._add_local_tile_axis(losses, "loss array"),
                regulariser_losses,
            )

        return filter_shard_map(
            tile_local_loss,
            mesh=self.mesh,
            in_specs=(
                P(),
                P(),
                P(self.AXIS_NAME),
                P(self.AXIS_NAME),
                P(),
                P(self.AXIS_NAME),
            ),
            out_specs=(
                P(),
                (
                    P(self.AXIS_NAME),
                    P(self.AXIS_NAME),
                    P(self.AXIS_NAME),
                    P(),
                ),
            ),
            # Native custom-call primitives do not provide shard-map
            # replication-analysis rules in JAX 0.5.
            check_rep=False,
        )

    def synchronise_loss(self, mean_loss, regulariser_losses):
        """Average scalar losses across tiles inside the sharded loss."""
        return (
            jax.lax.pmean(mean_loss, self.AXIS_NAME),
            self._array_pmean(regulariser_losses, self.AXIS_NAME),
        )

    def _pack_items(self, items, name, *, sharded=True, expected_ndim=None):
        """Stack two equal PyTrees as one ``[tile, ...]`` outer-B slot.

        Parameters
        ----------
        items:
            Length-two outer-B sequence. Corresponding array leaves must have
            identical shapes.
        sharded:
            Place each size-one leading slice directly on its owning tile.
            Static metadata uses an ordinary stack when ``False``.
        expected_ndim:
            Optional rank required before adding the tile axis.

        Returns
        -------
        list
            One outer-B slot whose array leaves have shape ``[2, ...]``.
        """
        if len(items) != self.TILE_COUNT:
            raise ValueError(
                f"Two-tile training currently requires exactly two {name}; "
                f"got {len(items)}"
            )

        def pack_leaf(left, right):
            if eqx.is_array(left) and eqx.is_array(right):
                if expected_ndim is not None and (
                    left.ndim != expected_ndim or right.ndim != expected_ndim
                ):
                    raise ValueError(
                        f"Each {name} leaf must have rank {expected_ndim} "
                        f"before adding the tile axis; got {left.shape} and "
                        f"{right.shape}"
                    )
                if sharded:
                    packed = self._make_tile_array(left, right)
                else:
                    packed = jnp.stack((left, right), axis=0)
                expected_shape = (self.TILE_COUNT, *left.shape)
                if packed.shape != expected_shape:
                    raise ValueError(
                        f"Packed {name} has shape {packed.shape}; expected "
                        f"one tile axis followed by {left.shape}"
                    )
                return packed
            if left == right:
                return left
            raise ValueError(
                f"Two-tile {name} contain unequal static leaves: "
                f"{left!r} and {right!r}"
            )

        try:
            packed = jtu.tree_map(pack_leaf, items[0], items[1])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"The two {name} must have identical PyTree structures and "
                "array shapes"
            ) from exc
        return [packed]

    def _make_tile_array(self, left, right):
        """Create a global ``[2, ...]`` array from two single-tile values."""
        self._ensure_devices()
        shape = (self.TILE_COUNT, *left.shape)
        local_arrays = [
            jax.device_put(value[None], device)
            for value, device in zip((left, right), self.devices)
        ]
        return jax.make_array_from_single_device_arrays(
            shape, self.tile_sharding, local_arrays
        )

    def _local_tile_trees(self, tree, name):
        """Return one single-device PyTree per tile without global indexing.

        A ``NamedSharding`` array has global indexing semantics. In particular,
        ``value[tile]`` may gather or reshard the global value rather than
        returning the physical buffer owned by that tile. Host-side callbacks
        and logging need the physical buffers, exposed by
        ``addressable_shards``.
        """
        self._ensure_devices()
        leaves, tree_definition = jtu.tree_flatten(tree)
        local_leaves = [[] for _ in range(self.TILE_COUNT)]
        for leaf in leaves:
            if not hasattr(leaf, "addressable_shards"):
                raise TypeError(
                    f"Two-tile {name} must contain sharded JAX arrays; got "
                    f"{type(leaf).__name__}"
                )
            shards_by_device = {
                shard.device: shard.data for shard in leaf.addressable_shards
            }
            for tile, device in enumerate(self.devices):
                if device not in shards_by_device:
                    raise ValueError(
                        f"Two-tile {name} has no addressable shard on {device}; "
                        f"available devices are {list(shards_by_device)}"
                    )
                local = shards_by_device[device]
                if local.ndim == 0 or local.shape[0] != 1:
                    raise ValueError(
                        f"Expected one physical {name} item on {device}, got "
                        f"local shape {local.shape}"
                    )
                local_leaves[tile].append(local[0])
        return [
            jtu.tree_unflatten(tree_definition, tile_leaves)
            for tile_leaves in local_leaves
        ]

    def _replicate(self, tree):
        """Replicate every array leaf across both tiles."""
        self._ensure_devices()
        return jtu.tree_map(
            lambda value: jax.device_put(value, self.replicated_sharding)
            if eqx.is_array(value)
            else value,
            tree,
        )

    @staticmethod
    def _select_slots(packed_slots, tile_index):
        """Select one tile from metadata shaped ``[tile, ...]`` inside SPMD."""
        return [
            jtu.tree_map(
                lambda value: jax.lax.dynamic_index_in_dim(
                    value, tile_index, axis=0, keepdims=False
                )
                if eqx.is_array(value)
                else value,
                slot,
            )
            for slot in packed_slots
        ]

    def _prepare_boundary_specs(self):
        """Encode two matching boundary callbacks as tile-indexed arrays."""
        left, right = self.trainer.BOUNDARY_CALLBACK
        if type(left) is not type(right):
            raise ValueError(
                "Two-tile training requires matching boundary modes for both "
                f"B leaves, got {type(left).__name__} and {type(right).__name__}"
            )
        if isinstance(left, no_boundary):
            return [(0, None)]
        if isinstance(left, model_boundary):
            return [(1, jnp.stack((left.MASK, right.MASK), axis=0))]
        if isinstance(left, hard_boundary):
            return [(2, jnp.stack((left.MASK, right.MASK), axis=0))]
        raise NotImplementedError(
            "Two-tile SYCL training currently supports no_boundary, "
            "model_boundary, and hard_boundary"
        )

    def boundary_callbacks(self):
        """Return the current tile's singleton outer-B boundary callback list."""
        if self.boundary_specs is None:
            return super().boundary_callbacks()
        tile_index = jax.lax.axis_index(self.AXIS_NAME)
        callbacks = []
        for boundary_code, packed_mask in self.boundary_specs:
            if boundary_code == 0:
                callbacks.append(no_boundary())
                continue
            mask = jax.lax.dynamic_index_in_dim(
                packed_mask, tile_index, axis=0, keepdims=False
            )
            callbacks.append(
                model_boundary(mask)
                if boundary_code == 1
                else hard_boundary(mask[None])
            )
        return callbacks

    def loss_time_channel_mask(self):
        """Return one tile-local loss-mask slot."""
        if self.loss_masks is None:
            return super().loss_time_channel_mask()
        return self._select_slots(
            self.loss_masks, jax.lax.axis_index(self.AXIS_NAME)
        )

    def loss_cache(self):
        """Return one tile-local loss-cache slot."""
        if self.cached_losses is None:
            return super().loss_cache()
        return self._select_slots(
            self.cached_losses, jax.lax.axis_index(self.AXIS_NAME)
        )

    def _slice_data_augmenter(self, augmenter, batch_index):
        """Copy one outer-B leaf of augmenter state onto its owning tile."""
        local = copy.copy(augmenter)
        device = self.devices[batch_index]

        def place(value):
            return jax.device_put(value, device) if eqx.is_array(value) else value

        for attribute in (
            "data_true",
            "data_saved",
            "channel_timestep_mask",
            "knockout_times",
        ):
            if not hasattr(local, attribute):
                continue
            value = getattr(local, attribute)
            if isinstance(value, list):
                local_value = [jtu.tree_map(place, value[batch_index])]
            elif isinstance(value, tuple):
                local_value = (jtu.tree_map(place, value[batch_index]),)
            elif hasattr(value, "shape") and value.ndim > 0:
                local_value = jax.device_put(
                    value[batch_index : batch_index + 1], device
                )
            else:
                continue
            setattr(local, attribute, local_value)
        return local

    def prepare_inputs(self, nca, x, y, opt_state, key):
        """Pack two outer-B leaves and replicate model and optimiser arrays.

        ``x`` and ``y`` are length-two sequences with leaves ``[N,C,H,W]``.
        Packed state and target leaves are global ``[2,N,C,H,W]`` arrays.
        The returned PRNG key has global shape ``[2,2]`` for legacy keys.
        """
        self._ensure_devices()
        if not isinstance(x, (list, tuple)) or not isinstance(y, (list, tuple)):
            raise TypeError(
                "Two-tile NCA training currently requires outer-B list/tuple data"
            )

        self.boundary_specs = self._prepare_boundary_specs()
        self.loss_masks = self._pack_items(
            self.trainer.LOSS_TIME_CHANNEL_MASK, "loss masks", sharded=False
        )
        self.cached_losses = self._pack_items(
            self.trainer.LOSS_CACHE, "loss-cache entries", sharded=False
        )
        self.data_augmenters = [
            self._slice_data_augmenter(self.trainer.DATA_AUGMENTER, tile)
            for tile in range(self.TILE_COUNT)
        ]
        packed_x = self._pack_items(
            x, "outer-B state leaves", expected_ndim=4
        )
        packed_y = self._pack_items(
            y, "outer-B target leaves", expected_ndim=4
        )
        left_key, right_key = jr.split(key, self.TILE_COUNT)
        tile_keys = self._make_tile_array(left_key, right_key)
        print(
            "NCA SYCL shard_map data parallelism enabled: one outer-B leaf "
            "per tile with replicated parameters and loss reduction.",
            flush=True,
        )
        return (
            self._replicate(nca),
            packed_x,
            packed_y,
            self._replicate(opt_state),
            tile_keys,
        )

    def fold_in_key(self, key, iteration):
        """Fold ``iteration`` into each physical tile's legacy ``[2]`` key."""
        local_keys = self._local_tile_trees(key, "PRNG key")
        return self._make_tile_array(
            jr.fold_in(local_keys[0], iteration),
            jr.fold_in(local_keys[1], iteration),
        )

    def split_key(self, key):
        """Split each tile key and return global next/callback key arrays."""
        local_keys = self._local_tile_trees(key, "PRNG key")
        pairs = [jr.split(tile_key) for tile_key in local_keys]
        next_key = self._make_tile_array(pairs[0][0], pairs[1][0])
        callback_key = self._make_tile_array(pairs[0][1], pairs[1][1])
        return next_key, callback_key

    def _pack_local_slots(self, local_trees, name):
        """Reassemble two tile-local singleton outer-B callback results."""
        if len(local_trees) != self.TILE_COUNT or any(
            len(tree) != 1 for tree in local_trees
        ):
            raise ValueError(
                f"Two-tile {name} callback must return one outer-B leaf per tile"
            )
        return [
            jtu.tree_map(
                self._make_tile_array,
                local_trees[0][0],
                local_trees[1][0],
            )
        ]

    def apply_data_callback(self, x, y, iteration, key):
        """Run each augmenter on single-device ``[N,C,H,W]`` state leaves."""
        local_x = self._local_tile_trees(x, "state")
        local_y = self._local_tile_trees(y, "target")
        local_keys = self._local_tile_trees(key, "PRNG key")
        eligible = max(0, local_x[0][0].shape[0] - 1)
        global_injections = (self.TILE_COUNT * eligible) // 2
        injections = [global_injections // self.TILE_COUNT] * self.TILE_COUNT
        for offset in range(global_injections % self.TILE_COUNT):
            injections[(iteration + offset) % self.TILE_COUNT] += 1

        outputs = []
        for tile in range(self.TILE_COUNT):
            augmenter = self.data_augmenters[tile]
            if getattr(augmenter, "supports_sharded_inject_count", False):
                augmenter._sharded_n_inject = injections[tile]
            outputs.append(
                augmenter.data_callback(
                    local_x[tile], local_y[tile], iteration, local_keys[tile]
                )
            )
        return (
            self._pack_local_slots([output[0] for output in outputs], "state"),
            self._pack_local_slots([output[1] for output in outputs], "target"),
        )

    def _unpack_slots(self, packed_slots):
        """Convert global ``[2,...]`` logging arrays back to outer-B leaves."""
        tile_trees = self._local_tile_trees(packed_slots, "logged state")
        return [slot for tile_tree in tile_trees for slot in tile_tree]

    def prepare_log_dict(self, log_dict):
        """Restore the reference trainer's outer-B logging structures."""
        result = dict(log_dict)
        for name in ("x_latent", "x_processed"):
            result[name] = self._unpack_slots(result[name])
        losses = result["losses"]
        result["losses"] = losses.reshape(
            (losses.shape[0] * losses.shape[1], *losses.shape[2:])
        )
        return result


__all__ = ["SyclTwoTileExecution"]
