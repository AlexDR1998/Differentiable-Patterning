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
from Common.trainer.loss_multi_target import (
    multi_target_assignment,
    multi_target_pairwise_costs,
)
from NCA.trainer.sycl_shard_map import filter_shard_map
from NCA.trainer.training_execution import TrainingExecution


class SyclTwoTileExecution(TrainingExecution):
    """Evenly split outer-batch leaves over two PVC tiles.

    With ``B = 2M`` outer batches, global state is represented as a length-``M``
    list of arrays shaped ``[2,N,C,H,W]``. Inside :func:`jax.shard_map`, each
    tile receives a length-``M`` list of ``[1,N,C,H,W]`` shards; removing the
    physical tile axis gives the NCA its usual ``[N,C,H,W]`` inputs.
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

    @classmethod
    def _split_between_tiles(cls, items, name):
        """Return two equal contiguous outer-B partitions.

        A sequence of length ``B = 2M`` becomes ``(items[:M], items[M:])``.
        Contiguous splitting makes unpacking restore the original batch order.
        """
        if not isinstance(items, (list, tuple)):
            raise TypeError(f"Two-tile {name} must be a list or tuple")
        if len(items) < cls.TILE_COUNT or len(items) % cls.TILE_COUNT != 0:
            raise ValueError(
                f"Two-tile {name} count must be a positive multiple of "
                f"{cls.TILE_COUNT}; got {len(items)}"
            )
        per_tile = len(items) // cls.TILE_COUNT
        return tuple(
            items[tile * per_tile : (tile + 1) * per_tile]
            for tile in range(cls.TILE_COUNT)
        )

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
                    f"Each packed outer-B {name} slot must have one physical "
                    f"value per shard_map tile; got shape {value.shape}"
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
                loss_diagnostics,
            ) = auxiliary
            return mean_loss, (
                self._add_local_tile_axis(x_latent, "state"),
                self._add_local_tile_axis(x_processed, "processed state"),
                self._add_local_tile_axis(losses, "loss array"),
                regulariser_losses,
                loss_diagnostics,
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
                    P(),
                ),
            ),
            # Native custom-call primitives do not provide shard-map
            # replication-analysis rules in JAX 0.5.
            check_rep=False,
        )

    def synchronise_loss(self, mean_loss, regulariser_losses):
        """Average scalar losses across tiles inside the sharded loss."""
        # With pmean disabled, P() deliberately treats the tile-local scalar as
        # replicated (check_rep=False). This is a diagnostic mode, not a
        # numerically equivalent training configuration.
        return (
            jax.lax.pmean(mean_loss, self.AXIS_NAME)
            if getattr(getattr(self, "trainer", None), "pmean_loss", True)
            else mean_loss,
            self._array_pmean(regulariser_losses, self.AXIS_NAME)
            if getattr(
                getattr(self, "trainer", None), "pmean_regularisers", True
            )
            else regulariser_losses,
        )

    def multi_target_loss(
        self, prediction, target, boundary, schema, params, key, args
    ):
        """Gather scalar cost rows instead of complete prediction images."""
        target = jax.lax.all_gather(
            target, self.AXIS_NAME, axis=0, tiled=False
        ).reshape((-1, *target.shape[1:]))
        key = jax.lax.all_gather(key, self.AXIS_NAME)[0]
        costs, components = multi_target_pairwise_costs(
            prediction, target, boundary, schema, params, key, args
        )
        gather_rows = lambda value: jax.lax.all_gather(
            value, self.AXIS_NAME, axis=1, tiled=True
        )
        return multi_target_assignment(
            gather_rows(costs),
            jtu.tree_map(gather_rows, components),
            schema,
            args,
        )

    def _pack_items(self, items, name, *, sharded=True, expected_ndim=None):
        """Pair equal tile partitions into global ``[tile,...]`` slots.

        Parameters
        ----------
        items:
            Length-``B`` outer-B sequence, where ``B`` is positive and even.
            Corresponding leaves in the two contiguous halves must match.
        sharded:
            Place each size-one leading slice directly on its owning tile.
            Static metadata uses an ordinary stack when ``False``.
        expected_ndim:
            Optional rank required before adding the tile axis.

        Returns
        -------
        list
            ``B/2`` slots whose array leaves have shape ``[2,...]``.
        """
        tile_items = self._split_between_tiles(items, name)

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

        packed_slots = []
        for slot, (left, right) in enumerate(zip(*tile_items)):
            try:
                packed_slots.append(jtu.tree_map(pack_leaf, left, right))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Tile-local {name} slot {slot} must have identical "
                    "PyTree structures and array shapes"
                ) from exc
        return packed_slots

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
        """Encode paired boundary callbacks as tile-indexed slot metadata."""
        tile_callbacks = self._split_between_tiles(
            self.trainer.BOUNDARY_CALLBACK, "boundary callbacks"
        )
        specs = []
        for slot, (left, right) in enumerate(zip(*tile_callbacks)):
            if type(left) is not type(right):
                raise ValueError(
                    f"Boundary slot {slot} has different modes across tiles: "
                    f"{type(left).__name__} and {type(right).__name__}"
                )
            if isinstance(left, no_boundary):
                specs.append((0, None))
            elif isinstance(left, model_boundary):
                specs.append((1, jnp.stack((left.MASK, right.MASK), axis=0)))
            elif isinstance(left, hard_boundary):
                specs.append((2, jnp.stack((left.MASK, right.MASK), axis=0)))
            else:
                raise NotImplementedError(
                    "Two-tile SYCL training supports no_boundary, "
                    "model_boundary, and hard_boundary"
                )
        return specs

    def boundary_callbacks(self):
        """Return the current tile's outer-B boundary callback list."""
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
        """Return all tile-local outer-B loss-mask slots."""
        if self.loss_masks is None:
            return super().loss_time_channel_mask()
        return self._select_slots(
            self.loss_masks, jax.lax.axis_index(self.AXIS_NAME)
        )

    def loss_cache(self):
        """Return all tile-local outer-B loss-cache slots."""
        if self.cached_losses is None:
            return super().loss_cache()
        return self._select_slots(
            self.cached_losses, jax.lax.axis_index(self.AXIS_NAME)
        )

    def _slice_data_augmenter(self, augmenter, batch_indices, tile_index):
        """Copy one tile's outer-B subset of augmenter state to that device."""
        local = copy.copy(augmenter)
        device = self.devices[tile_index]

        def place(value):
            return jax.device_put(value, device) if eqx.is_array(value) else value

        global_donors = getattr(augmenter, "supports_global_donor_pool", False)
        for attribute in (
            "data_true",
            "data_saved",
            "channel_timestep_mask",
            "knockout_times",
        ):
            if not hasattr(local, attribute):
                continue
            value = getattr(local, attribute)
            indices_to_copy = range(len(value)) if global_donors and attribute in {
                "data_true", "data_saved"
            } else batch_indices
            if isinstance(value, list):
                local_value = [jtu.tree_map(place, value[i]) for i in indices_to_copy]
            elif isinstance(value, tuple):
                local_value = tuple(
                    jtu.tree_map(place, value[i]) for i in indices_to_copy
                )
            elif hasattr(value, "shape") and value.ndim > 0:
                indices = jnp.asarray(tuple(indices_to_copy))
                local_value = jax.device_put(value[indices], device)
            else:
                continue
            setattr(local, attribute, local_value)
        local._global_batch_indices = jax.device_put(
            jnp.asarray(batch_indices), device
        )
        return local

    def prepare_inputs(self, nca, x, y, opt_state, key):
        """Evenly pack outer-B leaves and replicate model/optimiser arrays.

        ``x`` and ``y`` contain ``B = 2M`` leaves shaped ``[N,C,H,W]``.
        Packed values are length-``M`` lists of global ``[2,N,C,H,W]`` arrays.
        The returned legacy PRNG key has global shape ``[2,2]``.
        """
        self._ensure_devices()
        if not isinstance(x, (list, tuple)) or not isinstance(y, (list, tuple)):
            raise TypeError(
                "Two-tile NCA training currently requires outer-B list/tuple data"
            )
        if len(x) != len(y):
            raise ValueError(
                f"State and target outer-B counts differ: {len(x)} and {len(y)}"
            )

        for name, values in (
            ("boundary callbacks", self.trainer.BOUNDARY_CALLBACK),
            ("loss masks", self.trainer.LOSS_TIME_CHANNEL_MASK),
            ("loss-cache entries", self.trainer.LOSS_CACHE),
        ):
            if len(values) != len(x):
                raise ValueError(
                    f"Expected {len(x)} {name} for the outer-B leaves; "
                    f"got {len(values)}"
                )

        batch_indices = self._split_between_tiles(
            tuple(range(len(x))), "outer-B batches"
        )

        self.boundary_specs = self._prepare_boundary_specs()
        self.loss_masks = self._pack_items(
            self.trainer.LOSS_TIME_CHANNEL_MASK, "loss masks", sharded=False
        )
        self.cached_losses = self._pack_items(
            self.trainer.LOSS_CACHE, "loss-cache entries", sharded=False
        )
        self.data_augmenters = [
            self._slice_data_augmenter(
                self.trainer.DATA_AUGMENTER, batch_indices[tile], tile
            )
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
            f"NCA SYCL shard_map data parallelism enabled: {len(x)} outer-B "
            f"leaves split evenly across {self.TILE_COUNT} tiles with "
            "replicated parameters and loss reduction.",
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
        """Pair equal tile-local callback lists into global sharded slots."""
        if len(local_trees) != self.TILE_COUNT:
            raise ValueError(
                f"Two-tile {name} callback returned {len(local_trees)} tile trees"
            )
        slot_counts = {len(tree) for tree in local_trees}
        invalid_counts = (
            len(slot_counts) != 1
            or not slot_counts
            or next(iter(slot_counts)) < 1
        )
        if invalid_counts:
            raise ValueError(
                f"Two-tile {name} callbacks must return equal nonempty outer-B lists; "
                f"got {[len(tree) for tree in local_trees]}"
            )
        return [
            jtu.tree_map(self._make_tile_array, left, right)
            for left, right in zip(*local_trees)
        ]

    @staticmethod
    def _allocate_injections(eligible_counts, iteration):
        """Split half of all eligible pool entries proportionally over tiles."""
        total_eligible = sum(eligible_counts)
        total_injections = total_eligible // 2
        if total_eligible == 0:
            return [0] * len(eligible_counts)

        numerators = [total_injections * count for count in eligible_counts]
        allocations = [value // total_eligible for value in numerators]
        remainder = total_injections - sum(allocations)
        priorities = sorted(
            range(len(eligible_counts)),
            key=lambda tile: (
                numerators[tile] % total_eligible,
                -((tile - iteration) % len(eligible_counts)),
            ),
            reverse=True,
        )
        for tile in priorities[:remainder]:
            allocations[tile] += 1
        return allocations

    def apply_data_callback(self, x, y, iteration, key):
        """Run augmenters on each tile's list of ``[N,C,H,W]`` leaves."""
        local_x = self._local_tile_trees(x, "state")
        local_y = self._local_tile_trees(y, "target")
        local_keys = self._local_tile_trees(key, "PRNG key")
        eligible_counts = [
            sum(max(0, slot.shape[0] - 1) for slot in tile_tree)
            for tile_tree in local_x
        ]
        injections = self._allocate_injections(eligible_counts, iteration)

        outputs = []
        for tile in range(self.TILE_COUNT):
            augmenter = self.data_augmenters[tile]
            if getattr(augmenter, "supports_global_donor_pool", False):
                augmenter._sharded_global_key = jax.device_put(
                    local_keys[0], self.devices[tile]
                )
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
        """Restore tile-major global slots to the original outer-B order."""
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
