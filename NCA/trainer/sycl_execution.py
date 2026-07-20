"""Two-stack execution policy for SYCL NCA training."""

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
    """Replicate parameters and split two outer-B leaves over two PVC stacks."""

    AXIS_NAME = "nca_sycl_tiles"

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
        if len(self.devices) != 2:
            raise RuntimeError(
                "trainer.sharding=2 requires exactly two visible SYCL tiles; "
                f"JAX reports {self.devices}"
            )
        self.mesh = Mesh(np.asarray(self.devices), (self.AXIS_NAME,))
        self.replicated_sharding = NamedSharding(self.mesh, P())
        self.tile_sharding = NamedSharding(self.mesh, P(self.AXIS_NAME))

    @staticmethod
    def _array_pmean(tree, axis_name):
        return jtu.tree_map(
            lambda value: jax.lax.pmean(value, axis_name)
            if hasattr(value, "shape") and hasattr(value, "dtype")
            else value,
            tree,
        )

    @staticmethod
    def _remove_local_tile_axis(tree, name):
        """Remove shard_map's size-one physical tile dimension.

        This code runs while ``tree`` contains JAX tracers.  In particular, it
        must not use a host-side array type predicate: older Intel JAX and
        Equinox combinations do not consistently classify every shard_map
        tracer as an array.  The state/target/key boundary is deliberately an
        array-only contract, so checking its abstract ``shape`` is both simpler
        and stricter.
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
        def add(value):
            if not hasattr(value, "shape"):
                raise TypeError(
                    f"Tile-local {name} must contain only array leaves; got "
                    f"{type(value).__name__}"
                )
            return value[None]

        return jtu.tree_map(add, tree)

    def transform_step(self, make_step):
        self._ensure_devices()

        # The loss itself is mapped in ``transform_loss`` and differentiated
        # outside shard_map. The surrounding optimiser/update step is ordinary
        # global JIT code over correctly sharded arrays.
        return eqx.filter_jit(make_step)

    def transform_loss(self, compute_loss):
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
            check_rep=False,
        )

    def synchronise_gradients(self, gradients):
        # Reverse-mode autodiff is outside shard_map, so the transpose of the
        # pmean in ``synchronise_loss`` performs the parameter-gradient
        # reduction exactly once.
        return gradients

    def synchronise_loss(self, mean_loss, regulariser_losses):
        return (
            jax.lax.pmean(mean_loss, self.AXIS_NAME),
            self._array_pmean(regulariser_losses, self.AXIS_NAME),
        )

    def _pack_items(self, items, name, *, sharded=True, expected_ndim=None):
        if len(items) != 2:
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
                expected_shape = (2, *left.shape)
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
        self._ensure_devices()
        shape = (2, *left.shape)
        local_arrays = [
            jax.device_put(value[None], device)
            for value, device in zip((left, right), self.devices)
        ]
        return jax.make_array_from_single_device_arrays(
            shape, self.tile_sharding, local_arrays
        )

    def _replicate(self, tree):
        self._ensure_devices()
        return jtu.tree_map(
            lambda value: jax.device_put(value, self.replicated_sharding)
            if eqx.is_array(value)
            else value,
            tree,
        )

    @staticmethod
    def _select_slots(packed_slots, tile_index):
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
        if self.loss_masks is None:
            return super().loss_time_channel_mask()
        return self._select_slots(
            self.loss_masks, jax.lax.axis_index(self.AXIS_NAME)
        )

    def loss_cache(self):
        if self.cached_losses is None:
            return super().loss_cache()
        return self._select_slots(
            self.cached_losses, jax.lax.axis_index(self.AXIS_NAME)
        )

    def _slice_data_augmenter(self, augmenter, batch_index):
        if hasattr(augmenter, "for_batch_indices"):
            return augmenter.for_batch_indices([batch_index])
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
            for tile in range(2)
        ]
        packed_x = self._pack_items(
            x, "outer-B state leaves", expected_ndim=4
        )
        packed_y = self._pack_items(
            y, "outer-B target leaves", expected_ndim=4
        )
        left_key, right_key = jr.split(key, 2)
        tile_keys = self._make_tile_array(left_key, right_key)
        print(
            "NCA SYCL shard_map data parallelism enabled: one outer-B leaf "
            "per tile, replicated parameters, one gradient pmean.",
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
        return self._make_tile_array(
            jr.fold_in(key[0], iteration),
            jr.fold_in(key[1], iteration),
        )

    def split_key(self, key):
        pairs = [jr.split(key[tile]) for tile in range(2)]
        next_key = self._make_tile_array(pairs[0][0], pairs[1][0])
        callback_key = self._make_tile_array(pairs[0][1], pairs[1][1])
        return next_key, callback_key

    def _pack_local_slots(self, local_trees, name):
        if len(local_trees) != 2 or any(len(tree) != 1 for tree in local_trees):
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
        local_x = [[x[0][tile]] for tile in range(2)]
        local_y = [[y[0][tile]] for tile in range(2)]
        eligible = max(0, local_x[0][0].shape[0] - 1)
        global_injections = (2 * eligible) // 2
        injections = [global_injections // 2] * 2
        for offset in range(global_injections % 2):
            injections[(iteration + offset) % 2] += 1

        outputs = []
        for tile in range(2):
            augmenter = self.data_augmenters[tile]
            if getattr(augmenter, "supports_sharded_inject_count", False):
                augmenter._sharded_n_inject = injections[tile]
            outputs.append(
                augmenter.data_callback(
                    local_x[tile], local_y[tile], iteration, key[tile]
                )
            )
        return (
            self._pack_local_slots([output[0] for output in outputs], "state"),
            self._pack_local_slots([output[1] for output in outputs], "target"),
        )

    @staticmethod
    def _unpack_slots(packed_slots):
        return [
            jtu.tree_map(lambda value: value[tile], slot)
            for slot in packed_slots
            for tile in range(2)
        ]

    def prepare_log_dict(self, log_dict):
        result = dict(log_dict)
        if eqx.is_array(result["loss"]) and result["loss"].ndim > 0:
            result["loss"] = result["loss"][0]
        for name in ("x_latent", "x_processed"):
            result[name] = self._unpack_slots(result[name])
        losses = result["losses"]
        result["losses"] = losses.reshape(
            (losses.shape[0] * losses.shape[1], *losses.shape[2:])
        )
        for name, value in tuple(result.items()):
            if name not in ("x_latent", "x_processed", "losses", "loss"):
                if eqx.is_array(value) and value.ndim > 0 and value.shape[0] == 2:
                    result[name] = value[0]
        return result


__all__ = ["SyclTwoTileExecution"]
