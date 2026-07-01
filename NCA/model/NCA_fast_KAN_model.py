import time

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from Common.model.fast_kan import (
    FastLinearSplineKANLayer,
    FastRBFKANLayer,
    get_top_edges_from_layer,
)
from NCA.model.NCA_model import NCA, Ops


class SpatialFastRBFKANLayer(eqx.Module):
    layer: FastRBFKANLayer

    def __call__(
        self, x: Float[Array, "{self.layer.in_features} x y"], key=None
    ) -> Float[Array, "{self.layer.out_features} x y"]:
        apply_y = eqx.filter_vmap(self.layer, in_axes=1, out_axes=1)
        apply_x_y = eqx.filter_vmap(apply_y, in_axes=1, out_axes=1)
        return apply_x_y(x)


class FastKaNCA(NCA):
    layers: list
    KERNEL_STR: list
    N_CHANNELS: int
    N_FEATURES: int
    FIRE_RATE: float
    KAN_AUX: dict
    op: Ops
    perception: callable

    def __init__(
        self,
        N_CHANNELS,
        KERNEL_STR=["ID", "LAP"],
        ACTIVATION=jax.nn.relu,
        PADDING="CIRCULAR",
        FIRE_RATE=1.0,
        KERNEL_SCALE=1,
        KAN_AUX=None,
        key=None,
    ):
        if key is None:
            key = jax.random.PRNGKey(int(time.time()))
        super().__init__(
            N_CHANNELS,
            KERNEL_STR,
            ACTIVATION,
            PADDING,
            FIRE_RATE,
            KERNEL_SCALE,
            key,
        )

        default_kan_aux = {
            "basis": "rbf",
            "hidden_features": self.N_FEATURES,
            "num_basis": 8,
            "grid_min": -2.0,
            "grid_max": 2.0,
            "rbf_width": None,
            "trainable_width": True,
            "extrapolation": "constant",
            "use_base_branch": True,
            "base_activation": "identity",
            "use_layernorm": True,
            "spline_init_scale": 0.1,
            "base_init_scale": 0.1,
            "final_zero_init": True,
        }
        if KAN_AUX is not None:
            default_kan_aux.update(KAN_AUX)
        self.KAN_AUX = default_kan_aux

        hidden_features = self.KAN_AUX["hidden_features"]
        key1, key2 = jax.random.split(key, 2)
        layer_cls = self._resolve_kan_layer_class(self.KAN_AUX["basis"])

        input_layer = layer_cls(
            self.N_FEATURES,
            hidden_features,
            num_basis=self.KAN_AUX["num_basis"],
            grid_min=self.KAN_AUX["grid_min"],
            grid_max=self.KAN_AUX["grid_max"],
            rbf_width=self.KAN_AUX["rbf_width"],
            trainable_width=self.KAN_AUX["trainable_width"],
            extrapolation=self.KAN_AUX["extrapolation"],
            use_base_branch=self.KAN_AUX["use_base_branch"],
            base_activation=self.KAN_AUX["base_activation"],
            use_layernorm=self.KAN_AUX["use_layernorm"],
            spline_init_scale=self.KAN_AUX["spline_init_scale"],
            base_init_scale=self.KAN_AUX["base_init_scale"],
            final_zero_init=False,
            key=key1,
        )
        output_layer = layer_cls(
            hidden_features,
            self.N_CHANNELS,
            num_basis=self.KAN_AUX["num_basis"],
            grid_min=self.KAN_AUX["grid_min"],
            grid_max=self.KAN_AUX["grid_max"],
            rbf_width=self.KAN_AUX["rbf_width"],
            trainable_width=self.KAN_AUX["trainable_width"],
            extrapolation=self.KAN_AUX["extrapolation"],
            use_base_branch=self.KAN_AUX["use_base_branch"],
            base_activation=self.KAN_AUX["base_activation"],
            use_layernorm=self.KAN_AUX["use_layernorm"],
            spline_init_scale=self.KAN_AUX["spline_init_scale"],
            base_init_scale=self.KAN_AUX["base_init_scale"],
            final_zero_init=self.KAN_AUX["final_zero_init"],
            key=key2,
        )

        self.layers = [
            SpatialFastRBFKANLayer(input_layer),
            SpatialFastRBFKANLayer(output_layer),
        ]

    def _resolve_kan_layer_class(self, basis):
        if basis == "rbf":
            return FastRBFKANLayer
        if basis == "linear_spline":
            return FastLinearSplineKANLayer
        raise ValueError("KAN_AUX['basis'] must be 'rbf' or 'linear_spline'.")

    def get_config(self):
        return {
            "MODEL": "FastKaNCA",
            "N_CHANNELS": self.N_CHANNELS,
            "KERNEL_STR": self.KERNEL_STR,
            "PADDING": self.op.PADDING,
            "FIRE_RATE": self.FIRE_RATE,
            "KAN_AUX": {
                key: getattr(value, "__name__", "callable") if callable(value) else value
                for key, value in self.KAN_AUX.items()
            },
        }

    def get_weights(self):  # type: ignore
        diff_self, _ = self.partition()
        weights, _tree_def = jax.tree_util.tree_flatten(diff_self)
        return list(map(jnp.squeeze, weights))

    def get_edge_norms(self):
        return [layer.layer.edge_norms() for layer in self.layers]

    def evaluate_edge_functions(self, xs):
        return [layer.layer.evaluate_edge_functions(xs) for layer in self.layers]

    def get_kan_layer_inputs_outputs(self, x):
        """Return diagnostic KAN layer input/output tensors for one NCA state.

        This intentionally avoids fire-rate masking and boundary callbacks. It
        exposes the deterministic update rule internals used by the KAN layers.
        """
        if x.ndim != 3 or x.shape[0] != self.N_CHANNELS:
            raise ValueError(
                "FastKaNCA KAN diagnostics expect one state of shape "
                f"({self.N_CHANNELS}, x, y), got {x.shape}."
            )
        layer0_input = self.perception(x)
        layer0_output = self.layers[0](layer0_input)
        layer1_input = layer0_output
        layer1_output = self.layers[1](layer1_input)
        return (
            (layer0_input, layer0_output),
            (layer1_input, layer1_output),
        )

    def get_top_edges(self, k: int, layer_index: int = 0):
        top_edges = get_top_edges_from_layer(self.layers[layer_index].layer, k)
        if layer_index != 0:
            return top_edges

        feature_names = self.get_feature_names()
        return [
            {
                **edge,
                "input_name": feature_names[edge["input_index"]],
                "output_name": f"hidden_{edge['output_index']}",
            }
            for edge in top_edges
        ]

    def get_feature_names(self):
        names = []
        for op_name in ["ID", "DIFF", "GRAD", "AV", "LAP"]:
            if op_name not in self.KERNEL_STR:
                continue
            if op_name == "GRAD":
                names.extend(
                    [
                        f"channel_{channel}_GRAD_x"
                        for channel in range(self.N_CHANNELS)
                    ]
                )
                names.extend(
                    [
                        f"channel_{channel}_GRAD_y"
                        for channel in range(self.N_CHANNELS)
                    ]
                )
            else:
                names.extend(
                    [
                        f"channel_{channel}_{op_name}"
                        for channel in range(self.N_CHANNELS)
                    ]
                )
        return names
