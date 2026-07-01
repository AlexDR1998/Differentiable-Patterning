import time
from typing import Callable, List, Optional, Union

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def identity(x):
    return x


def resolve_base_activation(base_activation: Union[str, Callable, None]):
    if base_activation is None or base_activation == "none":
        return None, "none"
    if base_activation == "identity":
        return identity, "identity"
    if base_activation == "silu":
        return jax.nn.silu, "silu"
    if base_activation == "relu":
        return jax.nn.relu, "relu"
    if callable(base_activation):
        return base_activation, getattr(base_activation, "__name__", "callable")
    raise ValueError(
        "base_activation must be one of 'identity', 'silu', 'relu', 'none', None, "
        "or a callable."
    )


class FastRBFKANLayer(eqx.Module):
    """Vectorised Gaussian RBF KAN layer for vector inputs.

    This keeps one univariate RBF expansion per input-output edge, but computes
    all basis values once per input feature and projects with einsum.
    """

    in_features: int
    out_features: int
    num_basis: int
    grid_min: float
    grid_max: float
    rbf_width: float
    trainable_width: bool
    use_base_branch: bool
    use_layernorm: bool
    base_activation: Optional[Callable]
    base_activation_name: str
    spline_weight: Array
    base_weight: Optional[Array]
    base_bias: Optional[Array]
    log_rbf_width: Optional[Array]
    layernorm: Optional[eqx.nn.LayerNorm]
    extrapolation: str

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        num_basis: int = 8,
        grid_min: float = -2.0,
        grid_max: float = 2.0,
        rbf_width: Optional[float] = None,
        trainable_width: bool = True,
        use_base_branch: bool = True,
        base_activation: Union[str, Callable, None] = "silu",
        use_layernorm: bool = True,
        spline_init_scale: float = 0.1,
        base_init_scale: float = 0.1,
        final_zero_init: bool = False,
        extrapolation: str = "constant",
        key=None,
    ):
        if key is None:
            key = jax.random.PRNGKey(int(time.time()))
        key_spline, key_base = jax.random.split(key, 2)

        if rbf_width is None:
            rbf_width = (grid_max - grid_min) / max(num_basis - 1, 1)

        self.in_features = in_features
        self.out_features = out_features
        self.num_basis = num_basis
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.rbf_width = float(rbf_width)
        self.trainable_width = trainable_width
        self.extrapolation = extrapolation
        resolved_base_activation, base_activation_name = resolve_base_activation(
            base_activation
        )
        self.use_base_branch = use_base_branch and resolved_base_activation is not None
        self.use_layernorm = use_layernorm
        self.base_activation = resolved_base_activation
        self.base_activation_name = base_activation_name

        spline_scale = 0.0 if final_zero_init else spline_init_scale
        base_scale = 0.0 if final_zero_init else base_init_scale
        self.spline_weight = spline_scale * jax.random.normal(
            key_spline, shape=(in_features, out_features, num_basis)
        )

        if self.use_base_branch:
            self.base_weight = base_scale * jax.random.normal(
                key_base, shape=(out_features, in_features)
            )
            self.base_bias = jnp.zeros((out_features,))
        else:
            self.base_weight = None
            self.base_bias = None

        if trainable_width:
            self.log_rbf_width = jnp.array(jnp.log(self.rbf_width))
        else:
            self.log_rbf_width = None

        if use_layernorm:
            self.layernorm = eqx.nn.LayerNorm(in_features)
        else:
            self.layernorm = None

    def _width(self):
        if self.log_rbf_width is None:
            return self.rbf_width
        return jnp.exp(self.log_rbf_width)

    def _validate_vector_input(self, x):
        if x.ndim != 1 or x.shape[0] != self.in_features:
            raise ValueError(
                "FastRBFKANLayer expects a vector input of shape "
                f"({self.in_features},), got {x.shape}. Use "
                "SpatialFastRBFKANLayer for channel-first image tensors."
            )

    def basis(
        self, x: Float[Array, "{self.in_features}"]
    ) -> Float[Array, "{self.in_features} {self.num_basis}"]:
        self._validate_vector_input(x)
        if self.layernorm is not None:
            x = self.layernorm(x)
        grid = jnp.linspace(self.grid_min, self.grid_max, self.num_basis)
        return jnp.exp(-((x[:, None] - grid[None, :]) / self._width()) ** 2)

    def __call__(
        self, x: Float[Array, "{self.in_features}"], key=None
    ) -> Float[Array, "{self.out_features}"]:
        basis = self.basis(x)
        y = jnp.einsum("ik,iok->o", basis, self.spline_weight)
        if self.base_weight is not None and self.base_activation is not None:
            base = self.base_weight @ self.base_activation(x)
            if self.base_bias is not None:
                base = base + self.base_bias
            y = y + base
        return y

    def edge_norms(self) -> Float[Array, "{self.in_features} {self.out_features}"]:
        spline_norm = jnp.linalg.norm(self.spline_weight, axis=-1)
        if self.base_weight is None:
            return spline_norm
        return jnp.sqrt(spline_norm**2 + self.base_weight.T**2)

    def evaluate_edge_functions(
        self, xs: Float[Array, "samples"]
    ) -> Float[Array, "{self.in_features} {self.out_features} samples"]:
        """Evaluate complete edge functions over a 1D grid.

        If layernorm is enabled, these are functions of the post-normalisation
        scalar coordinate used by the RBF branch.
        """
        grid = jnp.linspace(self.grid_min, self.grid_max, self.num_basis)
        basis = jnp.exp(-((xs[:, None] - grid[None, :]) / self._width()) ** 2)
        edge_values = jnp.einsum("sk,iok->ios", basis, self.spline_weight)
        if self.base_weight is not None and self.base_activation is not None:
            base_values = self.base_weight.T[:, :, None] * self.base_activation(xs)
            edge_values = edge_values + base_values
        return edge_values

    def zero_init(self):
        weight_where = lambda layer: layer.spline_weight
        zeroed = eqx.tree_at(weight_where, self, jnp.zeros_like(self.spline_weight))
        if self.base_weight is not None:
            base_weight_where = lambda layer: layer.base_weight
            zeroed = eqx.tree_at(
                base_weight_where, zeroed, jnp.zeros_like(self.base_weight)
            )
        if self.base_bias is not None:
            base_bias_where = lambda layer: layer.base_bias
            zeroed = eqx.tree_at(base_bias_where, zeroed, jnp.zeros_like(self.base_bias))
        return zeroed


class FastLinearSplineKANLayer(FastRBFKANLayer):
    """Vectorised fixed-grid piecewise-linear KAN layer for vector inputs."""

    def __init__(self, *args, extrapolation: str = "constant", **kwargs):
        if extrapolation not in {"constant", "zero", "linear"}:
            raise ValueError(
                "extrapolation must be one of 'constant', 'zero', or 'linear'."
            )
        kwargs["trainable_width"] = False
        super().__init__(*args, extrapolation=extrapolation, **kwargs)
        if self.num_basis < 2:
            raise ValueError("FastLinearSplineKANLayer requires at least two knots.")

    def basis(
        self, x: Float[Array, "{self.in_features}"]
    ) -> Float[Array, "{self.in_features} {self.num_basis}"]:
        self._validate_vector_input(x)
        if self.layernorm is not None:
            x = self.layernorm(x)
        return self._linear_spline_basis(x)

    def _linear_spline_basis(self, x):
        grid = jnp.linspace(self.grid_min, self.grid_max, self.num_basis)
        step = (self.grid_max - self.grid_min) / max(self.num_basis - 1, 1)
        if self.num_basis == 1:
            return jnp.ones((*x.shape, 1))

        if self.extrapolation == "linear":
            left_t = (x - grid[0]) / step
            right_t = (x - grid[-2]) / step
            interior = jnp.maximum(1.0 - jnp.abs(x[..., None] - grid[None, :]) / step, 0.0)
            interior = interior.at[..., 0].set(
                jnp.where(x < self.grid_min, 1.0 - left_t, interior[..., 0])
            )
            interior = interior.at[..., 1].set(
                jnp.where(x < self.grid_min, left_t, interior[..., 1])
            )
            interior = interior.at[..., -2].set(
                jnp.where(x > self.grid_max, 1.0 - right_t, interior[..., -2])
            )
            interior = interior.at[..., -1].set(
                jnp.where(x > self.grid_max, right_t, interior[..., -1])
            )
            return interior

        x_clipped = jnp.clip(x, self.grid_min, self.grid_max)
        basis = jnp.maximum(1.0 - jnp.abs(x_clipped[..., None] - grid[None, :]) / step, 0.0)
        if self.extrapolation == "zero":
            in_grid = (x >= self.grid_min) & (x <= self.grid_max)
            basis = basis * in_grid[..., None]
        return basis

    def evaluate_edge_functions(
        self, xs: Float[Array, "samples"]
    ) -> Float[Array, "{self.in_features} {self.out_features} samples"]:
        basis = self._linear_spline_basis(xs)
        edge_values = jnp.einsum("sk,iok->ios", basis, self.spline_weight)
        if self.base_weight is not None and self.base_activation is not None:
            base_values = self.base_weight.T[:, :, None] * self.base_activation(xs)
            edge_values = edge_values + base_values
        return edge_values


class FastRBFKAN(eqx.Module):
    layers: list

    def __init__(
        self,
        layers_hidden: List[int],
        *,
        num_basis: int = 8,
        grid_min: float = -2.0,
        grid_max: float = 2.0,
        rbf_width: Optional[float] = None,
        trainable_width: bool = True,
        use_base_branch: bool = True,
        base_activation: Union[str, Callable, None] = "silu",
        use_layernorm: bool = True,
        spline_init_scale: float = 0.1,
        base_init_scale: float = 0.1,
        final_zero_init: bool = False,
        key=None,
    ):
        if len(layers_hidden) < 2:
            raise ValueError(
                "FastRBFKAN requires at least [in_features, out_features]."
            )
        if key is None:
            key = jax.random.PRNGKey(int(time.time()))
        keys = jax.random.split(key, len(layers_hidden) - 1)
        self.layers = [
            FastRBFKANLayer(
                in_features,
                out_features,
                num_basis=num_basis,
                grid_min=grid_min,
                grid_max=grid_max,
                rbf_width=rbf_width,
                trainable_width=trainable_width,
                use_base_branch=use_base_branch,
                base_activation=base_activation,
                use_layernorm=use_layernorm,
                spline_init_scale=spline_init_scale,
                base_init_scale=base_init_scale,
                final_zero_init=final_zero_init and i == len(layers_hidden) - 2,
                key=keys[i],
            )
            for i, (in_features, out_features) in enumerate(
                zip(layers_hidden[:-1], layers_hidden[1:])
            )
        ]

    def __call__(self, x, key=None):
        for layer in self.layers:
            x = layer(x)
        return x

    def get_edge_norms(self):
        return [layer.edge_norms() for layer in self.layers]

    def evaluate_edge_functions(self, xs):
        return [layer.evaluate_edge_functions(xs) for layer in self.layers]

    def get_top_edges(self, k: int, layer_index: int = 0):
        return get_top_edges_from_layer(self.layers[layer_index], k)


def get_top_edges_from_layer(layer: FastRBFKANLayer, k: int):
    edge_norms = jax.device_get(layer.edge_norms())
    flat_norms = jnp.ravel(edge_norms)
    k = min(k, flat_norms.shape[0])
    order = jnp.argsort(flat_norms)[::-1][:k]
    input_indices, output_indices = jnp.unravel_index(order, edge_norms.shape)
    return [
        {
            "rank": int(rank),
            "input_index": int(input_index),
            "output_index": int(output_index),
            "norm": float(flat_norms[flat_index]),
        }
        for rank, (flat_index, input_index, output_index) in enumerate(
            zip(order, input_indices, output_indices), start=1
        )
    ]


def plot_fast_rbf_kan_edges(
    kan: FastRBFKAN,
    *,
    layer_index: int = 0,
    input_indices: Optional[List[int]] = None,
    output_indices: Optional[List[int]] = None,
    edge_indices: Optional[List[tuple]] = None,
    xs=None,
    max_edges: int = 32,
    ax=None,
):
    """Plot selected learned 1D edge functions from a FastRBFKAN.

    This helper imports matplotlib lazily so the model module remains usable in
    non-plotting environments.
    """
    import matplotlib.pyplot as plt

    layer = kan.layers[layer_index]
    if xs is None:
        xs = jnp.linspace(layer.grid_min, layer.grid_max, 200)
    edge_values = layer.evaluate_edge_functions(xs)

    if input_indices is None:
        input_indices = list(range(layer.in_features))
    if output_indices is None:
        output_indices = list(range(layer.out_features))

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    if edge_indices is None:
        edge_indices = [
            (input_index, output_index)
            for output_index in output_indices
            for input_index in input_indices
        ]

    for plotted, (input_index, output_index) in enumerate(edge_indices):
        if plotted >= max_edges:
            break
        ax.plot(
            xs,
            edge_values[input_index, output_index],
            label=f"in {input_index} -> out {output_index}",
            alpha=0.8,
        )

    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
    ax.set_xlabel("Input value")
    ax.set_ylabel("Edge contribution")
    ax.set_title(f"FastRBFKAN layer {layer_index} edge functions")
    ax.legend(fontsize="small", ncols=2)
    return ax.figure


def plot_top_fast_rbf_kan_edges(
    kan: FastRBFKAN,
    *,
    k: int = 12,
    layer_index: int = 0,
    xs=None,
    ax=None,
):
    top_edges = kan.get_top_edges(k=k, layer_index=layer_index)
    edge_indices = [
        (edge["input_index"], edge["output_index"]) for edge in top_edges
    ]
    return plot_fast_rbf_kan_edges(
        kan,
        layer_index=layer_index,
        edge_indices=edge_indices,
        xs=xs,
        max_edges=k,
        ax=ax,
    )


def plot_fast_rbf_kan_edge_norms(
    kan: FastRBFKAN,
    *,
    layer_index: int = 0,
    ax=None,
):
    import matplotlib.pyplot as plt

    edge_norms = kan.get_edge_norms()[layer_index]
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(edge_norms.T, aspect="auto", origin="lower")
    ax.set_xlabel("Input index")
    ax.set_ylabel("Output index")
    ax.set_title(f"FastRBFKAN layer {layer_index} edge norms")
    ax.figure.colorbar(image, ax=ax, label="Edge norm")
    return ax.figure
