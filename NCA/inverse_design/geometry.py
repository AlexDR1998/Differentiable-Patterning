"""Low-dimensional, differentiable Fourier-radial geometries."""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array


def coordinate_grid(shape: tuple[int, int], dtype=jnp.float32) -> tuple[Array, Array]:
    """Return ``(x, y)`` coordinates spanning ``[-1, 1]``."""

    height, width = shape
    y = jnp.linspace(-1.0, 1.0, height, dtype=dtype)
    x = jnp.linspace(-1.0, 1.0, width, dtype=dtype)
    return jnp.meshgrid(x, y, indexing="xy")


class FourierRadialGeometry(eqx.Module):
    """Star-convex boundary with a Fourier series in log-radius.

    Every array leaf is an unconstrained differentiable parameter. Radius,
    angle and harmonic amplitudes are transformed only when rendering.
    """

    raw_radius: Array
    raw_cos: Array
    raw_sin: Array
    raw_angle: Array
    radius_range: tuple[float, float] = eqx.field(static=True)
    coefficient_limit: float = eqx.field(static=True)

    def __init__(
        self,
        harmonics: int,
        *,
        radius: float = 0.65,
        radius_range: tuple[float, float] = (0.1, 0.95),
        coefficient_limit: float = 0.35,
        angle: float = 0.0,
        key=None,
        initial_scale: float = 0.0,
    ):
        if harmonics < 1:
            raise ValueError("harmonics must be positive")
        lower, upper = radius_range
        if not lower < radius < upper:
            raise ValueError("radius must lie strictly inside radius_range")
        if coefficient_limit <= 0:
            raise ValueError("coefficient_limit must be positive")
        fraction = (radius - lower) / (upper - lower)
        self.raw_radius = jnp.asarray(jnp.log(fraction) - jnp.log1p(-fraction))
        if key is None or initial_scale == 0.0:
            raw_cos = jnp.zeros(harmonics)
            raw_sin = jnp.zeros(harmonics)
        else:
            cos_key, sin_key = jax.random.split(key)
            raw_cos = initial_scale * jax.random.normal(cos_key, (harmonics,))
            raw_sin = initial_scale * jax.random.normal(sin_key, (harmonics,))
        self.raw_cos = raw_cos
        self.raw_sin = raw_sin
        self.raw_angle = jnp.asarray(jnp.arctanh(jnp.clip(angle / jnp.pi, -0.999, 0.999)))
        self.radius_range = radius_range
        self.coefficient_limit = coefficient_limit

    @property
    def radius(self) -> Array:
        lower, upper = self.radius_range
        return lower + (upper - lower) * jax.nn.sigmoid(self.raw_radius)

    @property
    def coefficients_cos(self) -> Array:
        return self.coefficient_limit * jnp.tanh(self.raw_cos)

    @property
    def coefficients_sin(self) -> Array:
        return self.coefficient_limit * jnp.tanh(self.raw_sin)

    @property
    def angle(self) -> Array:
        return jnp.pi * jnp.tanh(self.raw_angle)

    def radial_boundary(self, theta: Array) -> Array:
        relative_theta = theta - self.angle
        harmonics = jnp.arange(1, self.raw_cos.size + 1, dtype=theta.dtype)
        phase = harmonics[:, None, None] * relative_theta[None]
        log_radius = jnp.sum(
            self.coefficients_cos[:, None, None] * jnp.cos(phase)
            + self.coefficients_sin[:, None, None] * jnp.sin(phase),
            axis=0,
        )
        radius = self.radius * jnp.exp(log_radius)
        # RMS radial normalisation makes pi * radius**2 the continuous area.
        return radius * self.radius / jnp.sqrt(jnp.mean(jnp.square(radius)))

    def level_set(self, grid: tuple[Array, Array]) -> Array:
        x, y = grid
        theta = jnp.arctan2(y, x)
        distance = jnp.sqrt(jnp.square(x) + jnp.square(y))
        boundary = self.radial_boundary(theta)
        return distance / jnp.maximum(boundary, 1.0e-6) - 1.0

    def occupancy(self, grid: tuple[Array, Array], softness: float | Array) -> Array:
        softness = jnp.maximum(jnp.asarray(softness), 1.0e-4)
        return jax.nn.sigmoid(-self.level_set(grid) / softness)
