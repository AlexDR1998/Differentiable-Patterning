import equinox as eqx
import jax
import jax.numpy as jnp

from NCA.inverse_design.geometry import FourierRadialGeometry, coordinate_grid
from NCA.inverse_design.initial_state import (
    PatchSamplingConfig,
    build_initial_state,
    sample_circular_patches,
)
from NCA.inverse_design.objectives import CellTypeObjective, GeometryPenalties
from NCA.inverse_design.optimise import OptimisationConfig, optimise_geometry


def test_fourier_geometry_has_gradients_for_every_parameter():
    geometry = FourierRadialGeometry(
        3,
        radius=0.6,
        key=jax.random.PRNGKey(0),
        initial_scale=0.1,
        angle=0.2,
    )
    grid = coordinate_grid((21, 19))

    gradients = jax.grad(
        lambda candidate: jnp.sum(
            candidate.occupancy(grid, 0.06)
            * jnp.linspace(0.5, 1.5, 21)[:, None]
        )
    )(geometry)

    assert jnp.isfinite(gradients.raw_radius)
    assert jnp.isfinite(gradients.raw_angle)
    assert jnp.all(jnp.isfinite(gradients.raw_cos))
    assert jnp.all(jnp.isfinite(gradients.raw_sin))
    assert gradients.raw_cos.shape == (3,)


def test_patch_sampling_is_key_driven_and_stops_gradients():
    reference = jnp.arange(2 * 15 * 15, dtype=jnp.float32).reshape(2, 15, 15)
    reference_mask = jnp.ones((15, 15), dtype=bool)
    config = PatchSamplingConfig(5, ((0, 1),))

    sample_a = sample_circular_patches(
        reference, reference_mask, (17, 13), jax.random.PRNGKey(4), config
    )
    sample_b = sample_circular_patches(
        reference, reference_mask, (17, 13), jax.random.PRNGKey(4), config
    )
    sample_c = sample_circular_patches(
        reference, reference_mask, (17, 13), jax.random.PRNGKey(5), config
    )
    gradient = jax.grad(
        lambda values: jnp.sum(
            sample_circular_patches(
                values, reference_mask, (10, 10), jax.random.PRNGKey(4), config
            )
        )
    )(reference)

    assert jnp.array_equal(sample_a, sample_b)
    assert not jnp.array_equal(sample_a, sample_c)
    assert jnp.all(gradient == 0)


def test_initial_state_remains_differentiable_through_occupancy():
    biology = jnp.ones((2, 7, 7))
    occupancy = jnp.full((7, 7), 0.4)

    gradient = jax.grad(
        lambda mask: jnp.sum(build_initial_state(biology, mask, 4, boundary_mode="soft"))
    )(occupancy)

    # Two biological channels plus the recurrent boundary channel.
    assert jnp.allclose(gradient, 3.0)


class _IdentityNCA(eqx.Module):
    N_CHANNELS: int = eqx.field(static=True, default=3)

    def __call__(self, state, boundary_callback, key):
        del key
        return boundary_callback(state)


def test_optimiser_updates_fourier_geometry():
    initial = FourierRadialGeometry(2, radius=0.45)
    biology = jnp.ones((1, 15, 15), dtype=jnp.float32)

    result = optimise_geometry(
        _IdentityNCA(),
        initial,
        lambda key: biology,
        (15, 15),
        1,
        jax.random.PRNGKey(8),
        CellTypeObjective((0,), (0.2,), (1,), temperature=0.05),
        OptimisationConfig(
            iterations=2,
            learning_rate=1.0e-2,
            total_steps=1,
            mask_softness=0.05,
        ),
        GeometryPenalties(target_area_fraction=0.3, area_weight=1.0),
    )

    assert result.final_state.shape == (3, 15, 15)
    assert result.loss_history.shape == (2,)
    assert not jnp.array_equal(result.geometry.raw_radius, initial.raw_radius)
