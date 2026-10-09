"""Restored PDE right-hand sides, initial conditions and solver."""

import inspect
from importlib import import_module

import jax.numpy as jnp
import jax.random as jr
import pytest

from PDE.catalogue import INITIAL_CONDITIONS, PDE_MODELS, build_pde
from PDE.initial_conditions import make_initial_condition
from PDE.model.solver.semidiscrete_solver import PDE_solver

SIZE = 16


@pytest.mark.parametrize("name", sorted(PDE_MODELS))
def test_catalogue_parameters_match_model_constructors(name):
    spec = PDE_MODELS[name]
    signature = inspect.signature(import_module(spec.module).F.__init__)
    accepted = set(signature.parameters) - {"self", "PADDING", "dx", "KERNEL_SCALE"}
    assert set(spec.parameter_names) == accepted


@pytest.mark.parametrize("name", sorted(PDE_MODELS))
def test_right_hand_side_is_finite_with_state_shape(name):
    channels = len(PDE_MODELS[name].channel_names)
    X = jr.uniform(jr.PRNGKey(0), (channels, SIZE, SIZE))
    dX = build_pde(name, "CIRCULAR", 1.0)(0.0, X, None)
    assert dX.shape == X.shape
    assert jnp.all(jnp.isfinite(dX))


def test_unknown_parameter_is_rejected():
    with pytest.raises(ValueError, match="Unknown parameters"):
        build_pde("heat", "CIRCULAR", 1.0, {"kill_rate": 0.1})


def test_heat_equation_conserves_mass_with_periodic_boundaries():
    X = jr.uniform(jr.PRNGKey(1), (1, SIZE, SIZE))
    dX = build_pde("heat", "CIRCULAR", 1.0)(0.0, X, None)
    assert jnp.abs(jnp.sum(dX)) < 1e-4


def test_gray_scott_homogeneous_steady_state():
    X = jnp.concatenate((jnp.ones((1, SIZE, SIZE)), jnp.zeros((1, SIZE, SIZE))))
    dX = build_pde("gray_scott", "CIRCULAR", 1.0)(0.0, X, None)
    assert jnp.allclose(dX, 0.0)


def test_solver_returns_state_at_each_time():
    func = build_pde("heat", "CIRCULAR", 1.0)
    x0 = jr.uniform(jr.PRNGKey(2), (1, SIZE, SIZE))
    ts, ys = PDE_solver(func, dt=0.1)(jnp.linspace(0.0, 1.0, 4), x0)
    assert ys.shape == (4, 1, SIZE, SIZE)
    assert jnp.allclose(ys[0], x0)


def test_solver_rejects_unknown_method():
    with pytest.raises(ValueError, match="Unknown solver"):
        PDE_solver(build_pde("heat", "CIRCULAR", 1.0), SOLVER="rk99")


@pytest.mark.parametrize("name", INITIAL_CONDITIONS)
def test_initial_conditions_have_requested_shape(name):
    x0 = make_initial_condition(name, jr.PRNGKey(3), 3, 2, SIZE, blur_passes=1)
    assert x0.shape == (3, 2, SIZE, SIZE)
    assert jnp.all(jnp.isfinite(x0))


def test_gray_scott_initial_conditions_need_two_channels():
    with pytest.raises(ValueError, match="2-channel"):
        make_initial_condition("gray_scott_spots", jr.PRNGKey(0), 1, 1, SIZE)
