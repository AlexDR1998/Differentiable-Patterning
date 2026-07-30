import jax
import jax.numpy as jnp
import pytest

from NCA.model.NCA_model_fast import NCA as FastNCA
from NCA.model.sycl.reference import jax_nca_forward


FLAGS = {"ID": 1, "DIFF": 2, "GRAD": 4, "AV": 8, "LAP": 16}
PADDING = {"ZEROS": 0, "REFLECT": 1, "REPLICATE": 2, "CIRCULAR": 3}


def _model_operands(model, state, key):
    kernels = jnp.concatenate(
        (
            model.op.grad_x.weight,
            model.op.grad_y.weight,
            model.op.average.weight,
            model.op.laplacian.weight,
        ),
        axis=0,
    )[:, 0]
    return (
        state,
        kernels,
        model.layers[0].weight[:, :, 0, 0],
        model.layers[2].weight[:, :, 0, 0],
        model.layers[2].bias.reshape(model.N_CHANNELS),
        jax.random.bernoulli(
            key, p=model.FIRE_RATE, shape=state.shape
        ).astype(state.dtype),
    )


@pytest.mark.parametrize("padding", list(PADDING))
def test_sycl_backward_reference_matches_fast_nca(padding):
    kernel_str = ["ID", "LAP", "DIFF"]
    keys = jax.random.split(jax.random.PRNGKey(12), 6)
    model = FastNCA(
        3,
        KERNEL_STR=kernel_str,
        PADDING=padding,
        FIRE_RATE=0.6,
        key=keys[0],
    )
    model.set_weights(
        [
            0.1 * jax.random.normal(keys[1], model.layers[0].weight.shape),
            0.1 * jax.random.normal(keys[2], model.layers[2].weight.shape),
            0.1 * jax.random.normal(keys[3], model.layers[2].bias.shape),
        ]
    )
    state = jax.random.normal(keys[4], (3, 7, 8))

    expected = model(state, key=keys[5])
    actual = jax_nca_forward(
        *_model_operands(model, state, keys[5]),
        kernel_flags=sum(FLAGS[name] for name in kernel_str),
        padding=PADDING[padding],
    )

    assert jnp.allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_sycl_backward_reference_has_finite_parameter_and_state_gradients():
    kernel_str = ["ID", "DIFF", "LAP"]
    model = FastNCA(
        2,
        KERNEL_STR=kernel_str,
        PADDING="CIRCULAR",
        FIRE_RATE=1.0,
        key=jax.random.PRNGKey(1),
    )
    model.set_weights(
        [
            0.1
            * jax.random.normal(
                jax.random.PRNGKey(4), model.layers[0].weight.shape
            ),
            0.1
            * jax.random.normal(
                jax.random.PRNGKey(5), model.layers[2].weight.shape
            ),
            0.1
            * jax.random.normal(
                jax.random.PRNGKey(6), model.layers[2].bias.shape
            ),
        ]
    )
    # Circular padding makes both spatial gradients exactly zero while the ID
    # feature still sends a nonzero cotangent through the perception MLP.
    state = jnp.ones((2, 5, 6), dtype=jnp.float32)
    operands = _model_operands(model, state, jax.random.PRNGKey(3))

    def loss(state_value, hidden_weight, output_weight, output_bias):
        output = jax_nca_forward(
            state_value,
            operands[1],
            hidden_weight,
            output_weight,
            output_bias,
            operands[5],
            kernel_flags=sum(FLAGS[name] for name in kernel_str),
            padding=PADDING["CIRCULAR"],
        )
        return jnp.sum(output**2)

    differentiable = (operands[0], operands[2], operands[3], operands[4])
    gradients = jax.grad(loss, argnums=(0, 1, 2, 3))(*differentiable)

    for gradient, value in zip(gradients, differentiable):
        assert gradient.shape == value.shape
        assert jnp.all(jnp.isfinite(gradient))
