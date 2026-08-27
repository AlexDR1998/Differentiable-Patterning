import jax.numpy as jnp
import pytest

from Common.trainer.variation_metrics import replicate_variation_metrics


def _replicates():
    base = jnp.zeros((3, 2, 5, 5))
    base = base.at[0, 0, 1:4, 1:4].set(0.2)
    base = base.at[1, 0, 1:4, 1:4].set(0.5)
    base = base.at[2, 0, 1:4, 1:4].set(0.9)
    base = base.at[:, 1, 2:4, 2:4].set(jnp.asarray([0.1, 0.6, 0.3])[:, None, None])
    return base


def test_identical_replicate_populations_have_matching_variation():
    target = _replicates()
    metrics = replicate_variation_metrics(
        target, target, jnp.ones((5, 5), dtype=bool), radial_bins=4
    )

    assert float(metrics["dispersion_ratio"]) == pytest.approx(1.0)
    assert float(metrics["pairwise_distance_correlation"]) == pytest.approx(1.0)
    assert float(metrics["energy_distance"]) == pytest.approx(0.0, abs=1e-6)


def test_collapsed_predictions_report_missing_replicate_dispersion():
    target = _replicates()
    collapsed = jnp.broadcast_to(target.mean(axis=0), target.shape)
    metrics = replicate_variation_metrics(
        collapsed, target, jnp.ones((5, 5), dtype=bool), radial_bins=4
    )

    assert float(metrics["dispersion_ratio"]) == pytest.approx(0.0, abs=1e-6)
    assert float(metrics["pairwise_distance_correlation"]) == pytest.approx(0.0)
    assert float(metrics["energy_distance"]) > 0.0


def test_variation_metrics_require_multiple_replicates():
    target = _replicates()[:1]
    with pytest.raises(ValueError, match="at least two replicates"):
        replicate_variation_metrics(target, target, jnp.ones((5, 5)))
