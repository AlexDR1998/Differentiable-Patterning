import numpy as np

from Experiments.micropatterns.texture_synthesis import (
    normalise_histograms_and_covariance,
    synthesise_patch_texture,
    synthesise_reference_texture,
)


def _reference(size=32, channels=3, seed=0):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[-1:1:complex(size), -1:1:complex(size)]
    support = x**2 + y**2 <= 0.8**2
    reference = rng.uniform(0.2, 0.9, size=(channels, size, size)).astype(np.float32)
    return reference * support[None], support


def test_patch_texture_shape_and_seed():
    reference, support = _reference()
    groups = ((0, 1), (2,))
    first, patch_size, _, _ = synthesise_patch_texture(
        reference, support, (24, 20), groups, 7, 3, 0.9, 1, 4, seed=3
    )
    second, *_ = synthesise_patch_texture(
        reference, support, (24, 20), groups, 7, 3, 0.9, 1, 4, seed=3
    )
    assert first.shape == (3, 24, 20)
    assert patch_size % 2 == 1
    np.testing.assert_array_equal(first, second)
    # Patches come from inside the colony, so the texture has no background zeros.
    inside = reference[:, support]
    assert first.min() >= inside.min() - 1e-5
    assert first.max() <= inside.max() + 1e-5


def test_patch_texture_keeps_co_measured_channels_together():
    reference, support = _reference()
    reference[1] = 2.0 * reference[0]
    texture, *_ = synthesise_patch_texture(
        reference, support, (20, 20), ((0, 1), (2,)), 5, 2, 0.9, 0, 2, seed=0
    )
    np.testing.assert_allclose(texture[1], 2.0 * texture[0], rtol=1e-5, atol=1e-6)


def test_reference_texture_matches_reference_values_in_mask():
    reference, support = _reference()
    output_mask = np.ones((16, 16), dtype=bool)
    texture = synthesise_reference_texture(
        reference, support, (16, 16), output_mask, iterations=2, seed=1
    )
    assert texture.shape == (3, 16, 16)
    for channel in range(3):
        samples = reference[channel][support]
        assert texture[channel].min() >= samples.min() - 1e-5
        assert texture[channel].max() <= samples.max() + 1e-5


def test_normalisation_keeps_shape_and_range():
    reference, support = _reference()
    texture, *_ = synthesise_patch_texture(
        reference, support, (16, 16), ((0, 1), (2,)), 5, 2, 0.9, 0, 2, seed=0
    )
    output_mask = np.ones((16, 16), dtype=bool)
    normalised = normalise_histograms_and_covariance(
        texture, reference, support, output_mask, ((0, 1), (2,)), iterations=2
    )
    assert normalised.shape == texture.shape
    samples = reference[:, support]
    assert normalised.min() >= samples.min() - 1e-5
    assert normalised.max() <= samples.max() + 1e-5
