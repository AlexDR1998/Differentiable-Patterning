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


def test_layer_cell_counts_count_cells_cut_by_the_mask_in_part():
    from Experiments.micropatterns.texture_synthesis import layer_cell_counts

    ranks = np.full((2, 2, 10, 10), -1.0, dtype=np.float32)
    ranks[0, :, 0:2, 0:2] = 0.1  # inside
    ranks[0, :, 4:6, 4:8] = 0.7  # half inside
    ranks[0, :, 8:10, 8:10] = 0.4  # outside
    ranks[1, :, 0:3, 0:3] = 0.2  # inside
    mask = np.zeros((10, 10), dtype=bool)
    mask[:, :6] = True

    np.testing.assert_allclose(layer_cell_counts(ranks, mask), [1.5, 1.0])


def test_count_nuclei_finds_separate_and_touching_nuclei():
    from Experiments.micropatterns.cell_counts import count_nuclei

    rows, columns = np.mgrid[:80, :80]
    centres = [(15, 15), (15, 50), (50, 20), (55, 55), (55, 66)]  # the last two touch
    image = np.zeros((80, 80), dtype=np.float32)
    for row, column in centres:
        image += ((rows - row) ** 2 + (columns - column) ** 2 <= 6**2).astype(np.float32)
    image = np.minimum(image, 1.0)

    count, labels, area = count_nuclei(image, np.ones((80, 80), dtype=bool), 12)

    assert count == len(centres)
    assert labels.max() > 0
    assert 0.5 * np.pi * 36 < area < 1.5 * np.pi * 36


def test_count_nuclei_seed_spacing_and_threshold():
    from Experiments.micropatterns.cell_counts import count_nuclei

    rows, columns = np.mgrid[:80, :80]
    image = np.zeros((80, 80), dtype=np.float32)
    for row, column, brightness in [(40, 30, 1.0), (40, 41, 1.0), (15, 15, 1.0), (65, 65, 0.45)]:
        image = np.maximum(image, brightness * ((rows - row) ** 2 + (columns - column) ** 2 <= 6**2))
    mask = np.ones((80, 80), dtype=bool)

    # The touching pair (centres 11 px apart) is split by default, merged with widely spaced seeds
    assert count_nuclei(image, mask, 12)[0] == 4
    assert count_nuclei(image, mask, 12, seed_spacing=1.2)[0] == 3
    # A higher stain threshold drops the dim nucleus
    assert count_nuclei(image, mask, 12, threshold_scale=1.0)[0] == 4
    assert count_nuclei(image, mask, 12, threshold_scale=1.6)[0] == 3


def _rings(centres, radius=7.0, width=2.0, size=80):
    """A nuclear rim stain (as LMBR): a bright ring per nucleus."""
    rows, columns = np.mgrid[:size, :size]
    image = np.zeros((size, size), dtype=np.float32)
    for row, column in centres:
        distance = np.hypot(rows - row, columns - column)
        image = np.maximum(image, (np.abs(distance - radius) <= width / 2).astype(np.float32))
    return image


def test_count_nuclei_rim_splits_touching_nuclei_along_their_rim():
    from Experiments.micropatterns.cell_counts import count_nuclei

    mask = np.ones((80, 80), dtype=bool)
    # Two rings pressed together (shared rim, no narrowing), and a separate one
    pressed = _rings([(40, 26), (40, 40), (15, 60)])
    assert count_nuclei(pressed, mask, 16, seed_spacing=1.0, split_method="rim")[0] == 3
    # Three rings around a small enclosed gap: the gap is not a nucleus
    triangle = _rings([(30, 30), (30, 44), (42, 37)])
    assert count_nuclei(triangle, mask, 16, split_method="rim")[0] == 3


def test_count_nuclei_roundness_and_intensity_split_merged_nuclei():
    from Experiments.micropatterns.cell_counts import count_nuclei

    rows, columns = np.mgrid[:80, :80]
    mask = np.ones((80, 80), dtype=bool)
    # Two filled nuclei overlapping into one pinched blob (solidity about 0.88 after smoothing)
    pair = np.maximum(np.hypot(rows - 40, columns - 34) <= 6, np.hypot(rows - 40, columns - 46) <= 6)
    pair = pair.astype(np.float32)
    assert count_nuclei(pair, mask, 12, seed_spacing=1.2)[0] == 1
    assert count_nuclei(pair, mask, 12, seed_spacing=1.2, roundness=0.9)[0] == 2
    # Two filled nuclei touching, dimmer where they meet (as DAPI)
    blobs = sum(np.exp(-((rows - 40) ** 2 + (columns - c) ** 2) / (2 * 4.0**2)) for c in (33, 45))
    assert count_nuclei(blobs.astype(np.float32), mask, 12, seed_spacing=0.5, split_method="intensity")[0] == 2


def test_count_nuclei_size_limits_drop_debris_and_split_clumps():
    from Experiments.micropatterns.cell_counts import count_nuclei

    rows, columns = np.mgrid[:80, :80]
    mask = np.ones((80, 80), dtype=bool)
    clump = (np.hypot(rows - 40, columns - 40) <= 14).astype(np.float32)  # about 5 nuclei of diameter 12
    speck = (np.hypot(rows - 8, columns - 8) <= 3).astype(np.float32)  # under half a nucleus
    image = np.maximum(clump, speck)

    count, labels, _ = count_nuclei(image, mask, 12)
    assert count == 2  # the clump as one nucleus, and the speck
    count, labels, _ = count_nuclei(image, mask, 12, largest=2.0)
    assert count == 6
    assert labels[40, 40] > 0 and len(np.unique(labels[np.hypot(rows - 40, columns - 40) <= 12])) == 5
    assert count_nuclei(image, mask, 12, largest=2.0, smallest=0.5)[0] == 5


def test_cells_and_density_convert_both_ways():
    from Experiments.micropatterns.cell_counts import cells_to_density, density_to_cells

    layer_cells, coverage = [200.0, 150.0], [0.45, 0.35]
    for density in (0.2, 0.5, 0.6):
        nuclei = density_to_cells(density, layer_cells, coverage, 1.5)
        assert np.isclose(cells_to_density(nuclei, layer_cells, coverage, 1.5), density, atol=1e-4)


def test_kept_cell_count_matches_the_cells_a_placement_keeps():
    import jax.numpy as jnp

    from Experiments.micropatterns.texture_synthesis import layer_cell_counts, place_cells
    from NCA.inverse_design.density import fill_level, kept_cell_count

    square = (np.ones((3, 3), dtype=bool), np.ones((1, 3, 3), dtype=np.float32))
    _, ranks, coverage = place_cells([((0,), [square])], 1, (60, 60), 2, 10, seed=0)
    mask = np.ones((60, 60), dtype=bool)
    cells = layer_cell_counts(ranks, mask)
    for density in (0.2, 0.5, 0.7):
        level = float(fill_level(density, jnp.asarray(coverage)))
        # Kept with no shift and a hard cut: layer + rank below the level
        kept = sum(
            len(np.unique(ranks[layer, 0][(ranks[layer, 0] >= 0) & (layer + ranks[layer, 0] < level)]))
            for layer in range(2)
        )
        expected = float(kept_cell_count(level, jnp.asarray(cells)))
        # Ranks are random, so the count is right on average (binomial spread)
        assert abs(kept - expected) <= 3.0 * np.sqrt(expected) + 1.0


def test_few_cells_show_few_cells_with_a_softness_in_cells():
    import jax
    import jax.numpy as jnp
    import scipy.ndimage as ndi

    from Experiments.micropatterns.cell_counts import cells_to_density, rank_softness
    from Experiments.micropatterns.texture_synthesis import layer_cell_counts, place_cells
    from NCA.inverse_design.density import CellDensity, CellGaps, pack_cells

    square = (np.ones((3, 3), dtype=bool), np.ones((1, 3, 3), dtype=np.float32))
    textures, ranks, coverage = place_cells([((0,), [square])], 1, (80, 80), 2, 10, seed=0)
    cells = layer_cell_counts(ranks, np.ones((80, 80), dtype=bool))
    gaps = CellGaps(
        jnp.zeros(1), jnp.zeros(1, dtype=int), jnp.asarray(coverage), panels=1,
        softness=rank_softness(0.5, ranks),
    )
    density = cells_to_density(10, cells, coverage, 1.0)
    image = np.asarray(gaps.apply(pack_cells(jnp.asarray(textures), jnp.asarray(ranks)), CellDensity(density),
                                  jax.random.PRNGKey(0)))[0]

    # About 10 (rank sampling spread), not the dozens a fringe in rank units shows
    assert ndi.label(image > 0.05)[1] <= 20


def test_downsample_cell_keeps_blocks_half_covered_with_the_cells_own_mean():
    from Experiments.micropatterns.cell_counts import downsample_cell

    mask = np.zeros((8, 8), dtype=bool)
    mask[0:4, 0:6] = True  # 1 x 1.5 blocks of 4
    values = np.stack([np.where(mask, 2.0, 100.0)]).astype(np.float32)

    small_mask, small_values = downsample_cell(mask, values, 4)

    np.testing.assert_array_equal(small_mask, [[True, True]])
    # Only the cell's own pixels are averaged, not the 100s around it
    np.testing.assert_allclose(small_values[0], [[2.0, 2.0]])


def test_match_values_maps_raw_to_model_scale():
    from Experiments.micropatterns.cell_counts import match_values

    rng = np.random.default_rng(0)
    source = rng.uniform(100, 1000, size=(5000, 1))
    target = (source - 100) / 900  # the model's [0, 1] scale

    matched = match_values(np.asarray([[[100.0, 550.0, 1000.0]]]), source, target)

    np.testing.assert_allclose(matched[0, 0], [0.0, 0.5, 1.0], atol=0.03)


def test_nuclear_cells_gives_one_piece_per_nucleus(tmp_path, monkeypatch):
    import Common.dataloader.micropattern_260726 as loader
    from Experiments.micropatterns import cell_counts

    rows, cols = np.mgrid[:200, :200]
    centres = [(80, 80), (80, 120), (120, 100)]
    nuclear = np.zeros((200, 200), dtype=np.float32)
    for row, column in centres:
        nuclear[(rows - row) ** 2 + (cols - column) ** 2 <= 7**2] = 1.0
    channels = np.stack([nuclear * 50.0 + 10.0, nuclear * 5.0], axis=-1)
    monkeypatch.setattr(loader, "read_nuclear_channel", lambda path, group: nuclear)
    monkeypatch.setattr(loader, "read_raw_channels", lambda path, group: (channels, ("p/A", "p/B"), nuclear))
    image = tmp_path / "p" / "image.tif"
    image.parent.mkdir()
    image.touch()

    class Alignment:
        pattern_radius = 80.0

        def centre(self, relative):
            return (100.0, 100.0)

    # Model colony at downsample 4, on a [0, 1] scale
    target = np.stack([np.asarray(channels[::4, ::4, 0] / 60.0), np.asarray(channels[::4, ::4, 1] / 5.0)])
    support = cell_counts._circle((50, 50), (25, 25), 20)
    cells, covered = cell_counts.nuclear_cells(
        str(image), "rna_expression", ["p/A", "p/B"], tmp_path, Alignment(), 4, target, support,
        nucleus_diameter_um=14 * cell_counts.um_per_pixel(80.0), expansion_um=0.0,
    )

    assert len(cells) == len(centres)
    # Three discs of radius 7 in a colony of radius 80
    assert abs(covered - 3 * 7**2 / 80**2) < 0.01
    for mask, values in cells:
        assert values.shape[0] == 2
        assert 4 <= mask.sum() <= 16
        assert np.all(values[:, mask] > 0.5)
