"""Texture synthesis for micropattern initial conditions.

Builds synthetic 0 h biology for a new colony geometry from a measured
circular reference colony. Used by ``boundary_mask_explorer.py`` and
``drawn_mask_explorer.py``. NumPy and SciPy only.

The cell density helpers at the end cut whole cells out of measured
colonies (``cell_coverage``, ``segment_cells``) and place them at random
over a new grid (``place_cells``), for the cell density texture model in
``NCA/inverse_design/density.py``.
"""

import numpy as np
import scipy.ndimage as ndi


def resize_image(image, output_shape):
    """Bilinear resize with ``np.interp``."""

    _input_y = np.arange(image.shape[-2], dtype=np.float32)
    _input_x = np.arange(image.shape[-1], dtype=np.float32)
    _output_y = np.linspace(0, image.shape[-2] - 1, output_shape[0])
    _output_x = np.linspace(0, image.shape[-1] - 1, output_shape[1])
    _rows = np.stack(
        [np.interp(_output_x, _input_x, _row) for _row in image], axis=0
    )
    return np.stack(
        [np.interp(_output_y, _input_y, _rows[:, _column])
         for _column in range(_rows.shape[1])],
        axis=1,
    )


def evenly_spaced_quantiles(samples, count):
    """``count`` quantiles of ``samples`` from 0 to 1, as ``np.quantile`` gives them.

    Sorting once and interpolating is much faster than ``np.quantile`` with
    thousands of quantile levels.
    """

    _sorted = np.sort(np.asarray(samples).ravel())
    _positions = np.linspace(0.0, 1.0, int(count)) * (_sorted.size - 1)
    return np.interp(_positions, np.arange(_sorted.size), _sorted).astype(np.float32)


def extend_masked_texture(image, mask):
    """Smoothly fill the outside of a colony, for estimating its Fourier spectrum.

    The outside starts at the interior mean and is relaxed by diffusion
    while the measured pixels stay fixed. (A random fill would add
    high-frequency power; a nearest-pixel fill gives Voronoi regions.)
    The fill is only used for the spectrum, not shown as data.
    """

    _mask = np.asarray(mask, dtype=bool)
    if not np.any(_mask):
        raise ValueError("The reference mask contains no interior pixels")
    _image = np.asarray(image, dtype=np.float32)
    _extended = np.full_like(_image, np.mean(_image[_mask]))
    _extended[_mask] = _image[_mask]
    # Repeated Gaussian smoothing approximates a harmonic fill. Reflecting
    # at the image edge avoids a second zero boundary.
    for _ in range(64):
        _relaxed = ndi.gaussian_filter(_extended, sigma=1.0, mode="reflect")
        _extended[~_mask] = _relaxed[~_mask]
        _extended[_mask] = _image[_mask]
    return _extended


def synthesise_reference_texture(
    reference,
    reference_mask,
    output_shape,
    output_mask,
    iterations,
    seed,
):
    """Multichannel IAAFT-like texture synthesis from a circular reference.

    Each channel alternates between matching the reference power spectrum
    and its value distribution. All channels start from the same random
    phases, so their relative phases are kept. The last distribution match
    is done inside the proposed geometry.
    """

    _rng = np.random.default_rng(seed)
    _reference_mask = np.asarray(reference_mask) > 0.5
    if not np.any(_reference_mask):
        raise ValueError("The reference mask contains no interior pixels")
    _shared_phase = np.angle(
        np.fft.rfft2(_rng.normal(size=output_shape))
    )
    _synthesised = []
    _target_inside = np.asarray(output_mask) > 0.5
    for _channel in np.asarray(reference):
        _samples = np.asarray(_channel)[_reference_mask]
        if _samples.size < 2:
            raise ValueError("Each reference channel needs at least two masked pixels")
        # Smooth continuation prevents both the circular zero edge and an
        # i.i.d. background fill from contaminating high-frequency power.
        _filled = extend_masked_texture(_channel, _reference_mask)
        _resized = resize_image(_filled, output_shape)
        _target_fft = np.fft.rfft2(_resized)
        _target_amplitude = np.abs(_target_fft)
        _current = np.fft.irfft2(
            _target_amplitude
            * np.exp(1j * (np.angle(_target_fft) + _shared_phase)),
            s=output_shape,
        )
        _quantiles = evenly_spaced_quantiles(_samples, _current.size)
        for _ in range(max(int(iterations), 0)):
            _current_fft = np.fft.rfft2(_current)
            _spectral_projection = np.fft.irfft2(
                _target_amplitude * np.exp(1j * np.angle(_current_fft)),
                s=output_shape,
            )
            _order = np.argsort(_spectral_projection, axis=None)
            _flat = np.empty(_spectral_projection.size, dtype=np.float32)
            _flat[_order] = _quantiles
            _current = _flat.reshape(output_shape)
        # Enforce the empirical distribution where cells actually exist.
        _inside_count = np.count_nonzero(_target_inside)
        if _inside_count:
            _inside_quantiles = evenly_spaced_quantiles(_samples, _inside_count)
            _inside_values = _current[_target_inside]
            _inside_order = np.argsort(_inside_values)
            _ranked = np.empty_like(_inside_values, dtype=np.float32)
            _ranked[_inside_order] = _inside_quantiles
            _current[_target_inside] = _ranked
        _synthesised.append(_current.astype(np.float32))
    return np.stack(_synthesised)


def synthesise_patch_texture(
    reference,
    reference_mask,
    output_shape,
    measurement_groups,
    patch_size,
    overlap,
    interior_fraction,
    jitter,
    candidate_count,
    seed,
):
    """Texture made from real multichannel patches, blended where they overlap.

    Channels of one staining panel share the same source patches and
    transforms, which keeps cell shapes and within-panel correlations
    without inventing correlations between panels. Patches come from the
    central part of the colony, away from edge artefacts. At each position
    the candidate with the smallest normalised overlap error is used.
    """

    _reference = np.asarray(reference, dtype=np.float32)
    _reference_mask = np.asarray(reference_mask, dtype=bool)
    _rng = np.random.default_rng(seed)
    _maximum_patch = min(*_reference_mask.shape, *output_shape)
    _patch_size = min(int(patch_size), _maximum_patch)
    if _patch_size % 2 == 0:
        _patch_size -= 1
    _patch_size = max(_patch_size, 3)
    _half = _patch_size // 2
    _mask_coordinates = np.argwhere(_reference_mask)
    _centre_y, _centre_x = np.mean(_mask_coordinates, axis=0)
    _grid_y, _grid_x = np.indices(_reference_mask.shape)
    _radial_distance = np.sqrt(
        (_grid_y - _centre_y) ** 2 + (_grid_x - _centre_x) ** 2
    )
    _reference_radius = np.max(_radial_distance[_reference_mask])
    _sampling_mask = _reference_mask & (
        _radial_distance <= float(interior_fraction) * _reference_radius
    )
    # Erode so that whole patches, not just their centres, lie within the radius.
    _valid = ndi.binary_erosion(_sampling_mask, iterations=_half)
    while not np.any(_valid) and _patch_size > 3:
        _patch_size -= 2
        _half = _patch_size // 2
        _valid = ndi.binary_erosion(_sampling_mask, iterations=_half)
    _valid_centres = np.argwhere(_valid)
    if not len(_valid_centres):
        raise ValueError("Reference colony is too small for a 3×3 source patch")
    _effective_overlap = min(int(overlap), _patch_size - 1)
    _stride = max(1, _patch_size - _effective_overlap)
    _effective_jitter = min(int(jitter), _effective_overlap)

    def _jittered_positions(length):
        _last = max(length - _patch_size, 0)
        _positions = [0]
        while _positions[-1] < _last:
            _candidate = _positions[-1] + _stride
            if _effective_jitter:
                _candidate += int(
                    _rng.integers(-_effective_jitter, _effective_jitter + 1)
                )
            # A step no larger than the patch width guarantees coverage;
            # strictly increasing positions guarantee termination.
            _candidate = min(
                max(_candidate, _positions[-1] + 1),
                _positions[-1] + _patch_size,
                _last,
            )
            if _candidate == _positions[-1]:
                break
            _positions.append(_candidate)
        return _positions

    # Flat-top window: only the overlap border is feathered (raised cosine
    # with a nonzero edge weight), so patch centres are not dimmed.
    _feather = min(max(_effective_overlap, 1), _patch_size // 2)
    _window_1d = np.ones(_patch_size, dtype=np.float32)
    _ramp = 0.5 - 0.5 * np.cos(
        np.pi * (np.arange(_feather, dtype=np.float32) + 1) / (_feather + 1)
    )
    _window_1d[:_feather] = _ramp
    _window_1d[-_feather:] = _ramp[::-1]
    _window = np.outer(_window_1d, _window_1d)
    _result = np.zeros((_reference.shape[0], *output_shape), dtype=np.float32)
    for _group in measurement_groups:
        _positions_y = _jittered_positions(output_shape[0])
        _indices = np.asarray(_group, dtype=int)
        _channel_scale = np.std(
            _reference[_indices][:, _reference_mask], axis=1
        )
        _channel_scale = np.maximum(_channel_scale, 1.0e-6)
        _accumulator = np.zeros((len(_indices), *output_shape), dtype=np.float32)
        _weights = np.zeros(output_shape, dtype=np.float32)
        for _top in _positions_y:
            # Independently jitter each row so vertical seams do not align
            # into a checkerboard lattice.
            _positions_x = _jittered_positions(output_shape[1])
            for _left in _positions_x:
                _height = min(_patch_size, output_shape[0] - _top)
                _width = min(_patch_size, output_shape[1] - _left)
                _blend = _window[:_height, :_width]
                _region_weights = _weights[
                    _top : _top + _height, _left : _left + _width
                ]
                _overlap_mask = _region_weights > 1.0e-6
                if np.any(_overlap_mask):
                    _existing = (
                        _accumulator[
                            :, _top : _top + _height, _left : _left + _width
                        ]
                        / np.maximum(_region_weights[None], 1.0e-6)
                    )
                _best_patch = None
                _best_score = np.inf
                for _ in range(max(int(candidate_count), 1)):
                    _centre_y, _centre_x = _valid_centres[
                        _rng.integers(len(_valid_centres))
                    ]
                    _candidate = _reference[
                        _indices,
                        _centre_y - _half : _centre_y + _half + 1,
                        _centre_x - _half : _centre_x + _half + 1,
                    ]
                    _rotation = int(_rng.integers(4))
                    _candidate = np.rot90(
                        _candidate, _rotation, axes=(-2, -1)
                    )
                    if _rng.random() < 0.5:
                        _candidate = _candidate[..., ::-1]
                    _candidate = _candidate[:, :_height, :_width]
                    if np.any(_overlap_mask):
                        _difference = (
                            _candidate[:, _overlap_mask]
                            - _existing[:, _overlap_mask]
                        ) / _channel_scale[:, None]
                        _score = float(np.mean(_difference**2))
                    else:
                        _score = 0.0
                    if _score < _best_score:
                        _best_score = _score
                        _best_patch = _candidate
                    if _score == 0.0:
                        break
                _accumulator[:, _top : _top + _height, _left : _left + _width] += (
                    _best_patch * _blend[None]
                )
                _weights[_top : _top + _height, _left : _left + _width] += _blend
        _result[_indices] = _accumulator / np.maximum(_weights[None], 1.0e-6)
    return _result, _patch_size, _effective_overlap, _effective_jitter


def normalise_histograms_and_covariance(
    synthetic,
    reference,
    reference_mask,
    output_mask,
    measurement_groups,
    iterations,
):
    """Match each channel's histogram and, approximately, each panel's covariance.

    Both cannot in general be matched exactly at once, so within each
    co-measured panel this alternates quantile matching per channel with
    whitening and recolouring towards the reference covariance. A final
    quantile match makes the histograms exact.
    """

    _result = np.array(synthetic, dtype=np.float32, copy=True)
    _reference_mask = np.asarray(reference_mask, dtype=bool)
    _output_mask = np.asarray(output_mask) > 0.5

    def _quantile_project(values, targets):
        _projected = np.empty_like(values, dtype=np.float32)
        for _channel in range(values.shape[0]):
            _quantiles = evenly_spaced_quantiles(targets[_channel], values.shape[1])
            _order = np.argsort(values[_channel])
            _projected[_channel, _order] = _quantiles
        return _projected

    def _matrix_power(matrix, power):
        _eigenvalues, _eigenvectors = np.linalg.eigh(matrix)
        _floor = max(float(np.max(_eigenvalues)) * 1.0e-6, 1.0e-8)
        _powered = np.maximum(_eigenvalues, _floor) ** power
        return (_eigenvectors * _powered[None]) @ _eigenvectors.T

    for _group in measurement_groups:
        _indices = np.asarray(_group, dtype=int)
        _target = np.asarray(reference)[_indices][:, _reference_mask]
        _values = _result[_indices][:, _output_mask]
        if _values.shape[1] < 2 or _target.shape[1] < 2:
            continue
        _target_mean = np.mean(_target, axis=1, keepdims=True)
        _target_covariance = np.atleast_2d(np.cov(_target, bias=False))
        for _ in range(max(int(iterations), 1)):
            _values = _quantile_project(_values, _target)
            if len(_indices) > 1:
                _value_mean = np.mean(_values, axis=1, keepdims=True)
                _value_covariance = np.atleast_2d(np.cov(_values, bias=False))
                _whiten = _matrix_power(_value_covariance, -0.5)
                _colour = _matrix_power(_target_covariance, 0.5)
                _values = _target_mean + _colour @ _whiten @ (
                    _values - _value_mean
                )
        _values = _quantile_project(_values, _target)
        for _local_index, _measurement_index in enumerate(_indices):
            _channel = _result[_measurement_index]
            _channel[_output_mask] = _values[_local_index]
            _result[_measurement_index] = _channel
    return _result


def otsu_threshold(values, bins=256):
    """Threshold that best splits ``values`` into two groups (Otsu's method)."""

    values = np.asarray(values, dtype=np.float64).ravel()
    counts, edges = np.histogram(values, bins=bins)
    centres = 0.5 * (edges[:-1] + edges[1:])
    weight_low = np.cumsum(counts)
    weight_high = weight_low[-1] - weight_low
    sum_low = np.cumsum(counts * centres)
    mean_low = sum_low / np.maximum(weight_low, 1)
    mean_high = (sum_low[-1] - sum_low) / np.maximum(weight_high, 1)
    between = weight_low * weight_high * (mean_low - mean_high) ** 2
    return float(edges[np.argmax(between) + 1])


def cell_coverage(measurements, support, marker_indices):
    """Pixels covered by cells ``[H, W]`` (bool) and the threshold used.

    A pixel is covered when the largest of the generic cell markers
    (``marker_indices`` into ``measurements`` ``[C, H, W]``) is above the
    Otsu threshold of that value inside the colony ``support``.
    """

    support = np.asarray(support, dtype=bool)
    if not np.any(support):
        raise ValueError("The colony support is empty")
    signal = np.max(np.asarray(measurements)[np.asarray(marker_indices, dtype=int)], axis=0)
    threshold = otsu_threshold(signal[support])
    return (signal > threshold) & support, threshold


def interior_mask(support, fraction):
    """Pixels of ``support`` within ``fraction`` of its largest radius from its centre."""

    support = np.asarray(support, dtype=bool)
    coordinates = np.argwhere(support)
    centre_y, centre_x = np.mean(coordinates, axis=0)
    grid_y, grid_x = np.indices(support.shape)
    distance = np.hypot(grid_y - centre_y, grid_x - centre_x)
    return support & (distance <= float(fraction) * np.max(distance[support]))


def segment_cells(measurements, support, coverage, signal, split_distance, interior_fraction, min_area):
    """Whole cells of one colony, as a list of ``(mask [h, w], values [C, h, w])``.

    Covered pixels (``coverage``) are split into cells by a watershed of the
    marker ``signal`` ``[H, W]``, started from its local maxima at least
    ``split_distance`` pixels apart (0: each connected patch of covered
    pixels is one cell). Only cells that lie wholly within
    ``interior_fraction`` of the colony radius (away from edge artefacts)
    and have at least ``min_area`` pixels are kept. ``values`` are the
    cell's bounding box of ``measurements`` ``[C, H, W]``.
    """

    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed

    coverage = np.asarray(coverage, dtype=bool) & np.asarray(support, dtype=bool)
    patches, count = ndi.label(coverage)
    if split_distance > 0 and count:
        seeds = np.zeros_like(patches)
        peaks = peak_local_max(
            np.asarray(signal, dtype=np.float32), min_distance=int(split_distance),
            labels=patches, exclude_border=False,
        )
        seeds[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
        # Patches without a maximum (flat ones) start from their brightest pixel
        seeded = set(np.unique(patches[seeds > 0]))
        for patch in range(1, count + 1):
            if patch not in seeded:
                pixels = np.argwhere(patches == patch)
                brightest = pixels[np.argmax(signal[tuple(pixels.T)])]
                seeds[tuple(brightest)] = seeds.max() + 1
        labels = watershed(-np.asarray(signal, dtype=np.float32), seeds, mask=coverage)
    else:
        labels = patches
    outside = ~interior_mask(support, interior_fraction)
    measurements = np.asarray(measurements, dtype=np.float32)
    cells = []
    for label, box in enumerate(ndi.find_objects(labels), start=1):
        if box is None:
            continue
        mask = labels[box] == label
        if np.count_nonzero(mask) < min_area or np.any(outside[box] & mask):
            continue
        cells.append((mask, measurements[(slice(None), *box)].copy()))
    return cells


def _turned(image, turns, flip):
    image = np.rot90(image, turns, axes=(-2, -1))
    return image[..., ::-1] if flip else image


def place_cells(panels, channel_count, output_shape, layers, attempts, seed):
    """Whole measured cells at random places, in ``layers`` independent layers.

    ``panels`` holds, for each staining panel, ``(channel indices, cells)``
    with cells from ``segment_cells`` (values in that panel's channels).
    Each layer is filled by trying random places in turn: at each place
    every panel draws one of its cells, randomly turned and flipped, centred
    on the place. They are added if all fit in the grid and none overlaps a
    cell of that panel already in the layer, so cells are never cut. About
    ``attempts`` places are tried per cell that would fit in the grid. Each
    added place gets a random rank in [0, 1), shared by its panels.

    Returns ``textures`` ``[layers, channel_count, H, W]`` (the cells'
    values, 0 elsewhere), ``ranks`` ``[layers, panels, H, W]`` (the rank of
    the cell on each pixel, -1 where there is none) and ``coverage``
    ``[layers]``, the share of the grid each layer covers (a band at the
    grid edge as wide as half the largest cell is left out, since cells
    there would not fit).
    """

    rng = np.random.default_rng(seed)
    height, width = output_shape
    if not panels or any(not cells for _, cells in panels):
        raise ValueError("Every panel needs at least one cell")
    mean_area = np.mean([np.mean([mask.sum() for mask, _ in cells]) for _, cells in panels])
    tries = int(np.ceil(attempts * height * width / mean_area))
    textures = np.zeros((layers, channel_count, height, width), dtype=np.float32)
    ranks = np.full((layers, len(panels), height, width), -1.0, dtype=np.float32)
    for layer in range(layers):
        for _ in range(tries):
            centre = rng.integers(0, height), rng.integers(0, width)
            drawn = []
            for panel, (_, cells) in enumerate(panels):
                mask, values = cells[rng.integers(len(cells))]
                turns, flip = int(rng.integers(4)), bool(rng.integers(2))
                mask, values = _turned(mask, turns, flip), _turned(values, turns, flip)
                top = centre[0] - mask.shape[0] // 2
                left = centre[1] - mask.shape[1] // 2
                if top < 0 or left < 0 or top + mask.shape[0] > height or left + mask.shape[1] > width:
                    break
                box = (slice(top, top + mask.shape[0]), slice(left, left + mask.shape[1]))
                if np.any(ranks[(layer, panel, *box)][mask] >= 0):
                    break
                drawn.append((panel, box, mask, values))
            else:
                rank = rng.random()
                for panel, box, mask, values in drawn:
                    indices = np.asarray(panels[panel][0], dtype=int)
                    region = textures[(layer, slice(None), *box)]
                    region[indices] = np.where(mask, values, region[indices])
                    ranks[(layer, panel, *box)][mask] = rank
    margin = max(max(mask.shape) for _, cells in panels for mask, _ in cells) // 2
    margin = min(margin, (min(height, width) - 1) // 2)
    inner = (slice(None), slice(None), slice(margin, height - margin), slice(margin, width - margin))
    coverage = np.mean(ranks[inner] >= 0, axis=(1, 2, 3))
    return textures, ranks, coverage.astype(np.float32)
