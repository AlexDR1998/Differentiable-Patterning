# /// script
# dependencies = [
#   "marimo",
#   "equinox",
#   "jax",
#   "matplotlib",
#   "numpy",
#   "omegaconf",
#   "scipy",
#   "scikit-image",
# ]
# ///

"""Interactive boundary-mask and initial-condition explorer.

Run from the repository root with:

    marimo run Experiments/micropatterns/boundary_mask_explorer.py

The functions use NumPy for a lightweight visual preview. Their equations are
composed from operations that have direct JAX equivalents, so they can be moved
to the differentiable inverse-design path without changing the parameterisation.
"""

import marimo

__generated_with = "0.23.10"
app = marimo.App(width="columns")

with app.setup:
    import os
    from pathlib import Path
    from pathlib import Path as _Path
    import sys as _sys
    _sys.path.append('/home/alex/PhD/Differentiable-Patterning/')
    import marimo as mo
    import jax.numpy as jnp
    import jax.random as jr
    import matplotlib.pyplot as plt
    import numpy as np
    import scipy.ndimage as ndi
    from omegaconf import OmegaConf

    from Common.dataloader.micropattern_260726 import load_micropattern_260726
    from Experiments.model_registry import ModelRegistry
    from NCA.trainer.intervention import rollout_model_sampled

    FATE_MARKERS = ("TBXT", "SOX17", "SOX2", "FOXA2")
    CELL_TYPES = ("Notochord", "Endoderm", "Mesoderm")
    DEFAULT_FATE_RULES = {
        "Notochord": {
            "TBXT": "high", "SOX17": "low", "SOX2": "low", "FOXA2": "high"
        },
        "Endoderm": {
            "TBXT": "any", "SOX17": "high", "SOX2": "any", "FOXA2": "any"
        },
        "Mesoderm": {
            "TBXT": "high", "SOX17": "low", "SOX2": "high", "FOXA2": "low"
        },
    }


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Micropattern boundary-mask explorer

    Compare low-dimensional adhesion-mask parameterisations and inspect the
    NCA initial condition induced by the selected mask.

    The preview follows the current **soft-boundary** convention: biological
    channels occupy the start of the state, hidden channels start at zero, and
    the final state channel contains the adhesion mask. Outside-mask biological
    state is zero, avoiding residual circular initial conditions when a new
    geometry is selected.
    """)
    return


@app.cell(hide_code=True)
def _():
    def coordinate_grid(size):
        _height, _width = (size, size) if np.isscalar(size) else size
        _axis_y = np.linspace(-1.0, 1.0, _height, dtype=np.float32)
        _axis_x = np.linspace(-1.0, 1.0, _width, dtype=np.float32)
        return np.meshgrid(_axis_x, _axis_y, indexing="xy")

    def soft_occupancy(level_set, softness):
        _width = max(float(softness), 1.0e-4)
        _scaled = np.clip(level_set / _width, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(_scaled))

    def ellipse_level_set(x_grid, y_grid, radius, aspect, angle):
        _cosine = np.cos(angle)
        _sine = np.sin(angle)
        _x_rot = _cosine * x_grid + _sine * y_grid
        _y_rot = -_sine * x_grid + _cosine * y_grid
        _radius_x = radius * np.sqrt(aspect)
        _radius_y = radius / np.sqrt(aspect)
        return np.sqrt(
            (_x_rot / _radius_x) ** 2 + (_y_rot / _radius_y) ** 2
        ) - 1.0

    def superellipse_level_set(
        x_grid, y_grid, radius, aspect, exponent, angle
    ):
        _cosine = np.cos(angle)
        _sine = np.sin(angle)
        _x_rot = _cosine * x_grid + _sine * y_grid
        _y_rot = -_sine * x_grid + _cosine * y_grid
        _radius_x = radius * np.sqrt(aspect)
        _radius_y = radius / np.sqrt(aspect)
        _power = max(float(exponent), 0.25)
        return (
            np.abs(_x_rot / _radius_x) ** _power
            + np.abs(_y_rot / _radius_y) ** _power
        ) ** (1.0 / _power) - 1.0

    def fourier_level_set(
        x_grid, y_grid, radius, coefficients_cos, coefficients_sin, angle
    ):
        _theta = np.arctan2(y_grid, x_grid) - angle
        _distance = np.sqrt(x_grid**2 + y_grid**2)
        _log_radius = np.zeros_like(_theta)
        for _harmonic, (_cos_coeff, _sin_coeff) in enumerate(
            zip(coefficients_cos, coefficients_sin), start=1
        ):
            _log_radius += _cos_coeff * np.cos(_harmonic * _theta)
            _log_radius += _sin_coeff * np.sin(_harmonic * _theta)
        # Normalisation approximately preserves area as Fourier amplitudes vary.
        _radial_boundary = radius * np.exp(_log_radius)
        _radial_boundary *= radius / np.sqrt(np.mean(_radial_boundary**2))
        return _distance / np.maximum(_radial_boundary, 1.0e-4) - 1.0

    def annulus_level_set(x_grid, y_grid, radius, hole_fraction):
        _distance = np.sqrt(x_grid**2 + y_grid**2)
        _outer = _distance / radius - 1.0
        _inner_radius = radius * hole_fraction
        _inner = 1.0 - _distance / max(_inner_radius, 1.0e-4)
        return np.maximum(_outer, _inner)

    def make_level_set(kind, x_grid, y_grid, parameters):
        if kind == "Circle":
            return ellipse_level_set(
                x_grid, y_grid, parameters["radius"], 1.0, 0.0
            )
        if kind == "Ellipse":
            return ellipse_level_set(
                x_grid,
                y_grid,
                parameters["radius"],
                parameters["aspect"],
                parameters["angle"],
            )
        if kind == "Superellipse":
            return superellipse_level_set(
                x_grid,
                y_grid,
                parameters["radius"],
                parameters["aspect"],
                parameters["exponent"],
                parameters["angle"],
            )
        if kind == "Fourier radial":
            return fourier_level_set(
                x_grid,
                y_grid,
                parameters["radius"],
                parameters["fourier_cos"],
                parameters["fourier_sin"],
                parameters["angle"],
            )
        if kind == "Annulus":
            return annulus_level_set(
                x_grid,
                y_grid,
                parameters["radius"],
                parameters["hole_fraction"],
            )
        raise ValueError(f"Unknown geometry {kind!r}")

    def construct_initial_state(
        mask,
        level_set,
        channel_count,
        observed_channels,
        selected_channel,
        interior_value,
        edge_value,
        edge_width,
        profile_mode,
        synthetic_channels=None,
    ):
        """Create [C,H,W] state consistent with the proposed geometry.

        The geometry channel is always the final channel. This matches
        ``model_boundary``, which clamps the final mask channels after every
        update. Biological channels are restricted to the adhesion domain and
        hidden channels remain zero.
        """

        if channel_count <= observed_channels:
            raise ValueError(
                "channel_count must exceed observed_channels so the final "
                "channel can be reserved for geometry"
            )
        _state = np.zeros(
            (channel_count, *mask.shape), dtype=np.float32
        )
        if synthetic_channels is not None:
            _channel_count = min(observed_channels, synthetic_channels.shape[0])
            _state[:_channel_count] = (
                synthetic_channels[:_channel_count] * mask[None]
            )
        elif profile_mode == "Uniform":
            _biological_profile = np.full_like(mask, interior_value)
        else:
            # level_set is zero at the edge and negative inside. The blend is
            # one near the edge and smoothly approaches zero in the interior.
            _edge_blend = np.exp(
                -np.maximum(-level_set, 0.0) / max(edge_width, 1.0e-4)
            )
            _biological_profile = (
                interior_value * (1.0 - _edge_blend)
                + edge_value * _edge_blend
            )
        if synthetic_channels is None:
            _state[selected_channel] = mask * _biological_profile
        _state[-1] = mask
        return _state

    def resize_image(image, output_shape):
        """Bilinear resize implemented without an additional dependency."""

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

    def extend_masked_texture(image, mask):
        """Smoothly continue a measured colony for Fourier analysis.

        An i.i.d. fill injects artificial Nyquist-scale power. Nearest-interior
        filling avoids that noise but creates visible Voronoi regions. Here the
        unmeasured exterior is initialized to the interior mean and relaxed by
        diffusion while measured pixels remain fixed. The continuation is used
        only to estimate a rectangular Fourier spectrum; it is not biological
        data and is not shown as part of the circular reference.
        """

        _mask = np.asarray(mask, dtype=bool)
        if not np.any(_mask):
            raise ValueError("The reference mask contains no interior pixels")
        _image = np.asarray(image, dtype=np.float32)
        _extended = np.full_like(_image, np.mean(_image[_mask]))
        _extended[_mask] = _image[_mask]
        # Repeated Gaussian relaxation approximates harmonic continuation and
        # has no nearest-neighbour ownership boundaries. Reflecting the outer
        # image edge avoids introducing a second artificial zero boundary.
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
        """Multichannel IAAFT-like synthesis from a circular reference.

        Each channel alternates between the reference power spectrum and its
        empirical marginal distribution. A shared random phase perturbation
        initializes all channels, retaining relative cross-channel phases.
        The final rank projection is performed inside the proposed geometry.
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
            _quantiles = np.quantile(
                _samples,
                np.linspace(0.0, 1.0, _current.size, dtype=np.float32),
            )
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
                _inside_quantiles = np.quantile(
                    _samples,
                    np.linspace(0.0, 1.0, _inside_count, dtype=np.float32),
                )
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
        """Resample real multichannel patches with overlap-add blending.

        Channels belonging to one staining panel use identical source patches
        and geometric transforms. This retains cellular morphology and local
        cross-channel dependence without inventing correlations between panels
        that were not co-measured. Source patches are restricted to a central
        fraction of the circular colony to exclude edge and registration artifacts.
        At each placement, several candidates compete on normalized multichannel
        overlap error so adjacent patches meet along compatible cellular texture.
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
        # Eroding the central sampling region ensures every pixel of a source
        # patch, rather than only its centre, lies within the requested radius.
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

        # Unlike a full Hann window, a flat-top window does not periodically
        # attenuate every patch centre. Only the overlap-width border is
        # feathered, using a raised cosine with a nonzero edge weight.
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
        """Match marginal histograms while approaching group covariance.

        Exact marginal histograms and exact Pearson covariance are not, in
        general, simultaneously attainable by a single affine transform. This
        alternates two projections within each genuinely co-measured panel:
        rank/quantile matching for every channel, then whitening and recolouring
        toward the reference covariance. A final quantile projection guarantees
        the requested marginal histograms while retaining the covariance fit as
        closely as the empirical distributions permit.
        """

        _result = np.array(synthetic, dtype=np.float32, copy=True)
        _reference_mask = np.asarray(reference_mask, dtype=bool)
        _output_mask = np.asarray(output_mask) > 0.5

        def _quantile_project(values, targets):
            _projected = np.empty_like(values, dtype=np.float32)
            for _channel in range(values.shape[0]):
                _quantiles = np.quantile(
                    targets[_channel],
                    np.linspace(0.0, 1.0, values.shape[1], dtype=np.float32),
                )
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

    return (
        construct_initial_state,
        coordinate_grid,
        extend_masked_texture,
        make_level_set,
        normalise_histograms_and_covariance,
        soft_occupancy,
        synthesise_patch_texture,
        synthesise_reference_texture,
    )


@app.cell(hide_code=True)
def _(
    micropattern_root,
    profile_mode,
    reference_condition,
    reference_replicate_mode,
    reference_replicate_seed,
    reference_replicates,
    reference_sample,
    reference_time,
    texture_downsample,
):
    reference_channels = None
    reference_measurements = None
    reference_support = None
    reference_channel_names = None
    reference_measurement_names = None
    reference_measurement_groups = None
    reference_primary_measurements = None
    reference_group_replicates = None
    reference_spatial_shape = None
    dataset_error = None
    if profile_mode.value != "Uniform":
        try:
            _dataset = load_micropattern_260726(
                root=str(Path(micropattern_root.value).expanduser()),
                conditions=(reference_condition.value,),
                timesteps=(0, 12, 24, 36, 48),
                downsample=int(texture_downsample.value),
                replicate_count=int(reference_replicates.value),
                experiment_groups=(
                    "cell_fate_s1",
                    "cell_fate_s2",
                    "rna_expression",
                    "protein_response",
                ),
            )
            _time = int(reference_time.value)
            _schema = _dataset.aux["channel_schema"]
            _all_measurements = np.asarray(_dataset.data[:, _time])
            _availability = np.asarray(_dataset.measurement_mask[:, _time])
            _replicate_labels = tuple(
                _dataset.aux.get(
                    "batch_replicates", range(1, _all_measurements.shape[0] + 1)
                )
            )
            _group_indices = tuple(
                tuple(_group) for _group in _schema.group_measurement_indices
            )
            if reference_replicate_mode.value == "sample_groups":
                _rng = np.random.default_rng(int(reference_replicate_seed.value))
                reference_measurements = np.zeros_like(_all_measurements[0])
                _assignments = []
                for _group_name, _indices_tuple in zip(
                    _schema.group_names, _group_indices
                ):
                    _indices = np.asarray(_indices_tuple, dtype=int)
                    _eligible = np.flatnonzero(
                        np.all(_availability[:, _indices], axis=1)
                    )
                    if not len(_eligible):
                        raise ValueError(
                            f"No loaded replicate measures {_group_name!r} at the selected time"
                        )
                    _chosen = int(_rng.choice(_eligible))
                    reference_measurements[_indices] = _all_measurements[
                        _chosen, _indices
                    ]
                    _assignments.append(
                        (_group_name, _replicate_labels[_chosen])
                    )
                reference_group_replicates = tuple(_assignments)
                _support_sample = int(_rng.choice(_all_measurements.shape[0]))
            else:
                _sample = int(reference_sample.value) % _all_measurements.shape[0]
                reference_measurements = _all_measurements[_sample]
                reference_group_replicates = tuple(
                    (_group_name, _replicate_labels[_sample])
                    for _group_name in _schema.group_names
                )
                _support_sample = _sample
            reference_channels = reference_measurements[
                np.asarray(_schema.primary_measurements)
            ]
            reference_support = np.asarray(
                _dataset.boundary_mask[_support_sample, 0], dtype=bool
            )
            reference_channel_names = tuple(_schema.state_channels)
            reference_measurement_names = tuple(_schema.measurement_names)
            reference_measurement_groups = _group_indices
            reference_primary_measurements = tuple(_schema.primary_measurements)
            reference_spatial_shape = reference_channels.shape[-2:]
        except (OSError, ValueError, IndexError, KeyError) as _error:
            dataset_error = str(_error)
    return (
        dataset_error,
        reference_channel_names,
        reference_channels,
        reference_group_replicates,
        reference_measurement_groups,
        reference_measurement_names,
        reference_measurements,
        reference_primary_measurements,
        reference_spatial_shape,
        reference_support,
    )


@app.cell(hide_code=True)
def _():
    fourier_values, set_fourier_values = mo.state(
        (
            0.00, 0.00, 0.15, 0.00, 0.10, 0.00, 0.00, 0.08,
            0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
        ),
        allow_self_loops=True,
    )
    return fourier_values, set_fourier_values


@app.cell(hide_code=True)
def _(fourier_values, set_fourier_values):
    _limits = (0.55, 0.55, 0.50, 0.50, 0.45, 0.45, 0.40, 0.40,
               0.35, 0.35, 0.32, 0.32, 0.30, 0.30, 0.28, 0.28)
    _values = fourier_values()
    _sliders = [
        mo.ui.slider(
            -_limit,
            _limit,
            value=float(_value),
            step=0.01,
            label=f"{'cos' if _index % 2 == 0 else 'sin'} "
            f"{_index // 2 + 1}φ",
        )
        for _index, (_limit, _value) in enumerate(zip(_limits, _values))
    ]
    (
        fourier_c1, fourier_s1, fourier_c2, fourier_s2,
        fourier_c3, fourier_s3, fourier_c4, fourier_s4,
        fourier_c5, fourier_s5, fourier_c6, fourier_s6,
        fourier_c7, fourier_s7, fourier_c8, fourier_s8,
    ) = _sliders

    def _randomise_fourier(_):
        _rng = np.random.default_rng()
        set_fourier_values(tuple(
            round(float(_rng.uniform(-_limit, _limit)), 2)
            for _limit in _limits
        ))

    randomise_fourier = mo.ui.button(
        label="Randomise Fourier modes",
        on_click=_randomise_fourier,
        kind="success",
    )
    mo.vstack(
        [
            mo.hstack([mo.md("### Fourier radial controls"), randomise_fourier]),
            mo.hstack([fourier_c1, fourier_s1, fourier_c2, fourier_s2]),
            mo.hstack([fourier_c3, fourier_s3, fourier_c4, fourier_s4]),
            mo.hstack([fourier_c5, fourier_s5, fourier_c6, fourier_s6]),
            mo.hstack([fourier_c7, fourier_s7, fourier_c8, fourier_s8]),
        ]
    )
    return (
        fourier_c1,
        fourier_c2,
        fourier_c3,
        fourier_c4,
        fourier_c5,
        fourier_c6,
        fourier_c7,
        fourier_c8,
        fourier_s1,
        fourier_s2,
        fourier_s3,
        fourier_s4,
        fourier_s5,
        fourier_s6,
        fourier_s7,
        fourier_s8,
    )


@app.cell(hide_code=True)
def _():
    geometry = mo.ui.dropdown(
        ["Circle", "Ellipse", "Superellipse", "Fourier radial", "Annulus"],
        value="Fourier radial",
        label="Selected geometry",
    )
    grid_size = mo.ui.slider(48, 256, value=128, step=16, label="Grid size")
    radius = mo.ui.slider(0.15, 0.90, value=0.62, step=0.01, label="Radius")
    softness = mo.ui.slider(
        0.002, 0.10, value=0.025, step=0.001, label="Edge softness"
    )
    aspect = mo.ui.slider(
        0.30, 3.00, value=1.45, step=0.05, label="Aspect ratio"
    )
    angle_degrees = mo.ui.slider(
        0, 180, value=20, step=1, label="Rotation (degrees)"
    )
    exponent = mo.ui.slider(
        0.5, 10.0, value=4.0, step=0.1, label="Superellipse exponent"
    )
    hole_fraction = mo.ui.slider(
        0.05, 0.85, value=0.45, step=0.01, label="Annulus hole/radius"
    )
    mo.vstack(
        [
            mo.md("### Geometry controls"),
            mo.hstack([geometry, grid_size, radius, softness]),
            mo.hstack([aspect, angle_degrees, exponent, hole_fraction]),
        ]
    )
    return (
        angle_degrees,
        aspect,
        exponent,
        geometry,
        grid_size,
        hole_fraction,
        radius,
        softness,
    )


@app.cell(hide_code=True)
def _(
    dataset_error,
    profile_mode,
    reference_channel_names,
    reference_group_replicates,
    reference_measurement_names,
    reference_spatial_shape,
    texture_downsample,
):
    if profile_mode.value == "Uniform":
        _dataset_status = mo.md("Reference dataset loading is inactive.")
    elif dataset_error is not None:
        _dataset_status = mo.callout(dataset_error, kind="danger")
    else:
        _assignment_text = ", ".join(
            f"{_group}→replicate {_replicate}"
            for _group, _replicate in reference_group_replicates
        )
        _dataset_status = mo.callout(
            f"Loaded {len(reference_measurement_names)} measurements (including "
            f"duplicates) mapping to {len(reference_channel_names)} NCA state channels at "
            f"downsample={texture_downsample.value}; synthesis/design grid is "
            f"{reference_spatial_shape[0]}×{reference_spatial_shape[1]}. "
            f"Group assignment: {_assignment_text}.",
            kind="success",
        )
    _dataset_status
    return


@app.cell(hide_code=True)
def _(
    angle_degrees,
    aspect,
    exponent,
    fourier_c1,
    fourier_c2,
    fourier_c3,
    fourier_c4,
    fourier_c5,
    fourier_c6,
    fourier_c7,
    fourier_c8,
    fourier_s1,
    fourier_s2,
    fourier_s3,
    fourier_s4,
    fourier_s5,
    fourier_s6,
    fourier_s7,
    fourier_s8,
    hole_fraction,
    radius,
):
    mask_parameters = {
        "radius": radius.value,
        "aspect": aspect.value,
        "angle": np.deg2rad(angle_degrees.value),
        "exponent": exponent.value,
        "hole_fraction": hole_fraction.value,
        "fourier_cos": np.asarray(
            [
                fourier_c1.value,
                fourier_c2.value,
                fourier_c3.value,
                fourier_c4.value,
                fourier_c5.value,
                fourier_c6.value,
                fourier_c7.value,
                fourier_c8.value,
            ]
        ),
        "fourier_sin": np.asarray(
            [
                fourier_s1.value,
                fourier_s2.value,
                fourier_s3.value,
                fourier_s4.value,
                fourier_s5.value,
                fourier_s6.value,
                fourier_s7.value,
                fourier_s8.value,
            ]
        ),
    }
    return (mask_parameters,)


@app.cell(hide_code=True)
def _(
    binary_threshold,
    coordinate_grid,
    geometry,
    grid_size,
    make_level_set,
    mask_parameters,
    profile_mode,
    reference_spatial_shape,
    soft_occupancy,
    softness,
):
    _design_shape = (
        reference_spatial_shape
        if profile_mode.value != "Uniform"
        and reference_spatial_shape is not None
        else (grid_size.value, grid_size.value)
    )
    _x_grid, _y_grid = coordinate_grid(_design_shape)
    selected_level_set = make_level_set(
        geometry.value, _x_grid, _y_grid, mask_parameters
    )
    area_reference_pixels = None
    area_error_pixels = None
    fourier_area_scale = 1.0
    area_pixel_corrections = 0
    if geometry.value == "Fourier radial":
        _circle_level_set = make_level_set(
            "Circle", _x_grid, _y_grid, mask_parameters
        )
        _circle_mask = soft_occupancy(_circle_level_set, softness.value)
        area_reference_pixels = int(
            np.count_nonzero(_circle_mask >= binary_threshold.value)
        )
        _best = None
        _lower_scale, _upper_scale = 1.0e-4, 64.0
        for _ in range(64):
            _candidate_scale = 0.5 * (_lower_scale + _upper_scale)
            _candidate_parameters = dict(mask_parameters)
            _candidate_parameters["radius"] = (
                float(mask_parameters["radius"]) * _candidate_scale
            )
            _candidate_level_set = make_level_set(
                "Fourier radial", _x_grid, _y_grid, _candidate_parameters
            )
            _candidate_mask = soft_occupancy(
                _candidate_level_set, softness.value
            )
            _candidate_pixels = int(np.count_nonzero(
                _candidate_mask >= binary_threshold.value
            ))
            _candidate_error = abs(
                _candidate_pixels - area_reference_pixels
            )
            if _best is None or (_candidate_error, abs(_candidate_scale - 1.0)) < (
                _best[0], abs(_best[1] - 1.0)
            ):
                _best = (
                    _candidate_error,
                    _candidate_scale,
                    _candidate_level_set,
                    _candidate_pixels,
                )
            if _candidate_pixels < area_reference_pixels:
                _lower_scale = _candidate_scale
            else:
                _upper_scale = _candidate_scale
        _, fourier_area_scale, selected_level_set, _matched_pixels = _best

        # Raster ties can make the thresholded count jump over the target.
        # Move only the closest boundary pixels across the threshold so the
        # final discrete area remains exact without changing the interior.
        _threshold_level = float(softness.value) * np.log(
            1.0 / float(binary_threshold.value) - 1.0
        )
        _pixel_delta = area_reference_pixels - _matched_pixels
        if _pixel_delta:
            _flat_level_set = selected_level_set.reshape(-1).copy()
            _flat_inside = _flat_level_set <= _threshold_level
            _eligible = np.flatnonzero(
                ~_flat_inside if _pixel_delta > 0 else _flat_inside
            )
            _distances = np.abs(
                _flat_level_set[_eligible] - _threshold_level
            )
            _chosen = _eligible[
                np.argsort(_distances)[:abs(_pixel_delta)]
            ]
            _epsilon = max(float(softness.value) * 1.0e-6, 1.0e-9)
            _flat_level_set[_chosen] = _threshold_level + (
                -_epsilon if _pixel_delta > 0 else _epsilon
            )
            selected_level_set = _flat_level_set.reshape(_design_shape)
            area_pixel_corrections = int(abs(_pixel_delta))
    selected_mask = soft_occupancy(selected_level_set, softness.value)
    if area_reference_pixels is not None:
        _actual_pixels = int(np.count_nonzero(
            selected_mask >= binary_threshold.value
        ))
        area_error_pixels = _actual_pixels - area_reference_pixels
        if abs(area_error_pixels) > 1:
            raise RuntimeError(
                "Fourier mask area normalisation exceeded one-pixel tolerance"
            )
    return (
        area_error_pixels,
        area_pixel_corrections,
        area_reference_pixels,
        fourier_area_scale,
        selected_level_set,
        selected_mask,
    )


@app.cell(hide_code=True)
def _(
    dataset_error,
    normalise_histograms_and_covariance,
    profile_mode,
    reference_measurement_groups,
    reference_measurements,
    reference_primary_measurements,
    reference_support,
    selected_mask,
    synthesise_patch_texture,
    synthesise_reference_texture,
    texture_downsample,
    texture_histogram_normalisation,
    texture_interior_fraction,
    texture_iterations,
    texture_normalisation_iterations,
    texture_patch_candidates,
    texture_patch_jitter,
    texture_patch_overlap,
    texture_patch_reference_downsample,
    texture_patch_size,
    texture_scale_patch_geometry,
    texture_seed,
):
    synthetic_initial_channels = None
    synthetic_full_initial_condition = None
    synthetic_measurements = None
    effective_patch_size = None
    effective_patch_overlap = None
    effective_patch_jitter = None
    texture_error = dataset_error
    if profile_mode.value != "Uniform":
        if texture_error is None:
            try:
                if profile_mode.value == "Multichannel patch resampling":
                    _patch_scale = (
                        float(texture_patch_reference_downsample.value)
                        / float(texture_downsample.value)
                        if texture_scale_patch_geometry.value
                        else 1.0
                    )
                    _scaled_width = float(texture_patch_size.value) * _patch_scale
                    _scaled_patch_size = max(
                        3, 2 * int(round((_scaled_width - 1.0) / 2.0)) + 1
                    )
                    _scaled_patch_overlap = max(
                        0, int(round(float(texture_patch_overlap.value) * _patch_scale))
                    )
                    _scaled_patch_jitter = max(
                        0, int(round(float(texture_patch_jitter.value) * _patch_scale))
                    )
                    (
                        synthetic_measurements,
                        effective_patch_size,
                        effective_patch_overlap,
                        effective_patch_jitter,
                    ) = (
                        synthesise_patch_texture(
                            reference_measurements,
                            reference_support,
                            selected_mask.shape,
                            reference_measurement_groups,
                            _scaled_patch_size,
                            _scaled_patch_overlap,
                            float(texture_interior_fraction.value),
                            _scaled_patch_jitter,
                            int(texture_patch_candidates.value),
                            int(texture_seed.value),
                        )
                    )
                else:
                    synthetic_measurements = synthesise_reference_texture(
                        reference_measurements,
                        reference_support,
                        selected_mask.shape,
                        selected_mask,
                        int(texture_iterations.value),
                        int(texture_seed.value),
                    )
                if texture_histogram_normalisation.value:
                    synthetic_measurements = normalise_histograms_and_covariance(
                        synthetic_measurements,
                        reference_measurements,
                        reference_support,
                        selected_mask,
                        reference_measurement_groups,
                        int(texture_normalisation_iterations.value),
                    )
                synthetic_initial_channels = synthetic_measurements[
                    np.asarray(reference_primary_measurements)
                ]
                synthetic_full_initial_condition = (
                    synthetic_measurements * selected_mask[None]
                )
            except ValueError as _error:
                texture_error = str(_error)
    return (
        effective_patch_jitter,
        effective_patch_overlap,
        effective_patch_size,
        synthetic_full_initial_condition,
        synthetic_initial_channels,
        synthetic_measurements,
        texture_error,
    )


@app.cell(hide_code=True)
def _(
    binary_threshold,
    construct_initial_state,
    edge_value,
    edge_width,
    interior_value,
    observed_channels,
    profile_mode,
    selected_channel,
    selected_level_set,
    selected_mask,
    synthetic_initial_channels,
    texture_error,
    total_channels,
):
    _observed_count = int(observed_channels.value)
    _total_count = int(total_channels.value)
    _preview_channel = min(int(selected_channel.value), _observed_count - 1)
    if texture_error is not None:
        initial_state_error = texture_error
        initial_state = None
        binary_mask = None
    elif _total_count <= _observed_count:
        initial_state_error = (
            "Total channels must be greater than biological state channels; "
            "the current soft-boundary convention reserves the final channel "
            "for geometry."
        )
        initial_state = None
        binary_mask = None
    else:
        initial_state_error = None
        initial_state = construct_initial_state(
            selected_mask,
            selected_level_set,
            _total_count,
            _observed_count,
            _preview_channel,
            interior_value.value,
            edge_value.value,
            edge_width.value,
            profile_mode.value,
            synthetic_initial_channels,
        )
        binary_mask = selected_mask >= binary_threshold.value
    return binary_mask, initial_state, initial_state_error


@app.cell(hide_code=True)
def _(
    coordinate_grid,
    grid_size,
    make_level_set,
    mask_parameters,
    soft_occupancy,
    softness,
):
    _gallery_x, _gallery_y = coordinate_grid(grid_size.value)
    geometry_names = (
        "Circle",
        "Ellipse",
        "Superellipse",
        "Fourier radial",
        "Annulus",
    )
    gallery_masks = {
        _name: soft_occupancy(
            make_level_set(
                _name, _gallery_x, _gallery_y, mask_parameters
            ),
            softness.value,
        )
        for _name in geometry_names
    }
    return gallery_masks, geometry_names


@app.cell(hide_code=True)
def _(gallery_masks, geometry_names):
    _figure, _axes = plt.subplots(1, len(geometry_names), figsize=(16, 3.2))
    for _axis, _name in zip(_axes, geometry_names):
        _axis.imshow(
            gallery_masks[_name],
            origin="lower",
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            extent=(-1, 1, -1, 1),
        )
        _axis.set_title(_name)
        _axis.set_aspect("equal")
        _axis.set_xticks([])
        _axis.set_yticks([])
    _figure.suptitle("Parameterisation comparison with shared controls")
    _figure.tight_layout()
    _figure
    return


@app.cell(hide_code=True)
def _():
    mo.callout(
        "Patch resampling is preferred when discrete nuclei and other cellular "
        "structures matter: all channels from a co-measured panel share each "
        "patch transform. IAAFT remains available for stationary spectrum and "
        "histogram matching. The 14-channel measurement initial condition keeps "
        "duplicate stains separate; the recurrent 10-channel NCA state is derived "
        "with the schema's primary-measurement mapping. Hidden channels remain "
        "neutral and the final channel is the recurrent boundary mask.",
        kind="info",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Cell-type scoring controls

    Each cell type has an independent threshold and High/Low/Irrespective rule
    for every fate marker. Prevalence is the fraction of pixels inside the
    adhesion domain satisfying the complete cell-type definition.
    """)
    return


@app.cell(hide_code=True)
def _(rollout_comparison):
    _hour_options = (
        {f"{int(_hour)} h": int(_hour) for _hour in rollout_comparison["hours"]}
        if rollout_comparison is not None
        else {}
    )
    score_evaluation_hour = mo.ui.dropdown(
        options=_hour_options,
        value=next(reversed(_hour_options), None),
        label="Cell-type scoring time",
    )
    score_evaluation_hour
    return (score_evaluation_hour,)


@app.cell(hide_code=True)
def _(
    cell_type_marker_rules,
    cell_type_marker_thresholds,
    rollout_comparison,
    score_evaluation_hour,
    scored_cell_types,
):
    cell_type_score_outputs = None
    if rollout_comparison is None:
        _score_output = mo.md(
            "Run both NCA trajectories to calculate cell-type scores."
        )
    elif not scored_cell_types.value:
        _score_output = mo.callout(
            "Select at least one cell type of interest.", kind="warn"
        )
    elif score_evaluation_hour.value is None:
        _score_output = mo.callout("Select a scoring time.", kind="warn")
    else:
        _channel_names = rollout_comparison["channel_names"]
        _missing_markers = tuple(
            _marker for _marker in FATE_MARKERS if _marker not in _channel_names
        )
        if _missing_markers:
            raise ValueError(
                "Cell-type scoring requires channels: "
                + ", ".join(_missing_markers)
            )
        _selected_types = tuple(scored_cell_types.value)
        _hours = np.asarray(rollout_comparison["hours"])
        _time_matches = np.flatnonzero(
            _hours == int(score_evaluation_hour.value)
        )
        if not len(_time_matches):
            raise ValueError("The selected scoring time is absent from the rollout")
        _time_index = int(_time_matches[0])
        _rules = cell_type_marker_rules.value
        _thresholds = cell_type_marker_thresholds.value
        _trajectory_inputs = (
            (
                "Measured circle",
                rollout_comparison["circular_states"][_time_index],
                rollout_comparison["circular_boundary"][0] >= 0.5,
            ),
            (
                "Synthetic geometry",
                rollout_comparison["synthetic_states"][_time_index],
                rollout_comparison["synthetic_boundary"][0] >= 0.5,
            ),
        )
        _trajectory_scores = {}
        for _trajectory_label, _state, _colony in _trajectory_inputs:
            _masks = {}
            _prevalences = {}
            for _cell_type in _selected_types:
                _conditions = []
                for _marker in FATE_MARKERS:
                    _rule = _rules[_cell_type][_marker]
                    if _rule == "any":
                        continue
                    _marker_values = _state[_channel_names.index(_marker)]
                    _high = _marker_values > float(
                        _thresholds[_cell_type][_marker]
                    )
                    _conditions.append(_high if _rule == "high" else ~_high)
                _cell_mask = (
                    np.logical_and.reduce(_conditions)
                    if _conditions
                    else np.ones_like(_colony, dtype=bool)
                )
                _cell_mask = np.asarray(_cell_mask & _colony)
                _masks[_cell_type] = _cell_mask
                _prevalences[_cell_type] = float(np.mean(_cell_mask[_colony]))
            _prevalence_sum = sum(_prevalences.values())
            _composition = {
                _cell_type: (
                    _prevalence / _prevalence_sum
                    if _prevalence_sum > 0.0
                    else np.nan
                )
                for _cell_type, _prevalence in _prevalences.items()
            }
            _pairwise = {}
            for _left_index, _left in enumerate(_selected_types):
                for _right in _selected_types[_left_index + 1:]:
                    _denominator = _prevalences[_right]
                    _pairwise[f"{_left}:{_right}"] = (
                        _prevalences[_left] / _denominator
                        if _denominator > 0.0
                        else np.nan
                    )
            _trajectory_scores[_trajectory_label] = {
                "boundary": _colony,
                "masks": _masks,
                "prevalences": _prevalences,
                "composition_ratios": _composition,
                "pairwise_prevalence_ratios": _pairwise,
            }
        cell_type_score_outputs = {
            "evaluation_hour": int(score_evaluation_hour.value),
            "cell_types": _selected_types,
            "marker_thresholds": {
                _cell_type: dict(_thresholds[_cell_type])
                for _cell_type in _selected_types
            },
            "marker_rules": {
                _cell_type: dict(_rules[_cell_type])
                for _cell_type in _selected_types
            },
            "trajectories": _trajectory_scores,
        }

        def _format_score(_value):
            return "undefined" if not np.isfinite(_value) else f"{_value:.4f}"

        _header_cells = ["Pattern"]
        _header_cells.extend(
            f"{_cell_type} prevalence" for _cell_type in _selected_types
        )
        _header_cells.extend(
            f"{_cell_type} share" for _cell_type in _selected_types
        )
        _ratio_names = tuple(next(iter(_trajectory_scores.values()))[
            "pairwise_prevalence_ratios"
        ])
        _header_cells.extend(f"{_ratio} ratio" for _ratio in _ratio_names)
        _table_lines = [
            "| " + " | ".join(_header_cells) + " |",
            "| " + " | ".join(["---"] * len(_header_cells)) + " |",
        ]
        for _trajectory_label, _scores in _trajectory_scores.items():
            _values = [_trajectory_label]
            _values.extend(
                _format_score(_scores["prevalences"][_cell_type])
                for _cell_type in _selected_types
            )
            _values.extend(
                _format_score(_scores["composition_ratios"][_cell_type])
                for _cell_type in _selected_types
            )
            _values.extend(
                _format_score(_scores["pairwise_prevalence_ratios"][_ratio])
                for _ratio in _ratio_names
            )
            _table_lines.append("| " + " | ".join(_values) + " |")
        _score_output = mo.vstack([
            mo.md(
                f"#### Cell-type scalar scores at {score_evaluation_hour.value} h"
            ),
            mo.md("\n".join(_table_lines)),
            mo.md(
                "Shares are normalised by the sum of selected prevalences. "
                "Because cell-type rules may overlap, shares are descriptive "
                "ratios rather than an exclusive partition."
            ),
        ])
    _score_output
    return (cell_type_score_outputs,)


@app.cell(hide_code=True)
def _(cell_type_score_outputs):
    if cell_type_score_outputs is None:
        _map_output = mo.md("")
    else:
        _colors = {
            "Notochord": np.asarray([214, 39, 160], dtype=float) / 255.0,
            "Endoderm": np.asarray([23, 190, 207], dtype=float) / 255.0,
            "Mesoderm": np.asarray([44, 160, 44], dtype=float) / 255.0,
        }
        _cell_types = cell_type_score_outputs["cell_types"]
        _trajectories = cell_type_score_outputs["trajectories"]
        _figure, _axes = plt.subplots(
            len(_trajectories),
            len(_cell_types) + 1,
            figsize=(3.0 * (len(_cell_types) + 1), 3.0 * len(_trajectories)),
            squeeze=False,
            constrained_layout=True,
        )
        for _row, (_trajectory_label, _scores) in enumerate(
            _trajectories.items()
        ):
            _boundary = _scores["boundary"]
            _composite = np.zeros((*_boundary.shape, 3), dtype=float)
            _occupancy_count = np.zeros_like(_boundary, dtype=np.int16)
            for _column, _cell_type in enumerate(_cell_types):
                _mask = _scores["masks"][_cell_type]
                _image = np.zeros((*_boundary.shape, 3), dtype=float)
                _image[_mask] = _colors[_cell_type]
                _axes[_row, _column].imshow(_image, vmin=0.0, vmax=1.0)
                _axes[_row, _column].set_title(_cell_type)
                _composite[_mask] += _colors[_cell_type]
                _occupancy_count += _mask.astype(np.int16)
            _composite = np.clip(_composite, 0.0, 1.0)
            _composite[_occupancy_count > 1] = 1.0
            _axes[_row, -1].imshow(_composite, vmin=0.0, vmax=1.0)
            _axes[_row, -1].set_title("Combined; overlaps white")
            _axes[_row, 0].set_ylabel(_trajectory_label)
            for _axis in _axes[_row]:
                _axis.set_xticks([])
                _axis.set_yticks([])
        _figure.suptitle(
            "Cell-type locations at "
            f"{cell_type_score_outputs['evaluation_hour']} h"
        )
        _map_output = _figure
    _map_output
    plt.show()
    return


@app.cell(hide_code=True)
def _():
    scored_cell_types = mo.ui.multiselect(
        options=list(CELL_TYPES),
        value=list(CELL_TYPES),
        label="Cell types of interest",
        full_width=True,
    )
    cell_type_marker_thresholds = mo.ui.dictionary({
        _cell_type: mo.ui.dictionary({
            _marker: mo.ui.slider(
                0.0,
                1.0,
                value=0.3,
                step=0.025,
                label=f"{_marker} threshold",
                full_width=True,
            )
            for _marker in FATE_MARKERS
        })
        for _cell_type in CELL_TYPES
    })
    cell_type_marker_rules = mo.ui.dictionary({
        _cell_type: mo.ui.dictionary({
            _marker: mo.ui.dropdown(
                {"High": "high", "Low": "low", "Irrespective": "any"},
                value={"high": "High", "low": "Low", "any": "Irrespective"}[
                    DEFAULT_FATE_RULES[_cell_type][_marker]
                ],
                label=_marker,
                full_width=True,
            )
            for _marker in FATE_MARKERS
        })
        for _cell_type in CELL_TYPES
    })
    mo.vstack([
        scored_cell_types,
        mo.md("#### Marker thresholds by cell type"),
        cell_type_marker_thresholds,
        mo.md("#### Marker-state rules by cell type"),
        cell_type_marker_rules,
    ])
    return (
        cell_type_marker_rules,
        cell_type_marker_thresholds,
        scored_cell_types,
    )


@app.cell(column=1, hide_code=True)
def _():
    total_channels = mo.ui.number(
        11, 128, value=48, step=1, label="Total NCA channels"
    )
    observed_channels = mo.ui.number(
        1, 10, value=10, step=1, label="Biological state channels"
    )
    selected_channel = mo.ui.number(
        0, 9, value=3, step=1, label="Preview biological channel"
    )
    profile_mode = mo.ui.dropdown(
        ["Uniform", "Multichannel patch resampling", "Reference texture (IAAFT)"],
        value="Multichannel patch resampling",
        label="Initial profile",
    )
    interior_value = mo.ui.slider(
        0.0, 1.0, value=0.70, step=0.01, label="Interior value"
    )
    edge_value = mo.ui.slider(
        0.0, 1.0, value=0.70, step=0.01, label="Edge value"
    )
    edge_width = mo.ui.slider(
        0.01, 0.50, value=0.12, step=0.01, label="Edge-profile width"
    )
    binary_threshold = mo.ui.slider(
        0.05, 0.95, value=0.50, step=0.05, label="Binary threshold"
    )
    mo.vstack(
        [
            mo.md("### Initial-condition controls"),
            mo.hstack([total_channels, observed_channels, selected_channel]),
            mo.hstack(
                [
                    profile_mode,
                    interior_value,
                    edge_value,
                    edge_width,
                    binary_threshold,
                ]
            ),
        ]
    )
    return (
        binary_threshold,
        edge_value,
        edge_width,
        interior_value,
        observed_channels,
        profile_mode,
        selected_channel,
        total_channels,
    )


@app.cell(hide_code=True)
def _():
    _data_base = os.environ.get("DATA_PATH_BASE")
    _default_root = (
        str(Path(_data_base) / "260726_nca_dataset")
        if _data_base
        else "../Data/260726_nca_dataset"
    )
    micropattern_root = mo.ui.text(
        value=_default_root,
        label="260726 micropattern dataset root",
        full_width=True,
    )
    reference_condition = mo.ui.dropdown(
        options={
            "Control": "ctrl",
            "Nodal knockout at 0 h": "sl0",
            "Nodal knockout at 24 h": "sl24",
        },
        value="Control",
        label="Condition",
    )
    reference_replicates = mo.ui.slider(
        1, 4, value=3, step=1, label="Replicates to load"
    )
    reference_sample = mo.ui.number(0, 32, value=0, step=1, label="Reference sample")
    reference_replicate_mode = mo.ui.dropdown(
        options={
            "Fixed sample": "fixed",
            "Sample replicate per experiment group": "sample_groups",
        },
        value="Sample replicate per experiment group",
        label="Replicate mode",
    )
    reference_replicate_seed = mo.ui.number(
        0, 100000, value=0, step=1, label="Replicate sampling seed"
    )
    reference_time = mo.ui.dropdown(
        options={"0 h": 0, "12 h": 1, "24 h": 2, "36 h": 3, "48 h": 4},
        value="0 h",
        label="Reference time",
    )
    texture_downsample = mo.ui.dropdown(
        options=[1, 2, 4, 8, 16, 32], value=4, label="Data downsampling"
    )
    texture_iterations = mo.ui.slider(
        0, 100, value=25, step=5, label="IAAFT iterations"
    )
    texture_patch_size = mo.ui.slider(
        3, 31, value=11, step=2, label="Patch width at reference scale (pixels)"
    )
    texture_patch_overlap = mo.ui.slider(
        0, 30, value=7, step=1, label="Patch overlap at reference scale (pixels)"
    )
    texture_interior_fraction = mo.ui.slider(
        0.40,
        1.00,
        value=0.80,
        step=0.01,
        label="Patch sampling radius fraction",
    )
    texture_patch_jitter = mo.ui.slider(
        0, 12, value=3, step=1, label="Patch jitter at reference scale (pixels)"
    )
    texture_patch_candidates = mo.ui.slider(
        1, 64, value=24, step=1, label="Quilting candidates"
    )
    texture_scale_patch_geometry = mo.ui.switch(
        value=True, label="Scale patch geometry with downsampling"
    )
    texture_patch_reference_downsample = mo.ui.dropdown(
        options=[1, 2, 4, 8, 16, 32],
        value=4,
        label="Patch-control reference downsample",
    )
    texture_histogram_normalisation = mo.ui.switch(
        value=False, label="Match histograms + covariance"
    )
    texture_normalisation_iterations = mo.ui.slider(
        1, 12, value=4, step=1, label="Normalisation iterations"
    )
    texture_seed = mo.ui.number(0, 100000, value=0, step=1, label="Texture seed")
    mo.vstack(
        [
            mo.md("### Circular reference texture"),
            mo.md(
                "Used by both measured-texture modes. "
                "Data are loaded through `load_micropattern_260726`. The selected "
                "downsampling sets both the reference texture and design-grid resolution."
            ),
            micropattern_root,
            mo.hstack(
                [
                    reference_condition,
                    reference_replicates,
                    reference_sample,
                    reference_time,
                    reference_replicate_mode,
                    reference_replicate_seed,
                ]
            ),
            mo.hstack(
                [
                    texture_downsample,
                    texture_patch_size,
                    texture_patch_overlap,
                    texture_interior_fraction,
                    texture_patch_jitter,
                    texture_patch_candidates,
                ]
            ),
            mo.hstack(
                [
                    texture_scale_patch_geometry,
                    texture_patch_reference_downsample,
                    texture_histogram_normalisation,
                    texture_normalisation_iterations,
                    texture_iterations,
                    texture_seed,
                ]
            ),
        ]
    )
    return (
        micropattern_root,
        reference_condition,
        reference_replicate_mode,
        reference_replicate_seed,
        reference_replicates,
        reference_sample,
        reference_time,
        texture_downsample,
        texture_histogram_normalisation,
        texture_interior_fraction,
        texture_iterations,
        texture_normalisation_iterations,
        texture_patch_candidates,
        texture_patch_jitter,
        texture_patch_overlap,
        texture_patch_reference_downsample,
        texture_patch_size,
        texture_scale_patch_geometry,
        texture_seed,
    )


@app.cell(hide_code=True)
def _(
    area_error_pixels,
    area_pixel_corrections,
    area_reference_pixels,
    binary_mask,
    fourier_area_scale,
    geometry,
    initial_state,
    initial_state_error,
    observed_channels,
    selected_channel,
    selected_mask,
):
    if initial_state_error is not None:
        _output = mo.callout(initial_state_error, kind="danger")
    else:
        _channel = min(
            int(selected_channel.value), int(observed_channels.value) - 1
        )
        _figure, _axes = plt.subplots(1, 4, figsize=(18, 6))
        _images = (
            (selected_mask, "Soft adhesion mask", "viridis", 0.0, 1.0),
            (binary_mask, "Thresholded mask", "gray", 0.0, 1.0),
            (
                initial_state[_channel],
                f"Initial biological channel {_channel}",
                "magma",
                0.0,
                None,
            ),
            (initial_state[-1], "Initial geometry channel −1", "viridis", 0.0, 1.0),
        )
        for _axis, (_image, _title, _cmap, _minimum, _maximum) in zip(
            _axes, _images
        ):
            _artist = _axis.imshow(
                _image,
                origin="lower",
                cmap=_cmap,
                vmin=_minimum,
                vmax=_maximum,
                extent=(-1, 1, -1, 1),
            )
            _axis.set_title(_title)
            _axis.set_aspect("equal")
            _axis.set_xticks([])
            _axis.set_yticks([])
            _figure.colorbar(_artist, ax=_axis, fraction=0.046, pad=0.04)
        _figure.suptitle(f"Selected parameterisation: {geometry.value}")
        _figure.tight_layout()
        _soft_area = float(np.mean(selected_mask))
        _hard_area = float(np.mean(binary_mask))
        _soft_tail_max = float(
            np.max(np.abs(initial_state[:-1] * (~binary_mask)))
        )
        _area_note = (
            f"**Circle-reference area:** {area_reference_pixels} pixels  \n"
            f"**Fourier area error:** {area_error_pixels:+d} pixels "
            f"(radius scale {fourier_area_scale:.6f}; "
            f"{area_pixel_corrections} boundary-pixel corrections)  \n"
            if area_reference_pixels is not None
            else ""
        )
        _metrics = mo.md(
            f"""
            **Soft area fraction:** {_soft_area:.4f}  
            **Thresholded area fraction:** {_hard_area:.4f}  
            {_area_note}
            **Maximum biological soft-edge tail below threshold:** {_soft_tail_max:.3e}  
            **Initial state shape:** `{initial_state.shape}`
            """
        )
        _output = mo.vstack([_figure, _metrics])
    _output
    plt.show()
    return


@app.cell(hide_code=True)
def _(
    binary_mask,
    extend_masked_texture,
    observed_channels,
    reference_channels,
    reference_support,
    selected_channel,
    synthetic_initial_channels,
):
    if synthetic_initial_channels is None:
        _comparison = mo.md(
            "Select a measured-texture mode to inspect texture-matching diagnostics."
        )
    else:
        _channel = min(
            int(selected_channel.value),
            int(observed_channels.value) - 1,
            reference_channels.shape[0] - 1,
        )
        _reference_values = reference_channels[_channel][reference_support]
        _synthetic_values = synthetic_initial_channels[_channel][binary_mask]
        _reference_filled = extend_masked_texture(
            reference_channels[_channel], reference_support
        )
        _reference_display = np.where(
            reference_support, reference_channels[_channel], np.nan
        )
        _reference_cmap = plt.get_cmap("magma").copy()
        _reference_cmap.set_bad(color="#d9d9d9")
        _synthetic_filled = extend_masked_texture(
            synthetic_initial_channels[_channel], binary_mask
        )
        _reference_power = np.log1p(
            np.abs(np.fft.fftshift(np.fft.fft2(_reference_filled))) ** 2
        )
        _synthetic_power = np.log1p(
            np.abs(np.fft.fftshift(np.fft.fft2(_synthetic_filled))) ** 2
        )
        def _high_frequency_fraction(_image):
            _centred = _image - np.mean(_image)
            _power = np.abs(np.fft.fft2(_centred)) ** 2
            _frequency_y = np.fft.fftfreq(_image.shape[0])[:, None]
            _frequency_x = np.fft.fftfreq(_image.shape[1])[None, :]
            _radius = np.sqrt(_frequency_x**2 + _frequency_y**2)
            _total = np.sum(_power)
            return float(np.sum(_power[_radius >= 0.25]) / max(_total, 1.0e-12))

        _reference_high_frequency = _high_frequency_fraction(_reference_filled)
        _synthetic_high_frequency = _high_frequency_fraction(_synthetic_filled)
        _frequency_kind = (
            "warn"
            if _synthetic_high_frequency > 1.2 * _reference_high_frequency
            else "success"
        )
        _frequency_note = mo.callout(
            "High-frequency power is within 20% of the reference."
            if _frequency_kind == "success"
            else "Synthetic high-frequency power exceeds the reference by more "
            "than 20%; adjust patch width/overlap or IAAFT iterations and inspect "
            "this channel before rollout.",
            kind=_frequency_kind,
        )
        _figure, _axes = plt.subplots(1, 4, figsize=(15, 3.4))
        _axes[0].hist(
            _reference_values, bins=50, density=True, alpha=0.55, label="reference"
        )
        _axes[0].hist(
            _synthetic_values, bins=50, density=True, alpha=0.55, label="synthetic"
        )
        _axes[0].set_title(f"Channel {_channel} distribution")
        _axes[0].legend()
        _axes[1].imshow(_reference_display, cmap=_reference_cmap, origin="lower")
        _axes[1].set_title("Measured circular reference")
        _axes[2].imshow(_reference_power, cmap="viridis", origin="lower")
        _axes[2].set_title("Reference log power")
        _axes[3].imshow(_synthetic_power, cmap="viridis", origin="lower")
        _axes[3].set_title("Synthetic log power")
        for _axis in _axes[1:]:
            _axis.set_xticks([])
            _axis.set_yticks([])
        _figure.tight_layout()
        _comparison = mo.vstack(
            [
                mo.md("### Reference-texture diagnostics"),
                _figure,
                mo.md(
                    f"Reference mean/std: `{np.mean(_reference_values):.4f}` / "
                    f"`{np.std(_reference_values):.4f}` · Synthetic mean/std: "
                    f"`{np.mean(_synthetic_values):.4f}` / "
                    f"`{np.std(_synthetic_values):.4f}`  \n"
                    f"Power above 0.5 Nyquist — reference: "
                    f"`{_reference_high_frequency:.3%}` · synthetic: "
                    f"`{_synthetic_high_frequency:.3%}`"
                ),
                _frequency_note,
            ]
        )
    _comparison
    return


@app.cell(hide_code=True)
def _(
    binary_mask,
    effective_patch_jitter,
    effective_patch_overlap,
    effective_patch_size,
    profile_mode,
    reference_measurement_groups,
    reference_measurement_names,
    reference_measurements,
    reference_support,
    synthetic_full_initial_condition,
    synthetic_measurements,
    texture_histogram_normalisation,
    texture_interior_fraction,
    texture_patch_candidates,
    texture_scale_patch_geometry,
):
    if synthetic_measurements is None:
        _full_measurement_output = mo.md(
            "Select a measured-texture mode to inspect all measurement channels."
        )
    else:
        _figure, _axes = plt.subplots(2, 7, figsize=(18, 5.4))
        for _index, _axis in enumerate(_axes.flat):
            _axis.imshow(
                synthetic_full_initial_condition[_index],
                cmap="magma",
                origin="lower",
            )
            _axis.set_title(reference_measurement_names[_index], fontsize=8)
            _axis.set_xticks([])
            _axis.set_yticks([])
        _figure.suptitle("Synthetic 14-channel measurement tensor (duplicates retained)")
        _figure.tight_layout()

        _reference_correlation = np.full((14, 14), np.nan, dtype=np.float32)
        _synthetic_correlation = np.full((14, 14), np.nan, dtype=np.float32)
        for _group in reference_measurement_groups:
            _indices = np.asarray(_group, dtype=int)
            _reference_group = reference_measurements[_indices][:, reference_support]
            _synthetic_group = synthetic_measurements[_indices][:, binary_mask]
            _reference_correlation[np.ix_(_indices, _indices)] = np.corrcoef(
                _reference_group
            )
            _synthetic_correlation[np.ix_(_indices, _indices)] = np.corrcoef(
                _synthetic_group
            )
        _correlation_figure, _correlation_axes = plt.subplots(1, 3, figsize=(13, 3.8))
        _correlation_images = (
            (_reference_correlation, "Co-measured reference correlation", "coolwarm", -1, 1),
            (_synthetic_correlation, "Synthetic correlation", "coolwarm", -1, 1),
            (
                _synthetic_correlation - _reference_correlation,
                "Synthetic − reference",
                "coolwarm",
                -0.5,
                0.5,
            ),
        )
        for _axis, (_matrix, _title, _cmap, _minimum, _maximum) in zip(
            _correlation_axes, _correlation_images
        ):
            _artist = _axis.imshow(
                _matrix, cmap=_cmap, vmin=_minimum, vmax=_maximum
            )
            _axis.set_title(_title, fontsize=9)
            _axis.set_xlabel("Measurement channel")
            _axis.set_ylabel("Measurement channel")
            _correlation_figure.colorbar(_artist, ax=_axis, fraction=0.046)
        _correlation_figure.tight_layout()
        _patch_note = (
            f" Effective patch width: **{effective_patch_size} px**; every source "
            f"patch lies within the central **{texture_interior_fraction.value:.0%}** "
            f"of the reference radius. Effective overlap: "
            f"**{effective_patch_overlap} px**; effective placement jitter: "
            f"**±{effective_patch_jitter} px** (capped by overlap to prevent gaps); "
            f"**{texture_patch_candidates.value}** candidates per placement. "
            f"Resolution scaling is **{'enabled' if texture_scale_patch_geometry.value else 'disabled'}**."
            if profile_mode.value == "Multichannel patch resampling"
            else ""
        )
        _normalisation_note = (
            " Joint marginal-histogram/covariance normalisation is **enabled**."
            if texture_histogram_normalisation.value
            else " Joint marginal-histogram/covariance normalisation is **disabled**."
        )
        _full_measurement_output = mo.vstack(
            [
                mo.md(
                    "### Full synthetic biological measurements\n\n"
                    "Correlation blocks are shown only within genuinely co-measured "
                    f"staining panels; cross-panel entries are intentionally blank.{_patch_note}  \n"
                    f"{_normalisation_note}  \n"
                    f"Full measurement initial condition shape: "
                    f"`{synthetic_full_initial_condition.shape}`"
                ),
                _figure,
                _correlation_figure,
            ]
        )
    _full_measurement_output
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## NCA rollout comparison

    Load one NCA from a YAML selection exported by `model_registry_explorer.py`,
    then compare matched rollouts from the measured circular reference and the
    generated synthetic geometry.
    """)
    return


@app.cell(hide_code=True)
def _():
    _default_store = os.environ.get(
        "MODEL_STORE_ROOT", str(Path(__file__).resolve().parents[2] / "models")
    )
    model_selection_path = mo.ui.text(
        placeholder="/path/to/model_registry_selection.yaml",
        label="Exported model selection",
        full_width=True,
    )
    model_store_root = mo.ui.text(
        value=_default_store, label="Model store", full_width=True
    )
    mo.vstack([model_selection_path, model_store_root])
    return model_selection_path, model_store_root


@app.cell(hide_code=True)
def _(model_selection_path):
    _path_text = model_selection_path.value.strip()
    model_selection_records = ()
    if not _path_text:
        _status = mo.md("Choose an exported model-selection YAML file.")
    elif not Path(_path_text).expanduser().is_file():
        _status = mo.callout(
            f"Selection file not found: `{Path(_path_text).expanduser()}`",
            kind="danger",
        )
    else:
        try:
            _document = OmegaConf.to_container(
                OmegaConf.load(Path(_path_text).expanduser()), resolve=False
            )
            if not isinstance(_document, dict) or _document.get("schema_version") != 1:
                raise ValueError(
                    "Expected a model registry export with schema_version: 1"
                )
            model_selection_records = tuple(_document.get("models", ()))
            if not model_selection_records:
                raise ValueError("The model selection contains no models")
            if any("model_id" not in _record for _record in model_selection_records):
                raise ValueError("Every selected model must contain model_id")
            _status = mo.callout(
                f"Loaded {len(model_selection_records)} model selection(s).",
                kind="success",
            )
        except (OSError, ValueError, TypeError, KeyError) as _error:
            model_selection_records = ()
            _status = mo.callout(str(_error), kind="danger")
    _status
    return (model_selection_records,)


@app.cell(hide_code=True)
def _(model_selection_records):
    _options = {}
    for _record in model_selection_records:
        _model_id = str(_record["model_id"])
        _label = (
            _record.get("notes")
            or _record.get("alias")
            or _record.get("display_name")
            or _model_id
        )
        _options[f"{_label} [{_model_id[-8:]}]"] = _model_id
    selected_model_id = mo.ui.dropdown(
        options=_options,
        value=next(iter(_options), None),
        label="NCA model",
        full_width=True,
    )
    rollout_hours = mo.ui.dropdown(
        options=[12, 24, 36, 48, 60, 72], value=48,
        label="Rollout horizon (hours)",
    )
    rollout_seed = mo.ui.number(
        0, 2**31 - 1, value=0, step=1, label="Rollout seed"
    )
    run_rollout_comparison = mo.ui.run_button(label="Run both NCA rollouts")
    mo.vstack([
        selected_model_id,
        mo.hstack([rollout_hours, rollout_seed, run_rollout_comparison]),
    ])
    return (
        rollout_hours,
        rollout_seed,
        run_rollout_comparison,
        selected_model_id,
    )


@app.cell(hide_code=True)
def _(reference_channel_names):
    _channel_options = list(reference_channel_names or ())
    _default_channels = [
        _name
        for _name in ("SOX2", "TBXT", "SOX17")
        if _name in _channel_options
    ]
    rollout_display_channels = mo.ui.multiselect(
        options=_channel_options,
        value=_default_channels,
        label="Displayed rollout channels",
        full_width=True,
    )
    rollout_display_channels
    return (rollout_display_channels,)


@app.cell(hide_code=True)
def _(model_selection_records, selected_model_id, texture_downsample):
    if selected_model_id.value is None or not model_selection_records:
        _downsample_status = mo.md(
            "Select a model to inspect its saved training downsampling."
        )
    else:
        try:
            _selected_record = next(
                (
                    _record
                    for _record in model_selection_records
                    if str(_record["model_id"]) == selected_model_id.value
                ),
                None,
            )
            if _selected_record is None:
                raise ValueError("The selected model is absent from the export")
            _bundle_path = _selected_record.get("path")
            if not _bundle_path:
                raise ValueError(
                    "The selection export does not contain the bundle path"
                )
            _config_path = Path(str(_bundle_path)).expanduser() / "config.yaml"
            if not _config_path.is_file():
                raise FileNotFoundError(
                    f"Model bundle config not found: {_config_path}"
                )
            _saved_config = OmegaConf.load(_config_path)
            _training_downsample = OmegaConf.select(
                _saved_config, "data.preprocessing.downsample"
            )
            if _training_downsample is None:
                _training_downsample = OmegaConf.select(
                    _saved_config, "data.downsample"
                )
            if _training_downsample is None:
                _downsample_status = mo.callout(
                    "This model bundle does not contain a saved training "
                    "downsampling ratio.",
                    kind="warn",
                )
            else:
                _training_downsample = int(_training_downsample)
                _current_downsample = int(texture_downsample.value)
                _matches = _training_downsample == _current_downsample
                _downsample_status = mo.callout(
                    f"Training data downsampling: **×{_training_downsample}** · "
                    f"Current notebook downsampling: **×{_current_downsample}**. "
                    + (
                        "The resolutions match."
                        if _matches
                        else "The resolutions differ."
                    ),
                    kind="success" if _matches else "warn",
                )
        except (OSError, ValueError, TypeError, KeyError) as _error:
            _downsample_status = mo.callout(
                f"Could not read model training downsampling: {_error}",
                kind="danger",
            )
    _downsample_status
    return


@app.cell(hide_code=True)
def _(
    initial_state_error,
    model_selection_records,
    model_store_root,
    reference_channel_names,
    reference_channels,
    reference_support,
    rollout_hours,
    rollout_seed,
    run_rollout_comparison,
    selected_mask,
    selected_model_id,
    synthetic_initial_channels,
):
    rollout_comparison = None
    if not run_rollout_comparison.value:
        _status = mo.md("Select a model and click **Run both NCA rollouts**.")
    else:
        try:
            if selected_model_id.value is None or not model_selection_records:
                raise ValueError("Choose a model from a non-empty selection")
            if initial_state_error is not None:
                raise ValueError(initial_state_error)
            if reference_channels is None or reference_support is None:
                raise ValueError(
                    "A measured-texture profile is required for the circular input"
                )
            if synthetic_initial_channels is None:
                raise ValueError("Generate a measured synthetic texture first")

            _bundle = ModelRegistry(
                Path(model_store_root.value).expanduser()
            ).get(selected_model_id.value)
            _model = _bundle.load_model(
                key=jr.PRNGKey(int(rollout_seed.value)), implementation="portable"
            )
            _n_channels = int(_model.N_CHANNELS)
            _n_biological = int(reference_channels.shape[0])
            if synthetic_initial_channels.shape[0] != _n_biological:
                raise ValueError("Circular and synthetic channel counts do not match")
            if _n_channels <= _n_biological:
                raise ValueError(
                    f"Model has {_n_channels} channels but needs more than "
                    f"{_n_biological} for hidden/boundary channels"
                )
            if reference_channels.shape[-2:] != selected_mask.shape:
                raise ValueError("Circular and synthetic spatial shapes do not match")

            _boundary_mode = _bundle.config.trainer.boundary_mode
            _circle_boundary = np.asarray(reference_support, np.float32)[None]
            _shape_boundary = np.asarray(selected_mask, np.float32)[None]
            _circle_biology = (
                np.asarray(reference_channels, np.float32) * _circle_boundary
            )
            _shape_biology = (
                np.asarray(synthetic_initial_channels, np.float32) * _shape_boundary
            )

            def _make_state(_biology, _boundary):
                _state = np.zeros(
                    (_n_channels, *_biology.shape[-2:]), dtype=np.float32
                )
                _state[:_n_biological] = _biology
                if _boundary_mode == "soft":
                    _state[-_boundary.shape[0]:] = _boundary
                return jnp.asarray(_state)

            _steps_per_12h = int(_bundle.config.run.t)
            _hours = np.arange(0, int(rollout_hours.value) + 1, 12, dtype=int)
            _observation_steps = jnp.asarray(
                (_hours // 12) * _steps_per_12h, dtype=jnp.int32
            )
            _total_steps = int(_observation_steps[-1])
            _shared_key = jr.PRNGKey(int(rollout_seed.value))
            _circle_states = rollout_model_sampled(
                _model,
                _make_state(_circle_biology, _circle_boundary),
                jnp.asarray(_circle_boundary),
                _boundary_mode,
                _shared_key,
                _total_steps,
                _observation_steps,
            )
            _shape_states = rollout_model_sampled(
                _model,
                _make_state(_shape_biology, _shape_boundary),
                jnp.asarray(_shape_boundary),
                _boundary_mode,
                _shared_key,
                _total_steps,
                _observation_steps,
            )
            rollout_comparison = {
                "model_id": _bundle.id,
                "boundary_mode": _boundary_mode,
                "channel_names": tuple(reference_channel_names),
                "hours": _hours,
                "circular_boundary": _circle_boundary,
                "synthetic_boundary": _shape_boundary,
                "circular_states": np.asarray(_circle_states),
                "synthetic_states": np.asarray(_shape_states),
            }
            _status = mo.callout(
                f"Completed both {_total_steps}-step rollouts for model "
                f"`{_bundle.id}` using boundary mode `{_boundary_mode}`.",
                kind="success",
            )
        except (OSError, ValueError, TypeError, KeyError, IndexError) as _error:
            rollout_comparison = None
            _status = mo.callout(str(_error), kind="danger")
    _status
    return (rollout_comparison,)


@app.cell(hide_code=True)
def _(rollout_comparison, rollout_display_channels):
    if rollout_comparison is None:
        _output = mo.md(
            "Rollout results will appear here after both trajectories complete."
        )
    elif not rollout_display_channels.value:
        _output = mo.callout(
            "Select at least one biological channel to display.", kind="warn"
        )
    else:
        _channel_names = rollout_comparison["channel_names"]
        _selected_names = tuple(rollout_display_channels.value)
        _missing = tuple(
            _name for _name in _selected_names if _name not in _channel_names
        )
        if _missing:
            raise ValueError(
                "Rollout results do not contain channels: " + ", ".join(_missing)
            )
        _hours = rollout_comparison["hours"]
        _trajectories = (
            ("Measured circle", rollout_comparison["circular_states"]),
            ("Synthetic geometry", rollout_comparison["synthetic_states"]),
        )
        _figures = []
        for _trajectory_label, _states in _trajectories:
            _figure, _axes = plt.subplots(
                len(_selected_names),
                len(_hours),
                figsize=(2.4 * len(_hours), 2.2 * len(_selected_names)),
                squeeze=False,
                constrained_layout=True,
            )
            for _row, _channel_name in enumerate(_selected_names):
                _channel = _channel_names.index(_channel_name)
                for _column, _hour in enumerate(_hours):
                    _axis = _axes[_row, _column]
                    _axis.imshow(
                        np.clip(_states[_column, _channel], 0.0, 1.0),
                        cmap="gray",
                        origin="lower",
                        vmin=0.0,
                        vmax=1.0,
                    )
                    if _row == 0:
                        _axis.set_title(f"{_hour} h")
                    if _column == 0:
                        _axis.set_ylabel(_channel_name)
                    _axis.set_xticks([])
                    _axis.set_yticks([])
            _figure.suptitle(_trajectory_label)
            _figures.append(_figure)
        _output = mo.vstack([
            mo.md(
                f"Model: `{rollout_comparison['model_id']}` · boundary mode: "
                f"`{rollout_comparison['boundary_mode']}` · grayscale range: "
                "`0` (black) to `1` (white)"
            ),
            *_figures,
        ])
    _output
    plt.show()
    return


if __name__ == "__main__":
    app.run()
