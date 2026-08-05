import matplotlib

matplotlib.use("Agg")

import jax
import jax.numpy as jnp
import numpy as np
from unittest.mock import patch

from NCA.model.NCA_fast_KAN_model import FastKaNCA
from NCA.model.NCA_model import NCA
from Common.dataloader.micropattern_schemas import MICROPATTERN_260726_SCHEMA
from NCA.trainer.trainer import select_wandb_train_logger_class
from NCA.trainer.logging.tensorboard import (
    NCA_Train_log,
    NCA_knockout_Train_log,
    _biomarker_name,
    _target_aligned_diagnostic_channels,
    _trajectory_snapshot_channels,
    compute_channel_correlation_diagnostics,
    compute_channel_time_diagnostics,
    plot_channel_correlation_diagnostics,
    plot_channel_time_grid,
    plot_radial_intensity_diagnostics,
    plot_radial_intensity_line_diagnostics,
    plot_total_intensity_diagnostics,
)
from NCA.trainer.logging.kan_tensorboard import (
    kaNCA_Train_log,
    uses_fast_kan_diagnostics,
)


def _logger_without_wandb():
    logger = object.__new__(kaNCA_Train_log)
    logger.diagnostic_targets = None
    return logger


def test_fast_kan_logger_selection_uses_kan_logger():
    key = jax.random.PRNGKey(0)
    fast_kan_nca = FastKaNCA(
        4,
        KERNEL_STR=["ID", "LAP"],
        KAN_AUX={"num_basis": 4},
        key=key,
    )
    nca = NCA(4, KERNEL_STR=["ID", "LAP"], key=key)

    assert uses_fast_kan_diagnostics(fast_kan_nca)
    assert select_wandb_train_logger_class(fast_kan_nca) is kaNCA_Train_log
    assert select_wandb_train_logger_class(nca) is NCA_Train_log
    assert (
        select_wandb_train_logger_class(fast_kan_nca, knockout_time=0)
        is NCA_knockout_Train_log
    )


def test_learning_rate_uses_training_wandb_category():
    logger = object.__new__(NCA_Train_log)
    logged = []
    logger.log_scalar = lambda tag, value, step=None: logged.append(
        (tag, value, step)
    )

    logger.tb_training_loop_log_sequence(
        {"learning_rate": 5e-4},
        i=1,
        model=None,
        write_images=False,
        LOG_EVERY=10,
    )

    assert logged == [("Train/learning_rate", 5e-4, 1)]


def test_fast_kan_diagnostic_plot_helpers_return_images():
    key = jax.random.PRNGKey(1)
    model = FastKaNCA(
        4,
        KERNEL_STR=["ID", "LAP"],
        KAN_AUX={"num_basis": 4},
        key=key,
    )
    logger = _logger_without_wandb()

    edge_norm_image = logger._plot_edge_norms(model.get_edge_norms()[0], layer_index=0)
    edge_function_image = logger._plot_top_edge_functions(model, layer_index=0, k=3)

    assert edge_norm_image.ndim == 4
    assert edge_function_image.ndim == 4
    assert edge_norm_image.shape[0] == 1
    assert edge_function_image.shape[0] == 1
    assert edge_norm_image.shape[-1] in {3, 4}
    assert edge_function_image.shape[-1] in {3, 4}


def test_fast_kan_log_model_parameters_uses_kan_tags():
    key = jax.random.PRNGKey(2)
    model = FastKaNCA(
        4,
        KERNEL_STR=["ID", "LAP"],
        KAN_AUX={"num_basis": 4},
        key=key,
    )
    logger = _logger_without_wandb()
    logged = {"histograms": [], "scalars": [], "images": []}

    logger.log_histogram = lambda tag, values, step=None: logged["histograms"].append(
        tag
    )
    logger.log_scalar = lambda tag, value, step=None: logged["scalars"].append(tag)
    logger.log_image = lambda tag, image, step=None: logged["images"].append(tag)

    logger.log_model_parameters(model, i=10)

    assert any(tag.startswith("Train/KAN/weight_") for tag in logged["histograms"])
    assert any("edge_norm_max" in tag for tag in logged["scalars"])
    assert any(tag.endswith("_edge_norms") for tag in logged["images"])
    assert any(tag.endswith("_top_edge_functions_by_norm") for tag in logged["images"])
    assert not any("weight_image" in tag for tag in logged["images"])


def test_fast_kan_rollout_stats_rank_edges_from_visited_inputs():
    key = jax.random.PRNGKey(3)
    model = FastKaNCA(
        4,
        KERNEL_STR=["ID", "LAP"],
        KAN_AUX={"num_basis": 4, "hidden_features": 5},
        key=key,
    )
    logger = _logger_without_wandb()
    states = [jax.random.normal(key, shape=(2, 4, 5, 6))]
    log_dict = {"states": states}

    stats = logger._collect_fast_kan_rollout_stats(
        model,
        log_dict,
        max_samples=32,
        max_states=2,
        k=3,
    )

    assert len(stats) == 2
    assert stats[0]["edge_var"].shape == (model.N_FEATURES, 5)
    assert stats[1]["edge_var"].shape == (5, 4)
    assert stats[0]["spline_input_samples"].shape == stats[0]["input_samples"].shape
    assert stats[0]["spline_input_std"].shape == stats[0]["input_std"].shape
    assert len(stats[0]["top_edges"]) == 3
    scores = [edge["score"] for edge in stats[0]["top_edges"]]
    assert scores == sorted(scores, reverse=True)
    assert {"input_name", "output_name", "relative_score"} <= set(
        stats[0]["top_edges"][0]
    )
    assert jnp.all(jnp.isfinite(jnp.asarray(stats[0]["edge_var"])))


def test_fast_kan_training_loop_logs_rollout_top_edges():
    key = jax.random.PRNGKey(4)
    model = FastKaNCA(
        4,
        KERNEL_STR=["ID", "LAP"],
        KAN_AUX={"num_basis": 4, "hidden_features": 5},
        key=key,
    )
    logger = _logger_without_wandb()
    logged = {"histograms": [], "scalars": [], "images": []}
    states = [jax.random.normal(key, shape=(2, 4, 5, 6))]
    log_dict = {
        "loss": 1.0,
        "states": states,
    }

    logger.log_histogram = lambda tag, values, step=None: logged["histograms"].append(
        tag
    )
    logger.log_scalar = lambda tag, value, step=None: logged["scalars"].append(tag)
    logger.log_image = lambda tag, image, step=None: logged["images"].append(tag)

    logger.tb_training_loop_log_sequence(
        log_dict,
        i=10,
        model=model,
        write_images=False,
        LOG_EVERY=10,
    )

    assert any(tag.endswith("_rollout_edge_variance") for tag in logged["images"])
    assert any(tag.endswith("_rollout_sorted_feature_std") for tag in logged["images"])
    assert any(
        tag.endswith("_rollout_pre_post_layernorm_histograms")
        for tag in logged["images"]
    )
    assert any(tag.endswith("_rollout_top_edge_functions") for tag in logged["images"])
    assert any("rollout_edge_var_max" in tag for tag in logged["scalars"])
    assert any("frac_abs_raw_input_lt_0p01" in tag for tag in logged["scalars"])
    assert any("frac_abs_spline_input_lt_0p01" in tag for tag in logged["scalars"])
    assert any("frac_abs_output_lt_0p01" in tag for tag in logged["scalars"])
    assert not any(tag.endswith("_top_edge_functions_by_norm") for tag in logged["images"])


def test_legacy_kan_fallback_logs_existing_weight_names():
    class OldKanLike:
        def get_weights(self):
            return [np.ones((2, 2)), np.zeros((2, 1))]

    logger = _logger_without_wandb()
    histograms = []
    logger.log_histogram = lambda tag, values, step=None: histograms.append(tag)

    logger.log_model_parameters(OldKanLike(), i=10)

    assert histograms == ["Input layer weights", "Output layer weights"]


def test_channel_time_diagnostics_compute_masked_totals_and_radial_means():
    predictions = np.ones((1, 2, 3, 5, 5), dtype=np.float32)
    targets = 2.0 * predictions
    boundary = np.zeros((1, 1, 5, 5), dtype=np.float32)
    boundary[:, :, 1:4, 1:4] = 1.0

    diagnostics = compute_channel_time_diagnostics(
        predictions,
        targets,
        boundary_masks=boundary,
        radial_bins=3,
    )

    assert diagnostics["prediction_total_intensity"].shape == (2, 3)
    assert np.allclose(diagnostics["prediction_total_intensity"], 9.0)
    assert np.allclose(diagnostics["target_total_intensity"], 18.0)
    nonempty_prediction_values = diagnostics["prediction_radial_profile"][
        diagnostics["prediction_radial_profile"] > 0
    ]
    nonempty_target_values = diagnostics["target_radial_profile"][
        diagnostics["target_radial_profile"] > 0
    ]
    assert np.allclose(nonempty_prediction_values, 1.0)
    assert np.allclose(nonempty_target_values, 2.0)
    assert diagnostics["radius"][-1] > 1.0


def test_radial_profile_line_plot_contains_one_line_per_timestep_and_channel():
    diagnostics = {
        "target_radial_profile": np.ones((3, 2, 4), dtype=np.float32),
        "prediction_radial_profile": np.zeros((3, 2, 4), dtype=np.float32),
        "radius": np.linspace(0.0, 1.5, 4),
    }

    image = plot_radial_intensity_line_diagnostics(
        diagnostics, ["a", "b"], ["t1", "t2", "t3"]
    )

    assert image.ndim == 4
    assert image.shape[0] == 1
    assert image.shape[-1] in {3, 4}


def test_grouped_diagnostic_outputs_are_aligned_to_12_target_channels():
    outputs = np.arange(9, dtype=np.float32).reshape(1, 1, 9, 1, 1)

    aligned = _target_aligned_diagnostic_channels(outputs, grouped_channels=True)

    assert aligned.shape == (1, 1, 12, 1, 1)
    assert aligned[0, 0, :, 0, 0].tolist() == [
        0,
        1,
        2,
        3,
        0,
        1,
        2,
        4,
        5,
        6,
        7,
        8,
    ]


def test_schema_diagnostic_outputs_are_aligned_to_measurement_channels():
    schema = MICROPATTERN_260726_SCHEMA
    outputs = np.arange(10, dtype=np.float32).reshape(1, 1, 10, 1, 1)

    aligned = _target_aligned_diagnostic_channels(outputs, channel_schema=schema)

    assert aligned.shape == (1, 1, 14, 1, 1)
    assert aligned[0, 0, :, 0, 0].tolist() == list(schema.target_to_state)

    snapshots = _trajectory_snapshot_channels(
        outputs[0], type("Augmenter", (), {"OBS_CHANNELS": 10})(), 1, schema
    )
    assert snapshots[:, :, 0, 0].tolist() == [list(schema.target_to_state)]


def test_channel_correlation_diagnostics_capture_correlation_and_anticorrelation():
    increasing = np.arange(4, dtype=np.float32).reshape(2, 2)
    decreasing = increasing[::-1, ::-1]
    prediction = np.stack([increasing, increasing, decreasing], axis=0)[None, None]
    target = np.stack([increasing, decreasing, decreasing], axis=0)[None, None]

    diagnostics = compute_channel_correlation_diagnostics(prediction, target)

    prediction_correlation = diagnostics["prediction_channel_correlation"][0]
    target_correlation = diagnostics["target_channel_correlation"][0]
    assert np.isclose(prediction_correlation[0, 1], 1.0)
    assert np.isclose(prediction_correlation[0, 2], -1.0)
    assert np.isclose(target_correlation[0, 1], -1.0)
    assert np.isclose(
        diagnostics["channel_correlation_difference"][0, 0, 1],
        2.0,
    )


def test_channel_correlation_diagnostics_mask_separate_experiment_groups():
    values = np.arange(12, dtype=np.float32).reshape(1, 1, 3, 2, 2)

    diagnostics = compute_channel_correlation_diagnostics(
        values, values, experiment_group_sizes=(2, 1)
    )

    correlations = diagnostics["prediction_channel_correlation"][0]
    assert np.isfinite(correlations[0, 1])
    assert np.isnan(correlations[0, 2])


def test_channel_correlation_plot_returns_timestep_grid():
    diagnostics = {
        "prediction_channel_correlation": np.eye(3, dtype=np.float32)[None],
        "channel_correlation_difference": np.zeros((1, 3, 3), dtype=np.float32),
    }

    image = plot_channel_correlation_diagnostics(
        diagnostics,
        ["a", "b", "c"],
        ["t12h"],
        experiment_group_sizes=(2, 1),
    )

    assert image.ndim == 4
    assert image.shape[0] == 1
    assert image.shape[-1] in {3, 4}


def test_radial_profile_plot_contains_all_channel_timestep_rows():
    diagnostics = {
        "target_radial_profile": np.ones((2, 3, 4), dtype=np.float32),
        "prediction_radial_profile": np.zeros((2, 3, 4), dtype=np.float32),
    }

    image = plot_radial_intensity_diagnostics(diagnostics, ["a", "b", "c"])

    assert image.ndim == 4
    assert image.shape[0] == 1
    assert image.shape[-1] in {3, 4}


def test_total_intensity_plot_coalesces_channel_timestep_values():
    diagnostics = {
        "target_total_intensity": np.ones((2, 3), dtype=np.float32),
        "prediction_total_intensity": np.zeros((2, 3), dtype=np.float32),
    }

    image = plot_total_intensity_diagnostics(
        diagnostics,
        ["a", "b", "c"],
        ["t12h", "t24h"],
    )

    assert image.ndim == 4
    assert image.shape[0] == 1
    assert image.shape[-1] in {3, 4}


def test_channel_time_grid_and_true_logging_retain_batch_labels():
    values = np.ones((2, 3, 4, 4), dtype=np.float32)
    image = plot_channel_time_grid(values, ["a", "b", "c"], ["t0", "t1"])
    assert image.ndim == 4

    logger = object.__new__(NCA_Train_log)
    logger.channel_names = ["a", "b", "c"]
    logger.timepoint_names = ["t1"]
    logged = {}
    logger.log_image = lambda tag, images, step=None: logged.update({tag: images})
    logger.log_data_at_init(np.stack((values, values)))
    assert logged["True sequence labelled"].shape[0] == 2
    assert _biomarker_name("cell_fate_s1/SOX2") == "SOX2"


def test_training_logger_emits_channel_time_diagnostics_without_wandb():
    logger = object.__new__(NCA_Train_log)
    logger.diagnostic_targets = np.ones((1, 2, 12, 5, 5), dtype=np.float32)
    logger.diagnostic_boundary_mask = np.ones((1, 1, 5, 5), dtype=np.float32)
    logger.diagnostic_grouped_channels = True
    logger.radial_bins = 4
    logger.radial_extent = 1.5
    logger.channel_names = [f"channel_{index + 1}" for index in range(12)]
    logger.timepoint_names = ["t12h", "t24h"]
    logged = {"scalars": {}, "images": []}
    logger.log_scalar = lambda tag, value, step=None: logged["scalars"].update(
        {tag: value}
    )
    logger.log_image = lambda tag, image, step=None: logged["images"].append(tag)
    predictions = [np.ones((2, 64, 5, 5), dtype=np.float32)]

    logger.log_channel_time_diagnostics({"states": predictions}, i=10)

    assert logged["scalars"] == {
        "Diagnostics/total_intensity/mean_absolute_error": 0.0
    }
    assert logged["images"] == [
        "Diagnostics/total_intensity",
        "Diagnostics/radial_intensity_profiles",
        "Diagnostics/radial_intensity_lines",
        "Diagnostics/channel_correlation",
    ]


def test_training_snapshots_include_every_batch_in_one_composite():
    logger = object.__new__(NCA_Train_log)
    logger.normalise_images = lambda values: values
    logged = {}
    logger.log_image = lambda tag, image, step=None: logged.update({tag: image})
    logger.log_scalar = lambda *args, **kwargs: None
    values = [np.ones((2, 4, 3, 5), dtype=np.float32) * batch for batch in range(3)]

    with patch("NCA.trainer.logging.tensorboard.get_jax_memory_stats", return_value={}):
        logger.log_model_outputs({"states": values}, 1)

    assert logged["Train/visible_batches"].shape == (9, 10, 3)
    assert np.all(logged["Train/visible_batches"][6:] == 2)


def test_group_timestep_losses_log_one_histogram_at_diagnostic_interval():
    logger = object.__new__(NCA_Train_log)
    logger.timepoint_names = ["t12h", "t24h"]
    logger.log_model_parameters = lambda *args: None
    logger.log_channel_time_diagnostics = lambda *args: None
    logged = []
    logger.log_scalar = lambda tag, value, step=None: logged.append((tag, value, step))
    logger.log_histogram = lambda tag, values, step=None: logged.append(
        (tag, np.asarray(values), step)
    )
    details = {
        "loss_detail/rna_expression/total": np.array([1.0, 2.0]),
        "loss_detail/cell_fate_s1/total": np.array([3.0, 4.0]),
        "loss_detail/rna_expression/radial": np.array([10.0, 20.0]),
    }

    logger.tb_training_loop_log_sequence(details, 9, None, False, 10)
    assert logged == []
    logger.tb_training_loop_log_sequence(details, 10, None, False, 10)
    assert len(logged) == 1
    assert logged[0][0] == "Train/loss_detail/group_timestep"
    np.testing.assert_array_equal(logged[0][1], [1.0, 2.0, 3.0, 4.0])
    assert logged[0][2] == 10
