"""Scores and ablation bookkeeping of Experiments/micropatterns/finetune_ablation.py."""

import numpy as np
import pytest

from Experiments.micropatterns.config import KnockoutConfig
from Experiments.micropatterns.finetune_ablation import (
    baseline_share,
    load_results,
    save_results,
    score_prediction,
)


@pytest.mark.parametrize(
    "weights,share",
    [
        (None, 1 / 3),
        ({"baseline": 0.0}, 0.0),
        ({"baseline": 4.0}, 2 / 3),
        ({"ko_0h": 0.0, "ko_24h": 0.0}, 1.0),
    ],
)
def test_baseline_share_of_the_loss_weight(weights, share):
    knockout = KnockoutConfig(
        curriculum=("baseline", "ko_0h", "ko_24h"), curriculum_weights=weights
    )
    assert baseline_share(knockout) == pytest.approx(share)


def test_baseline_only_runs_have_share_one():
    assert baseline_share(KnockoutConfig()) == 1.0


def _condition_data(target, measured):
    boundary = np.zeros((1, 8, 8), dtype=bool)
    boundary[0, 2:6, 2:6] = True
    return {
        "target": target,
        "measured": measured,
        "boundary": boundary,
        "channel_names": ("a", "b"),
    }


def test_scores_use_only_measured_channels_inside_the_colony():
    target = np.zeros((3, 2, 8, 8), dtype=np.float32)
    prediction = target.copy()
    prediction[:, 1] = 1.0  # wrong everywhere in channel b
    prediction[:, 0, 0, 0] = 5.0  # wrong outside the colony only
    measured = np.array([[True, True], [True, False], [True, True]])

    rows, radial_prediction, radial_target = score_prediction(
        prediction, _condition_data(target, measured), n_rings=2
    )

    assert [row["time_index"] for row in rows] == [1, 2]
    assert rows[0]["pixel_mse"] == 0.0 and rows[0]["radial_mse"] == 0.0
    assert rows[1]["pixel_mse"] == pytest.approx(0.5)
    assert rows[1]["radial_mse"] == pytest.approx(0.5)
    assert radial_prediction.shape == radial_target.shape == (3, 2, 2)


def test_time_points_without_measurements_are_skipped():
    target = np.zeros((2, 2, 8, 8), dtype=np.float32)
    measured = np.array([[True, True], [False, False]])
    rows, _, _ = score_prediction(target, _condition_data(target, measured), n_rings=2)
    assert rows == []


def test_results_round_trip(tmp_path):
    results = [
        {"model_id": "a", "pixel_mse": 1.0, "radial_prediction": np.ones((2, 3, 4)), "radial_target": None},
    ]
    back = load_results(save_results(tmp_path / "results.npz", results))
    assert back[0]["model_id"] == "a"
    assert back[0]["radial_prediction"].shape == (2, 3, 4)
    assert back[0]["radial_target"] is None
