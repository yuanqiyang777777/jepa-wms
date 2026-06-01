import json

import numpy as np
import pytest
import torch

from app.vjepa_wm.diagnostics.mgvt_d1_probes import (
    cross_position_margin,
    nearest_centroid_accuracy,
    ridge_r2,
    summarize_probe_bundle,
)
from app.vjepa_wm.diagnostics.skill_score import (
    _add_masked_patch_mse,
    _finalize_horizon_metrics,
    _make_moved_patch_mask,
)
from experiments.scripts.summarize_mgvt_d1 import _group_summary, _read_scores


def test_moved_patch_mask_uses_target_side_motion_only():
    persist = torch.tensor([[[0.0, 1.0, 4.0, 2.0]]])
    mask = _make_moved_patch_mask(persist, top_frac=0.5, min_floor=0.0, grid_size=2, dilation=0)
    assert mask.tolist() == [[[False, False, True, True]]]


def test_masked_moved_region_skill_math():
    model = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    persist = torch.tensor([[[2.0, 2.0, 6.0, 8.0]]])
    mask = torch.tensor([[[False, False, True, True]]])
    model_sums = {1: 0.0}
    persist_sums = {1: 0.0}
    counts = {1: 0}
    _add_masked_patch_mse(model, persist, mask, 1, model_sums, persist_sums, counts)
    metrics = _finalize_horizon_metrics(model_sums, persist_sums, counts, "moved_region")
    assert metrics["moved_region_skill"]["1"] == pytest.approx(1.0 - (7.0 / 14.0))


def test_probe_helpers_detect_relative_signal_and_cross_position_margin():
    rng = np.random.default_rng(7)
    d_h = rng.normal(size=(20, 4))
    relative_motion = d_h[:, :2] * 0.5
    future_z = rng.normal(size=(20, 6))
    labels = np.repeat([0, 1], 10)
    positive_pairs = np.array([[0, 1], [2, 3], [10, 11], [12, 13]])
    negative_pairs = np.array([[0, 10], [2, 12], [4, 14], [6, 16]])

    summary = summarize_probe_bundle(
        {
            "d_h": d_h,
            "relative_motion": relative_motion,
            "future_z": future_z,
            "absolute_region": labels,
            "positive_pairs": positive_pairs,
            "negative_pairs": negative_pairs,
        }
    )

    assert ridge_r2(d_h, relative_motion) > 0.9
    assert nearest_centroid_accuracy(d_h, labels) >= 0.5
    assert "future_z_r2" in summary
    assert "cross_position_margin" in summary


def test_cross_position_margin_positive_for_constructed_pairs():
    d_h = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [-1.0, 0.0],
            [-0.9, -0.1],
        ]
    )
    result = cross_position_margin(d_h, np.array([[0, 1], [2, 3]]), np.array([[0, 2], [1, 3]]))
    assert result["cross_position_margin"] > 1.0


def test_d1_summary_reads_backward_compatible_skill_json(tmp_path):
    run = tmp_path / "skill_scores" / "run1"
    run.mkdir(parents=True)
    (run / "skill_score.json").write_text(
        json.dumps(
            {
                "task": "PushT",
                "model": "mgvt_d_mlp",
                "seed": 234,
                "skill_by_horizon": {"4": 0.5},
                "change_skill_by_horizon": {"4": 0.6},
                "moved_region_change_skill_by_horizon": {"4": 0.7},
                "proprio_skill_by_horizon": {"4": 0.9},
                "param_count": 10,
                "flops_per_forward_estimate": 20,
            }
        ),
        encoding="utf-8",
    )

    rows = _read_scores(tmp_path / "skill_scores")
    summary = _group_summary(rows)

    assert rows[0]["moved_region_change_skill"] == 0.7
    assert summary[0]["moved_region_change_skill_mean"] == pytest.approx(0.7)

