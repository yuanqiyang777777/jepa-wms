import json

import numpy as np

from app.vjepa_wm.diagnostics.mgvt_d1_probes import summarize_probe_bundle, main


def test_d1r_action_residualized_delta_detects_extra_trend_signal():
    rng = np.random.default_rng(0)
    action = rng.normal(size=(64, 3))
    residual = rng.normal(size=(64, 2))
    delta_y = action[:, :2] * 0.5 + residual
    r_h = residual + 0.01 * rng.normal(size=(64, 2))
    d_h = residual + 0.05 * rng.normal(size=(64, 2))

    summary = summarize_probe_bundle({"action": action, "delta_y": delta_y, "r_h": r_h, "d_h": d_h})

    assert summary["action_residualized_delta_r2_r_h"] > 0.02
    assert summary["action_residualized_delta_r2_d_h"] > 0.01


def test_d1r_probe_cli_writes_required_placeholder_files(tmp_path, monkeypatch):
    skill_dir = tmp_path / "skill"
    skill_dir.mkdir()
    skill_json = skill_dir / "skill_score.json"
    skill_json.write_text(
        json.dumps(
            {
                "model": "mgvt_d1r",
                "task": "PushT",
                "seed": 234,
                "moved_region_change_skill_by_horizon": {"4": 0.25},
                "skill_by_horizon": {"4": 0.5},
                "change_skill_by_horizon": {"4": 0.3},
                "param_count": 123,
                "inference_path_param_count": 100,
                "flops_per_forward_estimate": 456,
                "oracle_rh_eval": True,
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "probes"
    monkeypatch.setattr(
        "sys.argv",
        ["mgvt_d1_probes", "--skill-json", str(skill_json), "--output", str(output_dir)],
    )

    main()

    required = {
        "mgvt_d1_probes.json",
        "trend_alignment.json",
        "inverse_readout.json",
        "action_residualized_delta.json",
        "collapse_rank.json",
        "leakage_probe.json",
        "cross_position_consistency.json",
        "dh_controls.json",
        "params_flops.json",
        "oracle_rh_score.json",
    }
    assert required.issubset({p.name for p in output_dir.iterdir()})
    params = json.loads((output_dir / "params_flops.json").read_text(encoding="utf-8"))
    assert params["param_count"] == 123
    assert params["inference_path_param_count"] == 100
    assert params["flops_per_forward_estimate"] == 456
