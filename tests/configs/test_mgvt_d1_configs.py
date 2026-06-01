from pathlib import Path

from experiments.scripts.gen_stage_d1_configs import OUT_DIR, TASKS, build_config, d1_variants, generate
from src.utils.yaml_utils import load_yaml


def test_stage_d1_generator_dry_run_matrix_is_expected_size():
    paths = generate(dry_run=True)
    assert len(paths) == len(TASKS) * len(d1_variants())
    assert all(path.parent == OUT_DIR for path in paths)


def test_stage_d1_configs_keep_prediction_only_h4_contract():
    for task in TASKS:
        for variant in d1_variants():
            cfg = build_config(task, variant)
            assert cfg["model"]["rollout_cfg"]["rollout_steps"] == 4
            assert cfg["model"]["rollout_cfg"]["ctxt_window_train_rollout"] == 2
            assert cfg["data"]["custom"]["num_hist"] == 2
            assert cfg["data"]["custom"]["num_pred"] == 4
            assert cfg.get("evals") is None
            assert cfg.get("unroll_decode_evals") is None
            assert cfg["model"]["predictor"]["pred_type"] == variant.pred_type
            assert cfg["model"]["predictor"]["confidence_head"] is False
            assert cfg["model"]["predictor"]["L_trend_weight"] == 0.0
            assert "mgvt_d/d1_20260602" in cfg["folder"]
            assert "mgvt_d/d1_20260602" in cfg["checkpoint_folder"]


def test_stage_d1_configs_do_not_include_forbidden_protocol_terms(tmp_path):
    paths = generate(dry_run=False)
    forbidden = ("cem", "planning", "stage-3", "stage3", "num_pred: 8", "rollout_steps: 8")
    for path in paths:
        text = path.read_text(encoding="utf-8").lower()
        for token in forbidden:
            assert token not in text, (path, token)


def test_generated_yaml_loads_and_contains_mamba_cuda_contract():
    generate(dry_run=False, block_filter="d1a")
    mamba_cfgs = sorted(Path(OUT_DIR).glob("*mamba_trend_dim16*.yaml"))
    assert mamba_cfgs
    for path in mamba_cfgs:
        cfg = load_yaml(path)
        predictor = cfg["model"]["predictor"]
        assert predictor["pred_type"] == "mgvt_d_mamba"
        assert predictor["require_cuda_mamba"] is True
        assert predictor["d_h_dim"] == 16

