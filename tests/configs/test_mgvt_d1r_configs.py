from experiments.scripts.gen_stage_d1r_configs import OUT_DIR, build_config, d1r_configs, generate
from src.utils.yaml_utils import load_yaml


def test_stage_d1r_generator_dry_run_matrix_is_expected_size():
    paths = generate(dry_run=True)
    assert len(paths) == len(d1r_configs())
    assert all(path.parent == OUT_DIR for path in paths)


def test_stage_d1r_configs_keep_oracle_sandwich_contract():
    for spec in d1r_configs():
        cfg = build_config(spec)
        predictor = cfg["model"]["predictor"]
        assert cfg["model"]["rollout_cfg"]["rollout_steps"] == 4
        assert cfg["model"]["rollout_cfg"]["ctxt_window_train_rollout"] == 2
        assert cfg["data"]["custom"]["num_hist"] == 2
        assert cfg["data"]["custom"]["num_pred"] == 4
        assert cfg.get("evals") is None
        assert cfg.get("unroll_decode_evals") is None
        assert cfg["model"]["proprio_encoder"]["proprio_emb_dim"] == 0
        assert cfg["model"]["proprio_encoder"]["proprio_tokens"] == 0
        assert cfg["loss"]["proprio_loss"] is False
        assert predictor["d1r_variant"] == spec.variant
        assert predictor["d1r_stage_name"] == spec.stage


def test_stage_d1r_c1_adaln_param_match_uses_frozen_width():
    cfg = build_config(next(s for s in d1r_configs() if s.variant == "c1_adaln_param_match"))
    predictor = cfg["model"]["predictor"]
    assert predictor["pred_type"] == "AdaLN"
    assert predictor["pred_embed_dim"] == 112
    assert predictor["pred_depth"] == 2


def test_stage_d1r_configs_do_not_include_forbidden_protocol_terms():
    paths = generate(dry_run=False)
    forbidden = ("cem", "planning", "h8", "stage-3", "stage3", "confidence", "num_pred: 8", "rollout_steps: 8")
    for path in paths:
        text = path.read_text(encoding="utf-8").lower()
        for token in forbidden:
            assert token not in text, (path, token)


def test_stage_d1r_generated_yaml_loads_and_pins_d1r_defaults():
    generate(dry_run=False, variant_filter="s1_trend_match")
    cfg = load_yaml(str(OUT_DIR / "pusht_d1r_s1_trend_match_r2_student_lance_h4.yaml"))
    predictor = cfg["model"]["predictor"]
    assert predictor["pred_type"] == "mgvt_d1r"
    assert predictor["d1r_stage"] == "r2_student"
    assert predictor["d_h_dim"] == 64
    assert predictor["r_h_dim"] == 64
    assert predictor["lambda_delta"] == 1.0
    assert predictor["lambda_trend"] == 1.0
    assert predictor["use_inv_d"] is False


def test_stage_d1r_s2_r2_enables_student_inverse_consistency():
    cfg = build_config(next(s for s in d1r_configs() if s.variant == "s2_trend_inv" and s.stage == "r2_student"))
    assert cfg["model"]["predictor"]["use_inv_d"] is True


def test_stage_d1r_aux_replace_stages_disable_training_rollout():
    for spec in d1r_configs():
        cfg = build_config(spec)
        rollout_cfg = cfg["model"]["rollout_cfg"]
        if spec.pred_type == "mgvt_d1r" and spec.stage in {"r1_teacher", "r2_student", "oracle"}:
            assert rollout_cfg["do_sequential_rollout"] is False
            assert rollout_cfg["do_parallel_rollout"] is False
        else:
            assert rollout_cfg["do_sequential_rollout"] is True


def test_stage_d1r_launcher_resets_epoch_for_staged_weight_handoff():
    launcher = (OUT_DIR.parents[2] / "experiments" / "scripts" / "mgvt_d1r_scan.sh").read_text(encoding="utf-8")
    assert 'cfg["meta"]["load_opt_scale_epoch"] = False' in launcher
    assert 'cfg["meta"]["reset_epoch_on_pretrained_load"] = True' in launcher


def test_stage_d1r_launcher_exposes_diagnostic_worker_overrides_without_changing_defaults():
    launcher = (OUT_DIR.parents[2] / "experiments" / "scripts" / "mgvt_d1r_scan.sh").read_text(encoding="utf-8")
    assert 'TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-}"' in launcher
    assert 'TRAIN_PERSISTENT_WORKERS="${TRAIN_PERSISTENT_WORKERS:-}"' in launcher
    assert 'loader["num_workers"] = int(train_num_workers)' in launcher
    assert 'loader["persistent_workers"] = value in {"1", "true", "yes"}' in launcher


def test_stage_d1r_launcher_rescores_s1_s2_dh_controls_without_retraining():
    launcher = (OUT_DIR.parents[2] / "experiments" / "scripts" / "mgvt_d1r_scan.sh").read_text(encoding="utf-8")
    assert "score_dh_controls" in launcher
    assert 'cfg.setdefault("model", {}).setdefault("predictor", {})["dh_ablation"] = mode' in launcher
    assert "--checkpoint \"$ckpt_dir/jepa-latest.pth.tar\"" in launcher
    assert "--batch-size \"$SKILL_BATCH_SIZE\"" in launcher
    assert "delta_semantics" in launcher
