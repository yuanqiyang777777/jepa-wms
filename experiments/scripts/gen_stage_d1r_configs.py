#!/usr/bin/env python
"""Generate MGVT-D STAGE-D1R oracle-sandwich smoke configs.

The first D1R scaffold is intentionally narrow: PushT, seed 234, H=4,
prediction-only.  C0 is a frozen metric-parity/reuse baseline and is not
trained here.  C1/C2 are freshly trained D1R controls.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE = REPO_ROOT / "configs" / "vjepa_wm" / "mgvt_stage2c" / "pusht_stage2c_pred_adaln_depth1_lance_h4.yaml"
OUT_DIR = REPO_ROOT / "configs" / "vjepa_wm" / "mgvt_d1r"
TASK = "pusht"
SEED = 234


@dataclass(frozen=True)
class D1RConfig:
    variant: str
    stage: str
    pred_type: str
    predictor_updates: dict[str, Any]
    trainable: bool = True
    order: int = 0


def _yaml() -> YAML:
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.default_flow_style = False
    yaml.width = 4096
    return yaml


def _load(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return _yaml().load(handle)


def _dump(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        _yaml().dump(data, handle)


def _common_d1r() -> dict[str, Any]:
    return {
        "pred_embed_dim": 128,
        "pred_depth": 1,
        "d_h_dim": 64,
        "r_h_dim": 64,
        "state_dim": 64,
        "p_dyn_dim": 64,
        "fdyn_type": "mlp",
        "context_window": 2,
        "proprio_flow": "no_proprio",
        "dh_ablation": "none",
        "lambda_delta": 1.0,
        "lambda_inv_r": 0.1,
        "lambda_sig": 0.01,
        "lambda_trend": 1.0,
        "lambda_inv_d": 0.1,
        "use_inv_d": False,
    }


def d1r_configs() -> tuple[D1RConfig, ...]:
    common = _common_d1r()
    return (
        D1RConfig(
            "c1_adaln_param_match",
            "train",
            "AdaLN",
            {"pred_embed_dim": 128, "pred_depth": 2, "init_scale_factor_adaln": 0},
        ),
        D1RConfig(
            "c2_raw_action_guidance",
            "train",
            "mgvt_d_raw_action",
            {
                "pred_embed_dim": 128,
                "pred_depth": 1,
                "d_h_dim": 64,
                "state_dim": 64,
                "fdyn_type": "raw_action",
                "context_window": 2,
                "proprio_flow": "no_proprio",
            },
        ),
        D1RConfig("t0_oracle_rh", "oracle", "mgvt_d1r", {**common, "d1r_stage": "oracle"}),
        D1RConfig("s0_implicit_conditioning", "implicit", "mgvt_d1r", {**common, "d1r_stage": "implicit"}),
        D1RConfig("s1_trend_match", "r1_teacher", "mgvt_d1r", {**common, "d1r_stage": "r1_teacher"}, order=1),
        D1RConfig("s1_trend_match", "r2_student", "mgvt_d1r", {**common, "d1r_stage": "r2_student"}, order=2),
        D1RConfig("s1_trend_match", "g_refine", "mgvt_d1r", {**common, "d1r_stage": "g_refine"}, order=3),
        D1RConfig("s2_trend_inv", "r1_teacher", "mgvt_d1r", {**common, "d1r_stage": "r1_teacher"}, order=1),
        D1RConfig(
            "s2_trend_inv",
            "r2_student",
            "mgvt_d1r",
            {**common, "d1r_stage": "r2_student", "use_inv_d": True},
            order=2,
        ),
        D1RConfig("s2_trend_inv", "g_refine", "mgvt_d1r", {**common, "d1r_stage": "g_refine", "use_inv_d": True}, order=3),
    )


def _stem(spec: D1RConfig) -> str:
    return f"{TASK}_d1r_{spec.variant}_{spec.stage}_lance_h4"


def _output_path(spec: D1RConfig) -> Path:
    return OUT_DIR / f"{_stem(spec)}.yaml"


def _clear_forbidden_protocol(cfg: dict[str, Any]) -> None:
    cfg["evals"] = None
    cfg["unroll_decode_evals"] = None
    cfg.setdefault("meta", {})["plan_only_eval_mode"] = False
    cfg["meta"]["light_eval_only_mode"] = False
    cfg["meta"]["eval_freq"] = 999999
    cfg["meta"]["light_eval_freq"] = 999999


def _uses_aux_replace_loss(spec: D1RConfig) -> bool:
    return spec.pred_type == "mgvt_d1r" and spec.stage in {"r1_teacher", "r2_student", "oracle"}


def build_config(spec: D1RConfig):
    cfg = deepcopy(_load(SOURCE))
    stem = _stem(spec)
    cfg["folder"] = f"${{JEPAWM_LOGS}}/mgvt_d1r_oracle_sandwich_pusht_seed234_20260604/{stem}"
    cfg["checkpoint_folder"] = f"${{JEPAWM_CKPT}}/mgvt_d1r_oracle_sandwich_pusht_seed234_20260604/{stem}"
    cfg.setdefault("logging", {}).setdefault("wandb", {})["use_wandb"] = False
    _clear_forbidden_protocol(cfg)

    cfg.setdefault("meta", {})["seed"] = SEED
    cfg.setdefault("data", {})["seed"] = SEED
    cfg.setdefault("data", {}).setdefault("custom", {})["num_pred"] = 4
    cfg.setdefault("model", {}).setdefault("rollout_cfg", {})["rollout_steps"] = 4
    cfg["model"]["rollout_cfg"]["ctxt_window_train_rollout"] = 2
    if _uses_aux_replace_loss(spec):
        cfg["model"]["rollout_cfg"]["do_sequential_rollout"] = False
        cfg["model"]["rollout_cfg"]["do_parallel_rollout"] = False
    cfg["loss"]["proprio_loss"] = False

    cfg["model"]["proprio_encoder"]["proprio_emb_dim"] = 0
    cfg["model"]["proprio_encoder"]["proprio_tokens"] = 0
    cfg["model"]["predictor"]["pred_type"] = spec.pred_type
    for key, value in spec.predictor_updates.items():
        cfg["model"]["predictor"][key] = value

    cfg["model"]["predictor"]["d1r_variant"] = spec.variant
    cfg["model"]["predictor"]["d1r_stage_name"] = spec.stage
    cfg["model"]["predictor"]["d1r_stage_order"] = int(spec.order)
    cfg["model"]["predictor"]["d1r_trainable"] = bool(spec.trainable)
    if cfg["data"]["custom"].get("num_hist") != 2:
        raise AssertionError(f"{stem}: D1R first smoke must keep D1-A context W=2")
    if cfg["data"]["custom"].get("num_pred") != 4:
        raise AssertionError(f"{stem}: D1R first smoke must keep H=4")
    if cfg["model"]["rollout_cfg"].get("rollout_steps") != 4:
        raise AssertionError(f"{stem}: rollout_steps drifted from H=4")
    if cfg.get("evals") or cfg.get("unroll_decode_evals"):
        raise AssertionError(f"{stem}: online/decode evals must be disabled")
    return cfg


def generate(dry_run: bool = False, variant_filter: str | None = None) -> list[Path]:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    written = []
    for spec in d1r_configs():
        if variant_filter and variant_filter not in spec.variant:
            continue
        output = _output_path(spec)
        cfg = build_config(spec)
        if not dry_run:
            _dump(cfg, output)
        written.append(output)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--variant-filter", default=None)
    args = parser.parse_args()
    paths = generate(dry_run=args.dry_run, variant_filter=args.variant_filter)
    action = "would write" if args.dry_run else "wrote"
    print(f"{action} {len(paths)} Stage-D1R configs")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
