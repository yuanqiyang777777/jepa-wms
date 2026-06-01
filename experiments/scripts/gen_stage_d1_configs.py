#!/usr/bin/env python
"""Generate MGVT-D Stage-D1 prediction-only configs.

All configs are derived from the frozen Stage-2c AdaLN H=4 templates and keep
the same datasets, seeds, Lance backend, W=2/H=4 rollout settings, and disabled
planning evals. Variant changes are limited to the predictor block, output
paths, and explicit D1 metadata.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = REPO_ROOT / "configs" / "vjepa_wm" / "mgvt_stage2c"
OUT_DIR = REPO_ROOT / "configs" / "vjepa_wm" / "mgvt_d1"

TASKS = ("pusht", "wall", "maze", "mw_r", "mw_rw")


@dataclass(frozen=True)
class Variant:
    block: str
    name: str
    pred_type: str
    predictor_updates: dict[str, Any]
    proprio_emb_dim: int | None = None


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


def _source_path(task: str) -> Path:
    return SOURCE_DIR / f"{task}_stage2c_pred_adaln_depth1_lance_h4.yaml"


def d1_variants() -> tuple[Variant, ...]:
    common = {
        "pred_embed_dim": 128,
        "pred_depth": 1,
        "d_h_dim": 16,
        "state_dim": 64,
        "context_window": 2,
        "dh_ablation": "none",
        "proprio_flow": "dyn_refine",
        "delta_p_dim": None,
    }
    return (
        # D1-A
        Variant("d1a", "raw_action_guidance", "mgvt_d_raw_action", {**common, "fdyn_type": "raw_action"}),
        Variant("d1a", "mlp_trend_dim16", "mgvt_d_mlp", {**common, "fdyn_type": "mlp"}),
        Variant("d1a", "gru_trend_dim16", "mgvt_d_gru", {**common, "fdyn_type": "gru"}),
        Variant(
            "d1a",
            "mamba_trend_dim16",
            "mgvt_d_mamba",
            {**common, "fdyn_type": "mamba", "require_cuda_mamba": True},
        ),
        Variant("d1a", "adaln_param_match", "AdaLN", {"pred_embed_dim": 128, "pred_depth": 2}),
        # D1-B, defaulting to the Mamba trend candidate until D1-A selects a winner.
        Variant("d1b", "mamba_trend_dim4", "mgvt_d_mamba", {**common, "fdyn_type": "mamba", "d_h_dim": 4, "require_cuda_mamba": True}),
        Variant("d1b", "mamba_trend_dim8", "mgvt_d_mamba", {**common, "fdyn_type": "mamba", "d_h_dim": 8, "require_cuda_mamba": True}),
        Variant("d1b", "mamba_trend_dim32", "mgvt_d_mamba", {**common, "fdyn_type": "mamba", "d_h_dim": 32, "require_cuda_mamba": True}),
        Variant("d1b", "mamba_trend_remove_dh", "mgvt_d_mamba", {**common, "fdyn_type": "mamba", "dh_ablation": "remove", "require_cuda_mamba": True}),
        Variant("d1b", "mamba_trend_shuffle_dh", "mgvt_d_mamba", {**common, "fdyn_type": "mamba", "dh_ablation": "shuffle", "require_cuda_mamba": True}),
        Variant("d1b", "mamba_trend_random_dh", "mgvt_d_mamba", {**common, "fdyn_type": "mamba", "dh_ablation": "random", "require_cuda_mamba": True}),
        # D1-C
        Variant("d1c", "no_proprio", "mgvt_d_mamba", {**common, "fdyn_type": "mamba", "proprio_flow": "no_proprio", "require_cuda_mamba": True}, proprio_emb_dim=0),
        Variant("d1c", "dyn_only_prop", "mgvt_d_mamba", {**common, "fdyn_type": "mamba", "proprio_flow": "dyn_only", "require_cuda_mamba": True}),
        Variant("d1c", "refine_only_prop", "mgvt_d_mamba", {**common, "fdyn_type": "mamba", "proprio_flow": "refine_only", "require_cuda_mamba": True}),
        Variant("d1c", "dyn_refine_prop", "mgvt_d_mamba", {**common, "fdyn_type": "mamba", "proprio_flow": "dyn_refine", "require_cuda_mamba": True}),
        Variant("d1c", "old_concat_prop", "AdaLN", {"pred_embed_dim": 128, "pred_depth": 2}),
        # D1-D
        Variant("d1d", "refine_depth_baseline", "mgvt_d_mamba", {**common, "fdyn_type": "mamba", "refiner_depth": 1, "require_cuda_mamba": True}),
        Variant("d1d", "refine_depth_half", "mgvt_d_mamba", {**common, "fdyn_type": "mamba", "refiner_depth": 1, "require_cuda_mamba": True}),
        Variant("d1d", "refine_depth_minimal", "mgvt_d_mamba", {**common, "fdyn_type": "mamba", "refiner_depth": 0, "require_cuda_mamba": True}),
        Variant("d1d", "no_fdyn_param_match", "mgvt_d_raw_action", {**common, "fdyn_type": "raw_action", "refiner_depth": 1}),
        # D1-E
        Variant("d1e", "sparse_update_control", "mgvt_d_sparse_control", {**common, "fdyn_type": "mlp", "guidance_mode": "sparse_control"}),
    )


def _stem(task: str, variant: Variant) -> str:
    return f"{task}_{variant.block}_{variant.name}_lance_h4"


def _output_path(task: str, variant: Variant) -> Path:
    return OUT_DIR / f"{_stem(task, variant)}.yaml"


def _clear_planning(cfg: dict[str, Any]) -> None:
    cfg["evals"] = None
    cfg["unroll_decode_evals"] = None
    cfg.setdefault("meta", {})["plan_only_eval_mode"] = False
    cfg["meta"]["light_eval_only_mode"] = False
    cfg["meta"]["eval_freq"] = 999999
    cfg["meta"]["light_eval_freq"] = 999999


def build_config(task: str, variant: Variant):
    cfg = deepcopy(_load(_source_path(task)))
    stem = _stem(task, variant)
    cfg["folder"] = f"${{JEPAWM_LOGS}}/mgvt_d/d1_20260602/{stem}"
    cfg["checkpoint_folder"] = f"${{JEPAWM_CKPT}}/mgvt_d/d1_20260602/{stem}"
    cfg.setdefault("logging", {}).setdefault("wandb", {})["use_wandb"] = False
    _clear_planning(cfg)

    cfg.setdefault("data", {}).setdefault("custom", {})["num_pred"] = 4
    cfg.setdefault("model", {}).setdefault("rollout_cfg", {})["rollout_steps"] = 4
    cfg["model"]["rollout_cfg"]["ctxt_window_train_rollout"] = 2
    cfg["model"]["predictor"]["pred_type"] = variant.pred_type
    for key, value in variant.predictor_updates.items():
        cfg["model"]["predictor"][key] = value
    if variant.proprio_emb_dim is not None:
        cfg["model"]["proprio_encoder"]["proprio_emb_dim"] = variant.proprio_emb_dim
        cfg["model"]["proprio_encoder"]["proprio_tokens"] = 0

    cfg["model"]["predictor"]["d1_block"] = variant.block
    cfg["model"]["predictor"]["d1_variant"] = variant.name
    cfg["model"]["predictor"]["confidence_head"] = False
    cfg["model"]["predictor"].setdefault("L_trend_weight", 0.0)

    if cfg["data"]["custom"].get("num_pred") != 4:
        raise AssertionError(f"{stem}: D1 must keep H=4")
    if cfg["model"]["rollout_cfg"].get("rollout_steps") != 4:
        raise AssertionError(f"{stem}: rollout_steps drifted from H=4")
    if cfg.get("evals"):
        raise AssertionError(f"{stem}: planning evals must be disabled")
    return cfg


def generate(dry_run: bool = False, block_filter: str | None = None) -> list[Path]:
    written = []
    variants = [v for v in d1_variants() if block_filter in (None, v.block)]
    for task in TASKS:
        if not _source_path(task).exists():
            raise FileNotFoundError(_source_path(task))
        for variant in variants:
            output = _output_path(task, variant)
            cfg = build_config(task, variant)
            if not dry_run:
                _dump(cfg, output)
            written.append(output)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--block", choices=["d1a", "d1b", "d1c", "d1d", "d1e"], default=None)
    args = parser.parse_args()

    paths = generate(dry_run=args.dry_run, block_filter=args.block)
    action = "would write" if args.dry_run else "wrote"
    print(f"{action} {len(paths)} Stage-D1 configs")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()

