#!/usr/bin/env python
"""Generate a CSA-MPC diagnostic eval config from one of the released JEPA-WM
"Ours" checkpoints. The config flips on ``csa_diagnostics.enabled`` so the
PlanEvaluator hook writes per-episode pooled-latent dumps; all support scoring +
diagnostics 0-4 run offline against those dumps. No support memory is loaded
during eval -- ``memory_path`` is intentionally NOT exposed here.
"""
from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]

ENV_DEFAULTS = {
    "pusht": {
        "base_config": "configs/evals/simu_env_planning/pt/jepa-wm/pt_L2_cem_sourcedset_H6_nas6_ctxt2_r224_alpha0.1_ep96_decode.yaml",
        "checkpoint": "jepa_wm_pusht.pth.tar",
        "folder": "${JEPAWM_LOGS}/csa_pilot/pusht_official_jepa_wm",
    },
    "wall": {
        "base_config": "configs/evals/simu_env_planning/wall/jepa-wm/wall_L2_cem_sourcerandstate_H6_nas6_ctxt2_r224_alpha0.1_ep96_decode.yaml",
        "checkpoint": "jepa_wm_wall.pth.tar",
        "folder": "${JEPAWM_LOGS}/csa_pilot/wall_official_jepa_wm",
    },
    "maze": {
        "base_config": "configs/evals/simu_env_planning/mz/jepa-wm/mz_L2_cem_sourcerandstate_H6_nas6_ctxt2_r224_alpha0.1_ep96_decode.yaml",
        "checkpoint": "jepa_wm_pointmaze.pth.tar",
        "folder": "${JEPAWM_LOGS}/csa_pilot/maze_official_jepa_wm",
    },
    # Alias kept for backwards compatibility with the earlier q0 pilot.
    "mz": {
        "base_config": "configs/evals/simu_env_planning/mz/jepa-wm/mz_L2_cem_sourcerandstate_H6_nas6_ctxt2_r224_alpha0.1_ep96_decode.yaml",
        "checkpoint": "jepa_wm_pointmaze.pth.tar",
        "folder": "${JEPAWM_LOGS}/csa_pilot/maze_official_jepa_wm",
    },
    "mw-reach": {
        "base_config": "configs/evals/simu_env_planning/mw/jepa-wm/reach_L2_cem_sourcexp_H6_nas3_ctxt2_r256_alpha0.1_ep48_decode.yaml",
        "checkpoint": "jepa_wm_metaworld.pth.tar",
        "folder": "${JEPAWM_LOGS}/csa_pilot/mw-reach_official_jepa_wm",
    },
    "mw-reach-wall": {
        "base_config": "configs/evals/simu_env_planning/mw/jepa-wm/reach-wall_L2_cem_sourcexp_H6_nas3_ctxt2_r256_alpha0.1_ep48_decode.yaml",
        "checkpoint": "jepa_wm_metaworld.pth.tar",
        "folder": "${JEPAWM_LOGS}/csa_pilot/mw-reach-wall_official_jepa_wm",
    },
}


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_config(
    *,
    env: str,
    base_config: str | None = None,
    eval_episodes: int = 96,
    folder: str | None = None,
    tag: str = "csa_diag/full",
    quick_debug: bool = False,
    dump_dir: str | None = None,
    extra_overrides: dict | None = None,
) -> dict:
    defaults = ENV_DEFAULTS[env]
    base_path = REPO_ROOT / (base_config or defaults["base_config"])
    cfg = deepcopy(_load_yaml(base_path))

    cfg["checkpoint_folder"] = "${JEPAWM_CKPT}"
    cfg["folder"] = folder or defaults["folder"]
    cfg["tag"] = tag
    cfg["model_kwargs"]["checkpoint"] = defaults["checkpoint"]

    cfg.setdefault("meta", {})
    cfg["meta"]["quick_debug"] = bool(quick_debug)
    cfg["meta"]["eval_episodes"] = int(eval_episodes)

    cfg.setdefault("distributed", {})
    # Keep distributed multi-task eval flag from the base config -- some envs
    # rely on it for the rng-shift logic. Only force false if base says nothing.
    cfg["distributed"].setdefault("distribute_multitask_eval", False)

    cfg.setdefault("logging", {})
    cfg["logging"].setdefault("optional_plots", False)
    cfg["logging"].setdefault("tqdm_silent", False)

    # Only ``enabled`` (and optional ``dump_dir``) are read by the eval hook --
    # the full diagnostic suite + support memory live offline in analysis.py.
    csa_block: dict[str, object] = {"enabled": True}
    if dump_dir is not None:
        csa_block["dump_dir"] = str(dump_dir)
    cfg["csa_diagnostics"] = csa_block

    if extra_overrides:
        _deep_merge(cfg, extra_overrides)

    return cfg


def _deep_merge(dst: dict, src: dict) -> None:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_merge(dst[key], value)
        else:
            dst[key] = value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=sorted(ENV_DEFAULTS), required=True)
    parser.add_argument("--output", required=True, help="YAML config path to write")
    parser.add_argument("--base-config", default=None)
    parser.add_argument("--eval-episodes", type=int, default=96)
    parser.add_argument("--folder", default=None)
    parser.add_argument("--tag", default="csa_diag/full")
    parser.add_argument("--quick-debug", action="store_true", help="Quick-debug mode (1 episode, 2 CEM samples, etc.)")
    parser.add_argument("--dump-dir", default=None, help="Optional override for the per-episode dump root")
    args = parser.parse_args()

    cfg = make_config(
        env=args.env,
        base_config=args.base_config,
        eval_episodes=args.eval_episodes,
        folder=args.folder,
        tag=args.tag,
        quick_debug=args.quick_debug,
        dump_dir=args.dump_dir,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


if __name__ == "__main__":
    sys.exit(main())
