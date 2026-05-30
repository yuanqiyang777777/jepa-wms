#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# Licensed under the MIT License.
"""Generate MGVT-JEPA Stage-2b breadth-scan configs.

Stage-2b intentionally reuses the validated Stage-2a model/optimizer knobs and
only stamps task-specific dataset fields. The generated configs are checked in so
lab01 runs do not depend on dynamic config generation.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from ruamel.yaml import YAML


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "configs" / "vjepa_wm" / "mgvt_stage2b"
BASE_DIR = REPO_ROOT / "configs" / "vjepa_wm" / "mgvt_stage2"

IMAGENET_NORM = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]

BACKBONES = {
    "adaln_depth1": BASE_DIR / "pusht_stage2_pred_adaln_depth1_lance_h4.yaml",
    "mgvt_convmixer": BASE_DIR / "pusht_stage2_pred_mgvt_convmixer_lance_h4.yaml",
    "mgvt_mamba": BASE_DIR / "pusht_stage2_pred_mgvt_mamba_lance_h4.yaml",
}

TASKS = {
    "pusht": {
        "datasets": ["PushT"],
        "lance_uri": "${JEPAWM_DSET_LANCE}/PushT.lance",
        "filter_tasks": None,
        "val_dataset_camera_views": None,
        "seed": 234,
    },
    "wall": {
        "datasets": ["Wall"],
        "lance_uri": "${JEPAWM_DSET_LANCE}/Wall.lance",
        "filter_tasks": None,
        "val_dataset_camera_views": None,
        "seed": 234,
    },
    "maze": {
        "datasets": ["PointMaze"],
        "lance_uri": "${JEPAWM_DSET_LANCE}/PointMaze.lance",
        "filter_tasks": None,
        "val_dataset_camera_views": None,
        "seed": 234,
    },
    "mw_r": {
        "datasets": ["METAWORLD_HF"],
        "lance_uri": "${JEPAWM_DSET_LANCE}/Metaworld.lance",
        "filter_tasks": ["mw-reach"],
        "val_dataset_camera_views": ["exterior_image_2_left"],
        "seed": 1,
    },
    "mw_rw": {
        "datasets": ["METAWORLD_HF"],
        "lance_uri": "${JEPAWM_DSET_LANCE}/Metaworld.lance",
        "filter_tasks": ["mw-reach-wall"],
        "val_dataset_camera_views": ["exterior_image_2_left"],
        "seed": 1,
    },
}


def _yaml() -> YAML:
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.default_flow_style = False
    return yaml


def _load(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return _yaml().load(handle)


def _dump(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        _yaml().dump(data, handle)


def _build_config(task_token: str, backbone_token: str):
    cfg = deepcopy(_load(BACKBONES[backbone_token]))
    task = TASKS[task_token]
    stem = f"{task_token}_stage2_pred_{backbone_token}_lance_h4"

    cfg["folder"] = f"${{JEPAWM_LOGS}}/mgvt_stage2b/{stem}"
    cfg["checkpoint_folder"] = f"${{JEPAWM_CKPT}}/mgvt_stage2b/{stem}"
    cfg["data"]["datasets"] = list(task["datasets"])
    cfg["data"]["seed"] = int(task["seed"])
    cfg["data"]["backend"]["lance_uri"] = task["lance_uri"]
    cfg["data"]["backend"]["fall_back_to_raw_if_unsupported"] = False
    cfg["data"]["validation"]["val_dataset_camera_views"] = task["val_dataset_camera_views"]
    cfg["data"]["custom"]["filter_tasks"] = task["filter_tasks"]
    cfg["data_aug"]["normalize"] = deepcopy(IMAGENET_NORM)
    cfg["meta"]["seed"] = int(task["seed"])
    cfg["evals"] = None
    return stem, cfg


def main() -> None:
    for task_token in TASKS:
        for backbone_token in BACKBONES:
            stem, cfg = _build_config(task_token, backbone_token)
            _dump(cfg, OUT_DIR / f"{stem}.yaml")
            print(f"wrote {OUT_DIR / f'{stem}.yaml'}")


if __name__ == "__main__":
    main()
