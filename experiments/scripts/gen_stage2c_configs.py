#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# Licensed under the MIT License.
"""Generate MGVT-JEPA Stage-2c AdaLN confirmation configs.

Stage-2c is a pure three-seed replication of the Stage-2b AdaLN arm. The only
checked-in config changes are the output/checkpoint folder tags.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from ruamel.yaml import YAML


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = REPO_ROOT / "configs" / "vjepa_wm" / "mgvt_stage2b"
OUT_DIR = REPO_ROOT / "configs" / "vjepa_wm" / "mgvt_stage2c"

TASKS = ("pusht", "wall", "maze", "mw_r", "mw_rw")


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


def _retag_path(value: str) -> str:
    value = value.replace("/mgvt_stage2b/", "/mgvt_stage2c/")
    return value.replace("_stage2_", "_stage2c_")


def _source_path(task: str) -> Path:
    return SOURCE_DIR / f"{task}_stage2_pred_adaln_depth1_lance_h4.yaml"


def _output_path(task: str) -> Path:
    return OUT_DIR / f"{task}_stage2c_pred_adaln_depth1_lance_h4.yaml"


def _without_output_paths(cfg):
    cfg = deepcopy(cfg)
    cfg.pop("folder", None)
    cfg.pop("checkpoint_folder", None)
    return cfg


def _build_config(task: str):
    source = _load(_source_path(task))
    cfg = deepcopy(source)
    cfg["folder"] = _retag_path(str(source["folder"]))
    cfg["checkpoint_folder"] = _retag_path(str(source["checkpoint_folder"]))

    if _without_output_paths(source) != _without_output_paths(cfg):
        raise AssertionError(f"{task}: generated config drifted beyond output paths")
    if cfg["folder"] == source["folder"] or cfg["checkpoint_folder"] == source["checkpoint_folder"]:
        raise AssertionError(f"{task}: output folders were not retagged")
    if "mgvt_stage2c" not in cfg["folder"] or "_stage2c_" not in cfg["folder"]:
        raise AssertionError(f"{task}: invalid folder tag {cfg['folder']!r}")
    if "mgvt_stage2c" not in cfg["checkpoint_folder"] or "_stage2c_" not in cfg["checkpoint_folder"]:
        raise AssertionError(f"{task}: invalid checkpoint tag {cfg['checkpoint_folder']!r}")
    return cfg


def main() -> None:
    written = []
    for task in TASKS:
        source = _source_path(task)
        if not source.exists():
            raise FileNotFoundError(source)
        output = _output_path(task)
        _dump(_build_config(task), output)
        written.append(output)
        print(f"wrote {output}")
        print(f"verified equivalence except folder/checkpoint_folder: {task}")

    if len(written) != len(TASKS):
        raise AssertionError(f"expected {len(TASKS)} configs, wrote {len(written)}")


if __name__ == "__main__":
    main()
