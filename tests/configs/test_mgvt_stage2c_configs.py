# Copyright (c) Facebook, Inc. and its affiliates.
# Licensed under the MIT License.

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from src.utils.yaml_utils import load_yaml


TASK_SPECS = {
    "pusht": {"seed": 234, "dataset": ["PushT"], "lance_suffix": "/PushT.lance", "filter_tasks": None, "val_camera": None},
    "wall": {"seed": 234, "dataset": ["Wall"], "lance_suffix": "/Wall.lance", "filter_tasks": None, "val_camera": None},
    "maze": {"seed": 234, "dataset": ["PointMaze"], "lance_suffix": "/PointMaze.lance", "filter_tasks": None, "val_camera": None},
    "mw_r": {
        "seed": 1,
        "dataset": ["METAWORLD_HF"],
        "lance_suffix": "/Metaworld.lance",
        "filter_tasks": ["mw-reach"],
        "val_camera": ["exterior_image_2_left"],
    },
    "mw_rw": {
        "seed": 1,
        "dataset": ["METAWORLD_HF"],
        "lance_suffix": "/Metaworld.lance",
        "filter_tasks": ["mw-reach-wall"],
        "val_camera": ["exterior_image_2_left"],
    },
}

IMAGENET_NORM = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]


def _task_token(path: Path) -> str:
    return path.name.split("_stage2c_")[0]


def _stage2b_source(path: Path) -> Path:
    task = _task_token(path)
    return Path("configs/vjepa_wm/mgvt_stage2b") / f"{task}_stage2_pred_adaln_depth1_lance_h4.yaml"


def _without_output_paths(cfg):
    cfg = deepcopy(cfg)
    cfg.pop("folder", None)
    cfg.pop("checkpoint_folder", None)
    return cfg


def test_mgvt_stage2c_configs_are_adaln_only_seed_replicates(monkeypatch):
    monkeypatch.setenv("JEPAWM_DSET_LANCE", "/tmp/jepawm_lance")
    monkeypatch.setenv("JEPAWM_LOGS", "/tmp/jepawm_logs")
    monkeypatch.setenv("JEPAWM_CKPT", "/tmp/jepawm_ckpt")

    config_dir = Path("configs/vjepa_wm/mgvt_stage2c")
    config_paths = sorted(config_dir.glob("*_stage2c_pred_adaln_depth1_lance_h4.yaml"))

    assert len(config_paths) == len(TASK_SPECS)
    assert {_task_token(path) for path in config_paths} == set(TASK_SPECS)

    for path in config_paths:
        task = _task_token(path)
        task_spec = TASK_SPECS[task]
        cfg = load_yaml(path)
        source = load_yaml(_stage2b_source(path))

        assert _without_output_paths(cfg) == _without_output_paths(source)
        assert cfg["folder"] == source["folder"].replace("/mgvt_stage2b/", "/mgvt_stage2c/").replace(
            "_stage2_", "_stage2c_"
        )
        assert cfg["checkpoint_folder"] == source["checkpoint_folder"].replace(
            "/mgvt_stage2b/", "/mgvt_stage2c/"
        ).replace("_stage2_", "_stage2c_")

        assert cfg["data"]["datasets"] == task_spec["dataset"]
        assert cfg["data"]["seed"] == task_spec["seed"]
        assert cfg["meta"]["seed"] == task_spec["seed"]
        assert cfg["data"]["custom"]["filter_tasks"] == task_spec["filter_tasks"]
        assert cfg["data"]["validation"]["val_dataset_camera_views"] == task_spec["val_camera"]
        assert cfg["data_aug"]["normalize"] == IMAGENET_NORM

        backend = cfg["data"]["backend"]
        assert backend["kind"] == "swm_lance"
        assert backend["fall_back_to_raw_if_unsupported"] is False
        assert backend["lance_uri"].startswith("/tmp/jepawm_lance/")
        assert backend["lance_uri"].endswith(task_spec["lance_suffix"])

        assert cfg["data"]["custom"]["num_hist"] == 2
        assert cfg["data"]["custom"]["num_pred"] == 4
        assert cfg["data"]["validation"]["num_frames_val"] >= 6
        assert cfg["model"]["num_frames_pred"] == 6
        assert cfg["model"]["wm_encoding"]["normalize_reps"] is True
        assert cfg["model"]["proprio_encoder"]["proprio_emb_dim"] == 16
        assert cfg["model"]["proprio_encoder"]["proprio_encoder_inpred"] is False
        assert cfg["model"]["predictor"]["pred_type"] == "AdaLN"
        assert cfg["model"]["predictor"]["pred_depth"] == 1
        assert cfg["model"]["predictor"]["init_scale_factor_adaln"] == 0
        assert cfg["model"]["rollout_cfg"]["rollout_steps"] == 4
        assert cfg["model"]["rollout_cfg"]["train_rollout_prefixes"] == "all"
        assert cfg["model"]["rollout_cfg"]["ctxt_window_train_rollout"] == 2
        assert cfg["model"]["rollout_cfg"]["rollout_stop_gradient"] is True
        assert cfg["model"]["rollout_cfg"]["do_sequential_rollout"] is True
        assert cfg["model"]["rollout_cfg"]["do_parallel_rollout"] is False
        assert cfg["meta"]["freeze_encoder"] is True
        assert cfg["meta"]["eval_freq"] == 999999
        assert cfg["meta"]["light_eval_freq"] == 999999
        assert cfg["meta"]["plan_only_eval_mode"] is False
        assert cfg["evals"] is None
        assert cfg["optimization"]["transition_model"]["num_epochs"] == 10


def test_stage2c_seed_lists_match_confirmation_plan():
    assert {"pusht", "wall", "maze"} == {task for task, spec in TASK_SPECS.items() if spec["seed"] == 234}
    assert {"mw_r", "mw_rw"} == {task for task, spec in TASK_SPECS.items() if spec["seed"] == 1}


def test_stage2c_scan_launcher_is_adaln_only_and_three_seed():
    script = Path("experiments/scripts/mgvt_stage2c_scan.sh").read_text(encoding="utf-8")

    assert 'PUSHT_SEEDS="${PUSHT_SEEDS:-234 235 236}"' in script
    assert 'WALL_SEEDS="${WALL_SEEDS:-234 235 236}"' in script
    assert 'MAZE_SEEDS="${MAZE_SEEDS:-234 235 236}"' in script
    assert 'MW_R_SEEDS="${MW_R_SEEDS:-1 2 3}"' in script
    assert 'MW_RW_SEEDS="${MW_RW_SEEDS:-1 2 3}"' in script
    assert 'task="${stem%%_stage2c_*}"' in script

    assert "mgvt_stage2c_horizon_raw.csv" in script
    assert "mgvt_stage2c_horizon_summary.csv" in script
    assert "skill_mean_minus_std" in script
    assert "change_skill_mean_minus_std" in script
    assert "proprio_skill_mean_minus_std" in script

    assert script.count("configs/vjepa_wm/mgvt_stage2c/") == len(TASK_SPECS)
    assert "mgvt_convmixer" not in script
    assert "mgvt_mamba" not in script
