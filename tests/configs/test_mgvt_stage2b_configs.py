# Copyright (c) Facebook, Inc. and its affiliates.
# Licensed under the MIT License.

from pathlib import Path

from src.utils.yaml_utils import load_yaml


IMAGENET_NORM = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]

TASK_SPECS = {
    "pusht": {
        "dataset": ["PushT"],
        "lance_suffix": "/PushT.lance",
        "filter_tasks": None,
        "seed": 234,
        "val_camera": None,
    },
    "wall": {
        "dataset": ["Wall"],
        "lance_suffix": "/Wall.lance",
        "filter_tasks": None,
        "seed": 234,
        "val_camera": None,
    },
    "maze": {
        "dataset": ["PointMaze"],
        "lance_suffix": "/PointMaze.lance",
        "filter_tasks": None,
        "seed": 234,
        "val_camera": None,
    },
    "mw_r": {
        "dataset": ["METAWORLD_HF"],
        "lance_suffix": "/Metaworld.lance",
        "filter_tasks": ["mw-reach"],
        "seed": 1,
        "val_camera": ["exterior_image_2_left"],
    },
    "mw_rw": {
        "dataset": ["METAWORLD_HF"],
        "lance_suffix": "/Metaworld.lance",
        "filter_tasks": ["mw-reach-wall"],
        "seed": 1,
        "val_camera": ["exterior_image_2_left"],
    },
}

BACKBONE_SPECS = {
    "adaln_depth1": "AdaLN",
    "mgvt_convmixer": "mgvt_convmixer",
    "mgvt_mamba": "mgvt_mamba",
}


def _task_token(path: Path) -> str:
    return path.name.split("_stage2_")[0]


def _backbone_token(path: Path) -> str:
    stem = path.stem
    prefix = f"{_task_token(path)}_stage2_pred_"
    suffix = "_lance_h4"
    assert stem.startswith(prefix)
    assert stem.endswith(suffix)
    return stem[len(prefix) : -len(suffix)]


def test_mgvt_stage2b_configs_are_complete_and_contract_safe(monkeypatch):
    monkeypatch.setenv("JEPAWM_DSET_LANCE", "/tmp/jepawm_lance")
    config_dir = Path("configs/vjepa_wm/mgvt_stage2b")
    config_paths = sorted(config_dir.glob("*_stage2_pred_*_lance_h4.yaml"))

    assert len(config_paths) == len(TASK_SPECS) * len(BACKBONE_SPECS)
    assert {_task_token(path) for path in config_paths} == set(TASK_SPECS)
    assert {_backbone_token(path) for path in config_paths} == set(BACKBONE_SPECS)

    for path in config_paths:
        task = _task_token(path)
        backbone = _backbone_token(path)
        task_spec = TASK_SPECS[task]
        cfg = load_yaml(path)

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
        assert cfg["model"]["predictor"]["pred_type"] == BACKBONE_SPECS[backbone]
        assert cfg["model"]["predictor"]["pred_depth"] == 1
        assert cfg["model"]["predictor"]["init_scale_factor_adaln"] == 0
        assert cfg["model"]["rollout_cfg"]["rollout_steps"] == 4
        assert cfg["model"]["rollout_cfg"]["train_rollout_prefixes"] == "all"
        assert cfg["model"]["rollout_cfg"]["ctxt_window_train_rollout"] == 2
        assert cfg["model"]["rollout_cfg"]["rollout_stop_gradient"] is True
        assert cfg["model"]["rollout_cfg"]["do_sequential_rollout"] is True
        assert cfg["model"]["rollout_cfg"]["do_parallel_rollout"] is False
        assert cfg["evals"] is None
        assert cfg["optimization"]["transition_model"]["num_epochs"] == 10
