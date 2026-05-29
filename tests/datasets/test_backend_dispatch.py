# Copyright (c) Facebook, Inc. and its affiliates.
# Licensed under the MIT License.

import pytest


def _import_init_data_with_optional_decord_stub():
    import importlib.util
    import sys
    import types

    decord_mod = sys.modules.get("decord")
    if decord_mod is None and importlib.util.find_spec("decord") is None:
        decord_stub = types.ModuleType("decord")
        decord_stub.bridge = types.SimpleNamespace(set_bridge=lambda *_args, **_kwargs: None)
        decord_stub.VideoReader = object
        decord_stub.cpu = lambda *_args, **_kwargs: None
        sys.modules["decord"] = decord_stub

    datasets_mod = sys.modules.get("datasets")
    datasets_spec = None if datasets_mod is not None else importlib.util.find_spec("datasets")
    datasets_origin = str(getattr(datasets_spec, "origin", "") or "")
    if datasets_mod is None and (
        datasets_spec is None or "tests\\datasets" in datasets_origin or "tests/datasets" in datasets_origin
    ):
        datasets_stub = types.ModuleType("datasets")
        datasets_stub.load_dataset = lambda *_args, **_kwargs: None
        sys.modules["datasets"] = datasets_stub

    from app.plan_common.datasets.utils import init_data
    return init_data


def test_invalid_backend_kind_raises_before_dataset_dispatch():
    init_data = _import_init_data_with_optional_decord_stub()
    with pytest.raises(ValueError, match="data.backend.kind"):
        init_data(
            data_paths=["point_maze"],
            batch_size=1,
            backend={"kind": "swl_lance"},
        )


def test_swm_lance_backend_requires_lance_uri_for_pusht():
    init_data = _import_init_data_with_optional_decord_stub()
    with pytest.raises(ValueError, match="data.backend.kind=swm_lance requires data.backend.lance_uri"):
        init_data(
            data_paths=["pusht"],
            batch_size=1,
            backend={"kind": "swm_lance", "fall_back_to_raw_if_unsupported": False},
        )


def test_swm_lance_backend_requires_lance_uri_for_metaworld():
    init_data = _import_init_data_with_optional_decord_stub()
    with pytest.raises(ValueError, match="data.backend.kind=swm_lance requires data.backend.lance_uri"):
        init_data(
            data_paths=["metaworld"],
            batch_size=1,
            backend={"kind": "swm_lance", "fall_back_to_raw_if_unsupported": False},
            filter_tasks=["reach-wall"],
        )


def test_mgvt_stage1_configs_use_lance_backend(monkeypatch):
    from pathlib import Path

    from src.utils.yaml_utils import load_yaml

    monkeypatch.setenv("JEPAWM_DSET_LANCE", "/tmp/jepawm_lance")
    config_dir = Path("configs/vjepa_wm/mgvt_stage1")
    config_paths = sorted(config_dir.glob("*_lance_1roll.yaml"))

    assert len(config_paths) == 8
    for path in config_paths:
        cfg = load_yaml(path)
        backend = cfg["data"]["backend"]
        assert backend["kind"] == "swm_lance"
        assert backend["fall_back_to_raw_if_unsupported"] is False
        assert backend["lance_uri"].startswith("/tmp/jepawm_lance/")
        if path.name.startswith("pusht_"):
            assert backend["lance_uri"].endswith("/PushT.lance")
            assert cfg["data"]["custom"]["filter_tasks"] is None
        else:
            assert backend["lance_uri"].endswith("/Metaworld.lance")
            assert cfg["data"]["custom"]["filter_tasks"] == ["reach-wall"]
        assert cfg["model"]["wm_encoding"]["normalize_reps"] is True
        assert cfg["model"]["predictor"]["pred_depth"] == 1
        assert cfg["model"]["rollout_cfg"]["rollout_steps"] == 1
