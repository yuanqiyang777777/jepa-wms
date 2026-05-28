# Copyright (c) Facebook, Inc. and its affiliates.
# Licensed under the MIT License.

import pytest


def _import_init_data_with_optional_decord_stub():
    import importlib.util
    import sys
    import types

    if importlib.util.find_spec("decord") is None:
        decord_stub = types.ModuleType("decord")
        decord_stub.bridge = types.SimpleNamespace(set_bridge=lambda *_args, **_kwargs: None)
        decord_stub.VideoReader = object
        decord_stub.cpu = lambda *_args, **_kwargs: None
        sys.modules["decord"] = decord_stub

    datasets_spec = importlib.util.find_spec("datasets")
    datasets_origin = str(getattr(datasets_spec, "origin", "") or "")
    if datasets_spec is None or "tests\\datasets" in datasets_origin or "tests/datasets" in datasets_origin:
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
