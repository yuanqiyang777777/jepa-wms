#!/usr/bin/env python3
"""Import the sealed stock-LeWM Cube probe caches and run the lambda sweep from them.

This is the only probe implementation for the user-directed 2026-07-15
EXPLORATORY stock-Cube lambda-transfer sweep (mlp family).  No probe is ever
fit in this run.  Its result-bearing operations are:

``import-caches``
    Copy the 30 sealed ``.npz`` cache artifacts of the source run
    stock_cube_probe_20260714 into this run root, verifying every byte against
    the preregistered per-file SHA-256 pins before and after the copy.

``run-cached``
    Wrap the hash-pinned probe runner, poison its original fit routes, and
    replace the selected fit site with a verified cache load.  Planning thus
    performs zero probe fits.

The full run uses the exact 50-evaluation exclusion set stored in each cache.
The two-episode smoke is intentionally evaluated only on episodes drawn from
that full exclusion set; it is a wiring test and is never result-bearing.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np


RUN_ID = "stock_cube_lambda_sweep_mlp_20260715"
SOURCE_RUN_ID = "stock_cube_probe_20260714"
SOURCE_FIT_SUMMARY_SHA256 = "4c15c04e1bf247da55494889fad3e8fef1938b8193c18afb2040151f7aa9c8df"
SOURCE_SMOKE_MANIFEST_SHA256 = "3646f155d6d90c3b1ce4f507a8d5f15adb1ddec668cd68cae2e2288d83860d43"
GATE_STATUS_EXPLORATORY = "NOT_APPLICABLE_EXPLORATORY_SWEEP"
TASK = "cube"
ARM = "stock"
TRAIN_SEEDS = (42, 43, 44)
EVAL_SEEDS = (42, 43, 44, 45, 46)
BUDGETS = (24, 48, 96, 192)
PROBE_SEED = 1703
PROBE_TRAIN_SAMPLES = 3000
PROBE_BATCH_SIZE = 128
MLP_EPOCHS = 200
SUCCESS_RADIUS_M = 0.04
PURE_GATE_THRESHOLD_M = SUCCESS_RADIUS_M / 5.0
HYBRID_GATE_THRESHOLD_M = SUCCESS_RADIUS_M / 3.0

IMPORTED_CACHE_SHA256: dict[str, str] = {
    "probe_caches/s0/ts42_es42.npz": "ed31c3d56403629f3c06ae7039bc8e45ed1ac897948c0ccffa7a19908a91069a",
    "probe_caches/s0/ts42_es43.npz": "5a24b06079b8eb16a7c415ee2fa38c5acb9ebdeb097d1a7285539c2b6590c0ff",
    "probe_caches/s0/ts42_es44.npz": "b16debfddb54443b4107a0f8e2b63080460d7b1468a801761b845d5aa09cfcd1",
    "probe_caches/s0/ts42_es45.npz": "2f65730300c70a1129841cd359918b28914cc063ad5400f5a3e451b6aa73e09c",
    "probe_caches/s0/ts42_es46.npz": "96d5f170946404df74d8cc702a4522a7bae020c39e4fff5d7ec1a331c8a8fe64",
    "probe_caches/s0/ts43_es42.npz": "89bd356ffdbdb0aa5680dd256540008acf5733cb526483e20334084c321459bd",
    "probe_caches/s0/ts43_es43.npz": "5884c40077de36517b37e91fab488905f523096deac616c4907b69f3d70d8d76",
    "probe_caches/s0/ts43_es44.npz": "71a9bd93e76f6cce5c6336ccb351672129fe024afa600c467c75b672d30f0258",
    "probe_caches/s0/ts43_es45.npz": "f2c968f7287ba035b4392f8d7235d628f44c834b55060a0ca86ddb2f6e553088",
    "probe_caches/s0/ts43_es46.npz": "f0882e1cde283f98d28ffa0a692c4f3be8911e3956a97abb27e4e54ab2a80c5a",
    "probe_caches/s0/ts44_es42.npz": "88bd4440e3479adea3ddba1039a123b6002b937fc23ce672b2c5bcdfa3f38227",
    "probe_caches/s0/ts44_es43.npz": "3805df4cd75cb6818b999a472c4f38eec6dea0769c19f82602314c0efbecf354",
    "probe_caches/s0/ts44_es44.npz": "659daadd902b2760ac3481bfef3ad041e896cc9977850811402ae058a352b3a7",
    "probe_caches/s0/ts44_es45.npz": "d5ff2bbdcdc1201154c11fde2c076f8e181859e2f559fc0191dd7572c20cbfeb",
    "probe_caches/s0/ts44_es46.npz": "4cb77f08690e37f96fa2bdf7c9e704d371a3868cd365f915d2989275bd8365a0",
    "probe_caches/s1/ts42_es42.npz": "ceeffda5c81bceeea1d597e6fd6f7c676c96c7875d1864451cc367699a480ad1",
    "probe_caches/s1/ts42_es43.npz": "4d541043cb3883afef2cef0e560663ce5cd0202161c5ba6cabef3b47b99dc383",
    "probe_caches/s1/ts42_es44.npz": "21f5ac8d0ef7b58093972dc2c2df2558bb8d8678caa0e58b5863c214f800b7a9",
    "probe_caches/s1/ts42_es45.npz": "df410003ea533ef027401e43aa17f994b69ba0535616cb5ba6ae5a525ee8c924",
    "probe_caches/s1/ts42_es46.npz": "9f1133add54684905879b32796aefac329d0a29edea23ebf0dc756d7271a3ac0",
    "probe_caches/s1/ts43_es42.npz": "2b0e41d05a4f28082e5d7ae388cf13715961dcda7876a739a60890cbbeba8cbc",
    "probe_caches/s1/ts43_es43.npz": "cce3550aa3535d13aac920d08d2c0978be9bf08f595dbec44c4dcd469969b8ca",
    "probe_caches/s1/ts43_es44.npz": "fe77dc4f0585a2cf4732775eae207398ae29c384cc93354e90f31c369d10b786",
    "probe_caches/s1/ts43_es45.npz": "3262c35da68cd19fc69775196e87fc2e1921d0a8b2d20c40c86a07deeb09f948",
    "probe_caches/s1/ts43_es46.npz": "eeee77050b3bd36021f75eb86bda2c72829200d344ea74c5eddf6562a4c207a4",
    "probe_caches/s1/ts44_es42.npz": "17dfa0a3f4e24985df7070024825e40f04c2c415cb88d60d36feb86f0bb9d251",
    "probe_caches/s1/ts44_es43.npz": "1a4cba2c5b794abf7f0d31716b300987eb6063eaadeabb580ba8abffa720fe70",
    "probe_caches/s1/ts44_es44.npz": "9d42bd411cbdef0e1f47276ad2868c4afac22bde19eab675219bb3af00f0b7ec",
    "probe_caches/s1/ts44_es45.npz": "85ddb261722f10c3d8d58ab6f05a3662b1b03cd9f973374c06b20373c6e5ddc3",
    "probe_caches/s1/ts44_es46.npz": "2f0ac557b03b8744a489b24ab7b083c380a00b0862cfe57fb0f54aa5bcd878da",
}

PROBE_RUNNER_SHA256 = "564d7469840a38bf6015149d9d61191694beef9e0f5ed6610c5db9934b7dbdc8"
PROBE_COMMON_SHA256 = "9c57a82343277e3378f24df8ca38e5f9a4dcc53a4fead909e2a6126001d1a89c"
BUDGET_RUNNER_SHA256 = "cefd2b267601377aa1201a5302828807f4189ca11f123c1128746938f51dfd8f"
TASK_MANIFEST_SHA256 = "1e769c4c9b7876374f5c059af351476b2a71a73be37156cabc26b328ac0a5561"
BASELINE_MANIFEST_SHA256 = "433cc4df1f963979941d7f78d0ce41b0043f64ac6a9ab353a47ef17861247da9"
BASELINE_EVAL_ARRAYS_SHA256 = "a900d1d04684eb97a76ff4acffd9d5731e1e8faa3062e22e1b7f747950b1a81d"
BASELINE_PAIR_EVAL_ARRAYS_SHA256 = {
    (42, 42): "021b323032f3581cfa8c6d5ac3706232e27d7950c6fe4fbd076b7ba36340f765",
    (42, 43): "0f4ddbeae23679cd15575111e0d282f018f3b7b3dffb66a483cdb2c0d365efcf",
    (42, 44): "f0d7da940c1111f5409f3ade9b6b0ecc209198226acbabc485d70465e05c6df6",
    (42, 45): "442366201a3861976f6e058ce95e48e2b4258445ec13927c5d8615e2fd61f102",
    (42, 46): "3d3ead55e327134b60434fae7a68fa0a348013c75062436a7f7601fd0aecfe0d",
    (43, 42): "30c345a8fc7d0afa30f5cc5b7479fced63b5845c5cfcc98649fd485026498ea6",
    (43, 43): "d9e431d4dfcd1f565c9fc1710f732e95a774f4fb38bb227275886e037c0169f2",
    (43, 44): "e9efe0229f21853522a2997bc41b9c9c3133fe93d70786977a2af9c0f1f4f606",
    (43, 45): "0113cc3c7e0c48b54ff6a15661c3d9df3bf82254d3861b76759bd9ebae576376",
    (43, 46): "f561ad65e175387cd02321131c832604e25f56721fb02b1f1886968f8c5b7507",
    (44, 42): "f989ce0dd13b5f9117f66a526445c8d0270a3c572cb7a7b3536b8f3c67681ece",
    (44, 43): "7887021469aed365caeb1251746319ff035e83742e0e92318b870b4b4f87732a",
    (44, 44): "3766ae7500ee9112bd6e71af97cc46ab8a11c08ef4bf7015658978857e5b2e3b",
    (44, 45): "b49c534b9613b9f3ee72d2f4dda0c0edd49d47f52ea108278d98ceec4c2fd901",
    (44, 46): "1f1689366c74113931f03b27ee49dbc344b5c0d87c877273e08340167f2808c0",
}
JEPA_WMS_BASE_COMMIT = "21389db737ddf1bcbc8dc695d27ca8f7a8b77cb0"
STABLE_WORLDMODEL_COMMIT = "314201d0347baf61304c7558a98463c6a01080f0"
CHECKPOINT_SHA256 = {
    42: "800184d028b3c7af8f3cb69f889307bc7a2433e911cbbcba92f5ddc265b1eb07",
    43: "ef44632fdb57d5c1ea28f52865c38ca9797ad7dd93f9ee542dc78a081937f700",
    44: "96e8e4401a330c08282efd5708f58240d500cf960648626ff0b084b1ef8148b3",
}
CONFIG_SHA256 = {
    42: "d0ca995d841ad41f207fb0ddacd1d35187dfd0faf53c6baf20f7ac8d09afb8e4",
    43: "3f52e56d9c809065c80a6082ba832b62462fc61f552f8d4d8b1e67dc360feb79",
    44: "545b8e1c5e51b364395fa2a1e74c24022d200f6e7330f3a14600c562b2c445a1",
}

FAMILY_SPECS: dict[str, dict[str, str]] = {
    "s0": {"probe_family": "linear", "target_variant": "success"},
    "s1": {"probe_family": "mlp", "target_variant": "success"},
}

CELL_SPECS: dict[str, dict[str, Any]] = {
    "m1_mlp_l03": {"family": "s1", "hybrid_lambda": 0.3, "mode": "hybrid"},
    "m2_mlp_l04": {"family": "s1", "hybrid_lambda": 0.4, "mode": "hybrid"},
    "m3_mlp_l06": {"family": "s1", "hybrid_lambda": 0.6, "mode": "hybrid"},
    "m4_mlp_l07": {"family": "s1", "hybrid_lambda": 0.7, "mode": "hybrid"},
}

SCHEMA_VERSION = "stock_cube_probe_cache_v1"
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")
STOCK_MODEL_RE = re.compile(r"stock_seed(?P<seed>42|43|44)")

DATASET_UNIT_CONTRACT = {
    "coordinate_frame": "OGBench cube world frame, metres",
    "position_fields": ["privileged_block_0_pos", "privileged_target_block_pos"],
    "success_field": "success",
    "episode_field": "ep_idx",
    "position_envelope_m": [-1.0, 1.0],
    "success_radius_m": 0.04,
    "success_radius_atol_m": 1.0e-9,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(canonical_json_bytes(list(array.shape)))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _require_sha256(value: str, label: str) -> str:
    value = str(value).strip().lower()
    if not HEX64.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA256 digest: {value!r}")
    return value


def _dataset_unit_attestation(dataset: Path) -> dict[str, Any]:
    """Prove the block_position target and the r=0.04 m gate share one metric frame.

    The OGBench cube environment declares success when the Euclidean distance
    between the block and its target position is <= 0.04 m
    (stable_worldmodel/envs/ogbench/cube_env.py, _compute_successes, at the
    pinned stable-worldmodel commit).  This attestation verifies, on the full
    dataset, that the recorded per-row ``success`` flag is consistent with that
    radius in the same coordinates the probe decodes (privileged_block_0_pos),
    which grounds the metre-unit FM-4 thresholds.
    """
    import h5py

    required = {"privileged_block_0_pos", "privileged_target_block_pos", "success", "ep_idx"}
    n_rows: int | None = None
    minimum_position = math.inf
    maximum_position = -math.inf
    maximum_success_distance = -math.inf
    minimum_nonsuccess_distance = math.inf
    n_success_rows = 0
    chunk = 250_000
    with h5py.File(dataset, "r") as handle:
        missing = sorted(required.difference(handle.keys()))
        if missing:
            raise ValueError(f"Cube dataset lacks unit-contract fields: {missing}")
        for name in ("privileged_block_0_pos", "privileged_target_block_pos"):
            value = handle[name]
            if value.ndim != 2 or value.shape[1] != 3:
                raise ValueError(f"Cube {name} must have shape (N,3), got {value.shape}")
            if n_rows is None:
                n_rows = int(value.shape[0])
            elif int(value.shape[0]) != n_rows:
                raise ValueError("Cube coordinate fields have different row counts")
        if (
            int(handle["success"].shape[0]) != n_rows
            or int(handle["ep_idx"].shape[0]) != n_rows
        ):
            raise ValueError("Cube unit-contract fields have different row counts")
        assert n_rows is not None
        for start in range(0, n_rows, chunk):
            stop = min(start + chunk, n_rows)
            block = np.asarray(handle["privileged_block_0_pos"][start:stop], dtype=np.float64)
            target = np.asarray(
                handle["privileged_target_block_pos"][start:stop], dtype=np.float64
            )
            success = np.asarray(handle["success"][start:stop], dtype=bool)
            if not (np.isfinite(block).all() and np.isfinite(target).all()):
                raise ValueError("Cube physical coordinate fields contain NaN/Inf")
            minimum_position = min(
                minimum_position, float(block.min()), float(target.min())
            )
            maximum_position = max(
                maximum_position, float(block.max()), float(target.max())
            )
            distance = np.linalg.norm(block - target, axis=1)
            if success.any():
                maximum_success_distance = max(
                    maximum_success_distance, float(distance[success].max())
                )
                n_success_rows += int(success.sum())
            if (~success).any():
                minimum_nonsuccess_distance = min(
                    minimum_nonsuccess_distance, float(distance[~success].min())
                )
    envelope_low, envelope_high = DATASET_UNIT_CONTRACT["position_envelope_m"]
    if minimum_position < envelope_low or maximum_position > envelope_high:
        raise ValueError(
            "Cube block/target coordinates escape the metre envelope "
            f"[{envelope_low}, {envelope_high}]: observed "
            f"[{minimum_position}, {maximum_position}]"
        )
    if n_success_rows <= 0:
        raise ValueError("Cube dataset contains no success rows to attest the radius")
    radius_bound = (
        DATASET_UNIT_CONTRACT["success_radius_m"]
        + DATASET_UNIT_CONTRACT["success_radius_atol_m"]
    )
    if maximum_success_distance > radius_bound:
        raise ValueError(
            "Cube success rows violate the r=0.04 m radius in the probe frame: "
            f"max success distance={maximum_success_distance}"
        )
    payload = {
        **DATASET_UNIT_CONTRACT,
        "n_rows_checked": n_rows,
        "position_min_m": minimum_position,
        "position_max_m": maximum_position,
        "n_success_rows": n_success_rows,
        "max_success_distance_m": maximum_success_distance,
        "min_nonsuccess_distance_m": minimum_nonsuccess_distance,
        "status": "PASS",
    }
    payload["attestation_sha256"] = sha256_json(payload)
    return payload


def _validate_dataset_unit_attestation(payload: Any) -> None:
    if not isinstance(payload, dict):
        raise ValueError("cache lacks the Cube dataset-unit attestation")
    actual = dict(payload)
    recorded = actual.pop("attestation_sha256", None)
    if recorded != sha256_json(actual):
        raise ValueError("Cube dataset-unit attestation hash drift")
    for name, expected in DATASET_UNIT_CONTRACT.items():
        if actual.get(name) != expected:
            raise ValueError(f"Cube unit-contract field drift: {name}")
    if actual.get("status") != "PASS" or int(actual.get("n_rows_checked", 0)) <= 0:
        raise ValueError("Cube dataset-unit attestation is not a positive PASS")
    if int(actual.get("n_success_rows", 0)) <= 0:
        raise ValueError("Cube dataset-unit attestation has no success rows")
    if float(actual["max_success_distance_m"]) > float(
        DATASET_UNIT_CONTRACT["success_radius_m"]
    ) + float(DATASET_UNIT_CONTRACT["success_radius_atol_m"]):
        raise ValueError("Cube success-radius attestation exceeds tolerance")
    envelope_low, envelope_high = DATASET_UNIT_CONTRACT["position_envelope_m"]
    if (
        float(actual["position_min_m"]) < float(envelope_low)
        or float(actual["position_max_m"]) > float(envelope_high)
    ):
        raise ValueError("Cube position-envelope attestation exceeds tolerance")


def _assert_plain_stock_model(model: Any) -> None:
    required = ("encode", "predict", "action_encoder", "predictor")
    missing = [name for name in required if not callable(getattr(model, name, None))]
    if missing:
        raise TypeError(f"stock LeWM is missing native methods: {missing}")
    if all(hasattr(model, name) for name in ("lewm", "nest", "predict_d")):
        raise TypeError("T1-T4 received a NestedLeWM/adapted wrapper instead of true stock LeWM")
    if model.__class__.__name__ != "LeWM":
        raise TypeError(f"T1-T4 require plain LeWM, got {model.__class__.__module__}.{model.__class__.__name__}")
    if bool(getattr(model, "training", True)):
        raise ValueError("stock LeWM must be in eval mode")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("stock LeWM parameters must all be frozen")


def _make_stock_full_a192_shim(stock_model: Any) -> Any:
    """Adapt native LeWM to the frozen nested adapter without changing A192 math."""
    import torch

    _assert_plain_stock_model(stock_model)

    class StockFullA192PlanningShim(torch.nn.Module):
        def __init__(self, model: Any) -> None:
            super().__init__()
            self.lewm = model
            self._stock_full_planning_shim = True

        @staticmethod
        def _require_full(value: Any, budget: int) -> None:
            if int(budget) != 192:
                raise ValueError(f"stock planning shim permits A192 only, got A{budget}")
            if value.shape[-1] != 192:
                raise RuntimeError(f"stock planning expected D=192, got {value.shape[-1]}")

        def encode(self, info: dict[str, Any]) -> tuple[Any, Any]:
            encoded = self.lewm.encode(info)
            if not isinstance(encoded, dict) or "emb" not in encoded:
                raise TypeError("native stock LeWM encode must return a dict containing 'emb'")
            emb = encoded["emb"]
            self._require_full(emb, 192)
            return emb, encoded.get("act_emb")

        def nest(self, emb: Any) -> Any:
            self._require_full(emb, 192)
            return emb

        def select_budget(self, value: Any, budget: int) -> Any:
            self._require_full(value, budget)
            return value

        def predict_d(self, z_ctx: Any, a_ctx: Any, budget: int) -> Any:
            self._require_full(z_ctx, budget)
            if z_ctx.ndim != 3 or a_ctx.ndim != 3:
                raise ValueError("stock predict_d expects rank-3 latent/action contexts")
            if z_ctx.shape[:2] != a_ctx.shape[:2] or a_ctx.shape[-1] != 192:
                raise ValueError("stock latent/action context axes differ")
            predicted = self.lewm.predict(z_ctx, a_ctx)
            if predicted.shape != z_ctx.shape or predicted.shape[-1] != 192:
                raise RuntimeError(
                    f"native stock prediction shape drift: {predicted.shape} != {z_ctx.shape}"
                )
            return predicted

    return StockFullA192PlanningShim(stock_model).eval().requires_grad_(False)


def _fsync_parent(path: Path) -> None:
    if os.name == "posix":
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def atomic_write_json(path: Path, value: Any, *, replace: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise FileExistsError(f"refusing to overwrite immutable file: {path}")
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if tmp.exists():
        raise FileExistsError(tmp)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    try:
        with tmp.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if replace:
            os.replace(tmp, path)
        else:
            os.link(tmp, path)
            tmp.unlink()
        _fsync_parent(path)
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_write_npz(path: Path, metadata: dict[str, Any], arrays: dict[str, np.ndarray]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable cache: {path}")
    clean_arrays: dict[str, np.ndarray] = {}
    for name, raw in arrays.items():
        value = np.asarray(raw)
        if value.dtype == object:
            raise TypeError(f"object dtype is forbidden in cache array {name}")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError(f"non-finite value in cache array {name}")
        clean_arrays[name] = np.ascontiguousarray(value)
    metadata = dict(metadata)
    metadata["array_sha256"] = {
        name: sha256_array(value) for name, value in sorted(clean_arrays.items())
    }
    clean_arrays["metadata_json_utf8"] = np.frombuffer(
        canonical_json_bytes(metadata), dtype=np.uint8
    ).copy()
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if tmp.exists():
        raise FileExistsError(tmp)
    try:
        with tmp.open("xb") as handle:
            np.savez_compressed(handle, **clean_arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(tmp, path)
        tmp.unlink()
        _fsync_parent(path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return sha256_file(path)


def load_npz_cache(path: Path, *, expected_sha256: str | None = None) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if expected_sha256 is not None:
        actual = sha256_file(path)
        if actual != expected_sha256:
            raise ValueError(
                f"cache hash mismatch for {path}: expected={expected_sha256} actual={actual}"
            )
    with np.load(path, allow_pickle=False) as archive:
        if "metadata_json_utf8" not in archive.files:
            raise ValueError(f"cache has no metadata payload: {path}")
        metadata = json.loads(archive["metadata_json_utf8"].tobytes().decode("utf-8"))
        arrays = {
            name: np.array(archive[name], copy=True)
            for name in archive.files
            if name != "metadata_json_utf8"
        }
    expected_arrays = metadata.get("array_sha256")
    if not isinstance(expected_arrays, dict) or set(expected_arrays) != set(arrays):
        raise ValueError(f"cache array manifest mismatch: {path}")
    for name, value in arrays.items():
        if expected_arrays[name] != sha256_array(value):
            raise ValueError(f"cache array hash mismatch: {path}:{name}")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError(f"cache contains non-finite values: {path}:{name}")
    return metadata, arrays


def verify_sources(args: argparse.Namespace) -> None:
    expected = {
        Path(args.probe_runner): PROBE_RUNNER_SHA256,
        Path(args.probe_common): PROBE_COMMON_SHA256,
        Path(args.budget_runner): BUDGET_RUNNER_SHA256,
        Path(args.task_manifest): TASK_MANIFEST_SHA256,
    }
    for path, wanted in expected.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual != wanted:
            raise ValueError(f"source hash mismatch: {path} expected={wanted} actual={actual}")


def _import_path(module_name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_frozen_modules(args: argparse.Namespace) -> tuple[Any, Any]:
    verify_sources(args)
    common_name = "candidate_ranking_alignment_adadim_block_20260703"
    sys.modules.pop(common_name, None)
    common = _import_path(common_name, Path(args.probe_common))
    runner_name = "probe_physical_cost_cem_adadim_block_20260703"
    sys.modules.pop(runner_name, None)
    runner = _import_path(runner_name, Path(args.probe_runner))
    if runner.common is not common:
        raise RuntimeError("frozen probe runner did not bind the verified common module")
    return common, runner


def _encode_stock_aware_latents(
    common: Any,
    base: Any,
    model: Any,
    h5_path: Path,
    indices: np.ndarray,
    *,
    batch_size: int,
    device: str,
) -> np.ndarray:
    """Frozen common encoding semantics extended only for native stock LeWM.

    The reviewed common snapshot assumes ``NestedLeWM.encode -> (emb, aux)``
    followed by ``nest``.  Stock ``LeWM.encode`` instead returns an info dict
    whose ``emb`` field is already the A192 latent.  The two branches below are
    the same branches used by the canonical static-probe implementation; all
    sampling, normalization, target, split, and probe-fit code remains frozen.
    """
    import h5py
    import torch

    features: np.ndarray | None = None
    is_nested = all(hasattr(model, name) for name in ("lewm", "nest", "predict_d"))
    with h5py.File(h5_path, "r") as handle, torch.inference_mode():
        pixels = handle["pixels"]
        for start in range(0, len(indices), batch_size):
            end = min(start + batch_size, len(indices))
            raw = common.read_h5_batch(pixels, indices[start:end])
            inputs = common.normalize_pixels(base, raw, device).unsqueeze(1)
            encoded = model.encode({"pixels": inputs})
            if is_nested:
                if not isinstance(encoded, tuple) or len(encoded) != 2:
                    raise TypeError("NestedLeWM encode contract drift")
                latent = model.nest(encoded[0])
            else:
                if not isinstance(encoded, dict) or "emb" not in encoded:
                    raise TypeError("stock LeWM encode must return an info dict with 'emb'")
                latent = encoded["emb"]
            if latent.ndim != 3 or latent.shape[1] != 1 or latent.shape[-1] != 192:
                raise ValueError(
                    "probe encoding expected (batch,1,192), got "
                    f"{tuple(latent.shape)}"
                )
            values = latent[:, 0].detach().float().cpu().numpy()
            if features is None:
                features = np.empty((len(indices), 192), dtype=np.float32)
            features[start:end] = values
    if features is None:
        raise RuntimeError("no stock latents encoded for probe")
    return features


def _val_mask(length: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 991)
    mask = rng.random(length) < 0.2
    if mask.all() or not mask.any():
        mask = np.zeros(length, dtype=bool)
        mask[: max(1, length // 5)] = True
    return mask


def _prepared_eval_rows(
    prepared: dict[str, Any], train_seed: int, eval_seed: int
) -> list[list[int]]:
    arrays = [
        np.asarray(prepared[name], dtype=np.int64).tolist()
        for name in ("eval_indices", "eval_episodes", "eval_start_idx")
    ]
    if any(len(values) != 50 for values in arrays):
        raise ValueError(
            f"prepared true-stock pair must contain 50 evaluation starts: "
            f"train={train_seed} eval={eval_seed}"
        )
    return [
        [
            int(train_seed),
            int(eval_seed),
            int(env_index),
            int(arrays[0][env_index]),
            int(arrays[1][env_index]),
            int(arrays[2][env_index]),
        ]
        for env_index in range(50)
    ]


def _assert_prepared_eval_identity(
    prepared: dict[str, Any], train_seed: int, eval_seed: int
) -> list[list[int]]:
    rows = _prepared_eval_rows(prepared, train_seed, eval_seed)
    actual = sha256_json(rows)
    expected = BASELINE_PAIR_EVAL_ARRAYS_SHA256[(train_seed, eval_seed)]
    if actual != expected:
        raise ValueError(
            "prepared evaluation arrays differ from the frozen true-stock baseline "
            f"for train={train_seed} eval={eval_seed}: expected={expected} actual={actual}"
        )
    return rows


def _fit_mlp_probes_with_metrics(
    common: Any,
    runner: Any,
    base: Any,
    model: Any,
    dataset: Path,
    *,
    exclude_episodes: set[int],
    seed: int,
    device: str,
) -> tuple[dict[int, Any], dict[int, dict[str, Any]], np.ndarray, np.ndarray]:
    """Literal extension of the frozen MLP fit that retains validation metrics."""
    import torch
    from sklearn.preprocessing import StandardScaler

    indices = common.make_probe_indices(
        dataset,
        exclude_episodes=exclude_episodes,
        n_train=PROBE_TRAIN_SAMPLES,
        seed=seed,
    )
    z_full = common.encode_latents(
        base,
        model,
        dataset,
        indices,
        batch_size=PROBE_BATCH_SIZE,
        device=device,
    )
    y, target_names, target_blocks = common.target_matrix(TASK, dataset, indices)
    val_mask = _val_mask(len(indices), seed)
    y_scaler = StandardScaler().fit(y[~val_mask])
    y_z = y_scaler.transform(y).astype(np.float32)
    dev = torch.device(device)
    dtype = next(model.parameters()).dtype
    if dtype != torch.float32:
        raise ValueError(
            f"frozen S1 fit requires canonical float32 model parameters, found {dtype}"
        )
    probes: dict[int, Any] = {}
    details: dict[int, dict[str, Any]] = {}
    for budget in BUDGETS:
        x = z_full[:, :budget]
        x_scaler = StandardScaler().fit(x[~val_mask])
        x_z = x_scaler.transform(x).astype(np.float32)
        torch.manual_seed(seed + budget)
        net = runner._MLPNet(budget, y_z.shape[1]).to(dev)
        optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        x_train = torch.as_tensor(x_z[~val_mask], device=dev)
        y_train = torch.as_tensor(y_z[~val_mask], device=dev)
        net.train()
        for _ in range(MLP_EPOCHS):
            optimizer.zero_grad()
            ((net(x_train) - y_train) ** 2).mean().backward()
            optimizer.step()
        q95 = float(np.quantile(np.mean(x_z[~val_mask] ** 2, axis=1), 0.95))
        net.eval()
        with torch.inference_mode():
            pred_val = (
                net(torch.as_tensor(x_z[val_mask], device=dev))
                .detach()
                .float()
                .cpu()
                .numpy()
            )
        residual = pred_val.astype(np.float64) - y_z[val_mask].astype(np.float64)
        mse_z = np.mean(residual**2, axis=0)
        rmse_z = np.sqrt(mse_z)
        y_scale = np.maximum(y_scaler.scale_, 1e-12).astype(np.float64)
        rmse_phys = rmse_z * y_scale
        probe = runner.TorchMLPProbe(
            budget,
            net,
            x_scaler.mean_.astype(np.float64),
            np.maximum(x_scaler.scale_, 1e-12).astype(np.float64),
            target_blocks,
            target_names,
            y.shape[1],
            q95,
            dev,
            dtype,
        )
        probes[budget] = probe
        details[budget] = {
            "x_mean": x_scaler.mean_.astype(np.float64),
            "x_scale": np.maximum(x_scaler.scale_, 1e-12).astype(np.float64),
            "y_mean": y_scaler.mean_.astype(np.float64),
            "y_scale": y_scale,
            "val_rmse_z": rmse_z.astype(np.float64),
            "val_rmse_phys": rmse_phys.astype(np.float64),
            "val_mse_z": mse_z.astype(np.float64),
            "n_val": int(val_mask.sum()),
            "target_names": tuple(str(x) for x in target_names),
            "target_blocks": tuple((int(s), int(e)) for s, e in target_blocks),
            "target_dim": int(y.shape[1]),
            "whitened_q95": q95,
            "state_dict": {
                name: tensor.detach().float().cpu().numpy().copy()
                for name, tensor in net.state_dict().items()
            },
        }
    return probes, details, indices, val_mask


def _pair_arrays(prepared: dict[str, Any], sample_indices: np.ndarray, val_mask: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "sample_indices": np.asarray(sample_indices, dtype=np.int64),
        "val_mask": np.asarray(val_mask, dtype=np.uint8),
        "planning_eval_indices": np.asarray(prepared["eval_indices"], dtype=np.int64),
        "planning_eval_episodes": np.asarray(prepared["eval_episodes"], dtype=np.int64),
        "planning_eval_start_idx": np.asarray(prepared["eval_start_idx"], dtype=np.int64),
    }


def _linear_cache_payload(
    probes: dict[int, Any],
    prepared: dict[str, Any],
    sample_indices: np.ndarray,
    val_mask: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    arrays = _pair_arrays(prepared, sample_indices, val_mask)
    budget_meta: dict[str, Any] = {}
    for budget in BUDGETS:
        probe = probes[budget]
        prefix = f"b{budget}_"
        values = {
            "x_mean": np.asarray(probe.x_mean, dtype=np.float64),
            "x_scale": np.asarray(probe.x_scale, dtype=np.float64),
            "y_mean": np.asarray(probe.y_mean, dtype=np.float64),
            "y_scale": np.asarray(probe.y_scale, dtype=np.float64),
            "coef": np.asarray(probe.coef, dtype=np.float64),
            "intercept": np.asarray(probe.intercept, dtype=np.float64),
            "val_rmse_z": np.asarray(probe.val_rmse_z, dtype=np.float64),
        }
        values["val_rmse_phys"] = values["val_rmse_z"] * values["y_scale"]
        for name, value in values.items():
            arrays[prefix + name] = value
        budget_meta[str(budget)] = {
            "target_dim": int(probe.target_dim),
            "target_names": list(probe.target_names),
            "target_blocks": [list(block) for block in probe.target_blocks],
            "whitened_q95": float(probe.whitened_q95),
            "val_rmse_z_max": float(values["val_rmse_z"].max()),
            "val_rmse_phys_max": float(values["val_rmse_phys"].max()),
        }
    return arrays, budget_meta


def _mlp_cache_payload(
    details: dict[int, dict[str, Any]],
    prepared: dict[str, Any],
    sample_indices: np.ndarray,
    val_mask: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    arrays = _pair_arrays(prepared, sample_indices, val_mask)
    budget_meta: dict[str, Any] = {}
    for budget in BUDGETS:
        detail = details[budget]
        prefix = f"b{budget}_"
        for name in (
            "x_mean",
            "x_scale",
            "y_mean",
            "y_scale",
            "val_rmse_z",
            "val_rmse_phys",
            "val_mse_z",
        ):
            arrays[prefix + name] = np.asarray(detail[name])
        state_keys: dict[str, str] = {}
        for index, (name, value) in enumerate(sorted(detail["state_dict"].items())):
            array_name = f"{prefix}state_{index}"
            arrays[array_name] = np.asarray(value)
            state_keys[name] = array_name
        budget_meta[str(budget)] = {
            "target_dim": int(detail["target_dim"]),
            "target_names": list(detail["target_names"]),
            "target_blocks": [list(block) for block in detail["target_blocks"]],
            "whitened_q95": float(detail["whitened_q95"]),
            "n_val": int(detail["n_val"]),
            "val_rmse_z_max": float(np.max(detail["val_rmse_z"])),
            "val_rmse_phys_max": float(np.max(detail["val_rmse_phys"])),
            "state_keys": state_keys,
            "architecture": [budget, 256, 256, int(detail["target_dim"])],
        }
    return arrays, budget_meta


def _gate_rows(
    family: str,
    train_seed: int,
    eval_seed: int,
    budget_meta: dict[str, Any],
    arrays: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for budget in BUDGETS:
        meta = budget_meta[str(budget)]
        per_coordinate = arrays[f"b{budget}_val_rmse_phys"].astype(float).tolist()
        rows.append(
            {
                "family": family,
                "task": TASK,
                "train_seed": train_seed,
                "eval_seed": eval_seed,
                "effective_seed": PROBE_SEED + eval_seed,
                "budget": budget,
                "target_names": meta["target_names"],
                "val_rmse_z_per_coordinate": arrays[f"b{budget}_val_rmse_z"].astype(float).tolist(),
                "val_rmse_phys_per_coordinate": per_coordinate,
                "val_rmse_z_max": float(max(arrays[f"b{budget}_val_rmse_z"])),
                "val_rmse_phys_max": float(max(per_coordinate)),
            }
        )
    return rows


def _assert_probe_partition(
    dataset: Path,
    sample_indices: np.ndarray,
    exclude_episodes: set[int],
    eval_episodes: np.ndarray,
) -> None:
    import h5py

    indices = np.asarray(sample_indices, dtype=np.int64)
    if indices.shape != (PROBE_TRAIN_SAMPLES,) or len(np.unique(indices)) != len(indices):
        raise ValueError(
            f"probe fit must use exactly {PROBE_TRAIN_SAMPLES} unique rows, got {indices.shape}"
        )
    eval_episode_ids = np.asarray(eval_episodes, dtype=np.int64)
    if eval_episode_ids.shape != (50,):
        raise ValueError(
            "planning evaluation identity must contain exactly 50 ordered starts, "
            f"got {eval_episode_ids.shape}"
        )
    expected_exclusions = {int(value) for value in eval_episode_ids.tolist()}
    if exclude_episodes != expected_exclusions:
        raise ValueError("planning exclusion set differs from the prepared evaluation episodes")
    with h5py.File(dataset, "r") as handle:
        n_rows = int(handle["ep_idx"].shape[0])
        if int(indices.min()) < 0 or int(indices.max()) >= n_rows:
            raise ValueError("probe sample index is outside the Cube dataset")
        sample_episodes = np.asarray(handle["ep_idx"][indices], dtype=np.int64)
    overlap = sorted(set(sample_episodes.tolist()).intersection(exclude_episodes))
    if overlap:
        raise ValueError(f"probe fit leaked planning-evaluation episodes: {overlap[:10]}")


def _assert_cube_target_contract(budget_meta: dict[str, Any]) -> None:
    for budget in BUDGETS:
        metadata = budget_meta[str(budget)]
        if (
            int(metadata.get("target_dim", -1)) != 3
            or list(metadata.get("target_names", [])) != ["block_position"]
            or [list(block) for block in metadata.get("target_blocks", [])] != [[0, 3]]
        ):
            raise ValueError(
                "Cube success probe must decode block_position (3 coordinates, metres): "
                f"budget={budget} metadata={metadata}"
            )


def _base_cache_metadata(
    *,
    family: str,
    train_seed: int,
    eval_seed: int,
    checkpoint: Path,
    dataset: Path,
    exclude_episodes: Iterable[int],
    budget_meta: dict[str, Any],
    dataset_unit_attestation: dict[str, Any],
) -> dict[str, Any]:
    excluded = sorted({int(x) for x in exclude_episodes})
    spec = FAMILY_SPECS[family]
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": RUN_ID,
        "family": family,
        "probe_family": spec["probe_family"],
        "target_variant": spec["target_variant"],
        "probe_mode": "linear",
        "task": TASK,
        "arm": ARM,
        "train_seed": train_seed,
        "eval_seed": eval_seed,
        "checkpoint_epoch": 10,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": CHECKPOINT_SHA256[train_seed],
        "checkpoint_config_sha256": CONFIG_SHA256[train_seed],
        "dataset_path": str(dataset),
        "dataset_stat": {
            "size": dataset.stat().st_size,
            "mtime_ns": dataset.stat().st_mtime_ns,
        },
        "budgets": list(BUDGETS),
        "probe_train_samples": PROBE_TRAIN_SAMPLES,
        "probe_batch_size": PROBE_BATCH_SIZE,
        "probe_seed": PROBE_SEED,
        "effective_seed": PROBE_SEED + eval_seed,
        "split": "80/20 mask from default_rng(effective_seed+991)",
        "exclude_episodes": excluded,
        "exclude_episodes_sha256": sha256_json(excluded),
        "budget_metadata": budget_meta,
        "probe_runner_sha256": PROBE_RUNNER_SHA256,
        "probe_common_sha256": PROBE_COMMON_SHA256,
        "budget_runner_sha256": BUDGET_RUNNER_SHA256,
        "task_manifest_sha256": TASK_MANIFEST_SHA256,
        "baseline_manifest_sha256": BASELINE_MANIFEST_SHA256,
        "baseline_eval_arrays_sha256": BASELINE_EVAL_ARRAYS_SHA256,
        "baseline_pair_eval_arrays_sha256": BASELINE_PAIR_EVAL_ARRAYS_SHA256[
            (train_seed, eval_seed)
        ],
        "jepa_wms_base_commit": JEPA_WMS_BASE_COMMIT,
        "stable_worldmodel_commit": STABLE_WORLDMODEL_COMMIT,
        "success_radius_m": SUCCESS_RADIUS_M,
        "pure_gate_threshold_m": PURE_GATE_THRESHOLD_M,
        "hybrid_gate_threshold_m": HYBRID_GATE_THRESHOLD_M,
        "physical_coordinate_unit": "metres (OGBench cube world frame)",
        "dataset_unit_attestation": dataset_unit_attestation,
        "mlp_epochs": MLP_EPOCHS if family == "s1" else None,
        "created_unix": time.time(),
    }


def _cache_relpath(family: str, train_seed: int, eval_seed: int) -> Path:
    return Path("probe_caches") / family / f"ts{train_seed}_es{eval_seed}.npz"


def _validate_checkpoint(checkpoint: Path, train_seed: int) -> None:
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    expected_name = f"stock_lewm_cube_matched3_seed{train_seed}_20260615"
    if checkpoint.name != "weights_epoch_10.pt" or checkpoint.parent.name != expected_name:
        raise ValueError(f"canonical checkpoint lineage drift: {checkpoint}")
    actual = sha256_file(checkpoint)
    if actual != CHECKPOINT_SHA256[train_seed]:
        raise ValueError(
            f"checkpoint hash mismatch seed={train_seed}: expected={CHECKPOINT_SHA256[train_seed]} actual={actual}"
        )
    config = checkpoint.parent / "config.json"
    if not config.is_file():
        raise FileNotFoundError(config)
    actual_config = sha256_file(config)
    if actual_config != CONFIG_SHA256[train_seed]:
        raise ValueError(
            f"checkpoint config hash mismatch seed={train_seed}: "
            f"expected={CONFIG_SHA256[train_seed]} actual={actual_config}"
        )
    config_payload = json.loads(config.read_text(encoding="utf-8"))
    model_target = str(config_payload.get("model", {}).get("_target_", ""))
    if not model_target.endswith(".LeWM") or "NestedLeWM" in model_target:
        raise ValueError(f"checkpoint config is not native stock LeWM: {model_target!r}")


def import_caches(args: argparse.Namespace) -> dict[str, Any]:
    """Import the sealed probe caches from the source run, hash-verified end to end.

    No probe is refit anywhere in this run: every cache byte comes from the sealed
    source run and must match the preregistered per-file pins both before and after
    the copy.  The fit summary and smoke manifest are carried over byte-identically.
    """
    run_root = Path(args.run_root).resolve()
    if run_root.name != RUN_ID:
        raise ValueError(f"run root basename must be {RUN_ID}: {run_root}")
    source_root = Path(args.source_root).resolve()
    if source_root.name != SOURCE_RUN_ID:
        raise ValueError(f"source root basename must be {SOURCE_RUN_ID}: {source_root}")
    probes_root = run_root / "probes"
    caches_root = run_root / "probe_caches"
    if probes_root.exists() or caches_root.exists():
        raise FileExistsError("cache import target already exists")

    src_summary = (source_root / "probes" / "probe_fit_summary.json").read_bytes()
    if hashlib.sha256(src_summary).hexdigest() != SOURCE_FIT_SUMMARY_SHA256:
        raise ValueError("source fit summary hash drift")
    summary = json.loads(src_summary.decode("utf-8"))
    if summary.get("n_cache_artifacts") != 30:
        raise ValueError("source fit summary cache count drift")
    if summary.get("cache_artifact_sha256") != IMPORTED_CACHE_SHA256:
        raise ValueError("source cache hash map differs from the preregistered import pins")
    _validate_dataset_unit_attestation(summary.get("dataset_unit_attestation"))
    src_smoke = (source_root / "probes" / "smoke_eval_seed42.json").read_bytes()
    if hashlib.sha256(src_smoke).hexdigest() != SOURCE_SMOKE_MANIFEST_SHA256:
        raise ValueError("source smoke manifest hash drift")

    staged: list[tuple[str, bytes]] = []
    for rel_path in sorted(IMPORTED_CACHE_SHA256):
        payload = (source_root / rel_path).read_bytes()
        if hashlib.sha256(payload).hexdigest() != IMPORTED_CACHE_SHA256[rel_path]:
            raise ValueError(f"cache hash drift during import: {rel_path}")
        staged.append((rel_path, payload))

    for rel_path, payload in staged:
        target = run_root / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".import.tmp")
        tmp.write_bytes(payload)
        if sha256_file(tmp) != IMPORTED_CACHE_SHA256[rel_path]:
            tmp.unlink()
            raise ValueError(f"cache verification failed after write: {rel_path}")
        tmp.replace(target)

    probes_root.mkdir(parents=True, exist_ok=False)
    smoke_tmp = probes_root / "smoke_eval_seed42.json.import.tmp"
    smoke_tmp.write_bytes(src_smoke)
    if sha256_file(smoke_tmp) != SOURCE_SMOKE_MANIFEST_SHA256:
        raise ValueError("smoke manifest verification failed after write")
    smoke_tmp.replace(probes_root / "smoke_eval_seed42.json")

    receipt = {
        "schema_version": "stock_cube_lambda_sweep_import_v1",
        "run_id": RUN_ID,
        "source_run_id": SOURCE_RUN_ID,
        "source_fit_summary_sha256": SOURCE_FIT_SUMMARY_SHA256,
        "source_smoke_manifest_sha256": SOURCE_SMOKE_MANIFEST_SHA256,
        "n_cache_files": len(IMPORTED_CACHE_SHA256),
        "cache_artifact_sha256": IMPORTED_CACHE_SHA256,
    }
    atomic_write_json(probes_root / "cache_import_receipt.json", receipt)

    summary_tmp = probes_root / "probe_fit_summary.json.import.tmp"
    summary_tmp.write_bytes(src_summary)
    if sha256_file(summary_tmp) != SOURCE_FIT_SUMMARY_SHA256:
        raise ValueError("fit summary verification failed after write")
    summary_tmp.replace(probes_root / "probe_fit_summary.json")
    print(f"cache_import_ok n={len(IMPORTED_CACHE_SHA256)}")
    return receipt


def _record_lookup(summary: dict[str, Any]) -> dict[tuple[str, int, int], dict[str, Any]]:
    records = summary.get("records")
    if not isinstance(records, list) or len(records) != 30:
        raise ValueError("probe_fit_summary does not contain exactly 30 cache records")
    lookup: dict[tuple[str, int, int], dict[str, Any]] = {}
    for record in records:
        key = (str(record["family"]), int(record["train_seed"]), int(record["eval_seed"]))
        if key in lookup:
            raise ValueError(f"duplicate cache record {key}")
        lookup[key] = record
    return lookup


def recompute_prospective_predictions(
    run_root: Path,
    fit_summary: dict[str, Any],
) -> dict[str, Any]:
    """Deterministic exploratory artifact: no FM-4 gate, no predictions.

    The prospective-publication ordering machinery of the source protocol is kept
    intact, but the artifact only republishes descriptive A192 readout errors from
    the imported sealed caches and records that every cell launches unconditionally.
    It can never gate, upgrade, or downgrade anything.
    """
    del run_root  # deterministic function of the fit summary alone
    per_family: dict[str, list[float]] = {}
    for record in fit_summary["records"]:
        family = str(record["family"])
        for gate_row in record["gate_rows"]:
            if int(gate_row["budget"]) != 192:
                continue
            per_family.setdefault(family, []).append(float(gate_row["val_rmse_phys_max"]))
    descriptive = {
        family: {
            "n_caches": len(values),
            "a192_val_rmse_phys_max_worst": max(values),
            "a192_val_rmse_phys_max_mean": sum(values) / len(values),
        }
        for family, values in sorted(per_family.items())
    }
    cells = {
        cell: {
            "family": spec["family"],
            "mode": spec["mode"],
            "hybrid_lambda": spec["hybrid_lambda"],
            "gate_status": GATE_STATUS_EXPLORATORY,
            "prediction": "NONE_EXPLORATORY_SWEEP",
            "launch_policy": "RUN_REGARDLESS_EXPLORATORY",
        }
        for cell, spec in CELL_SPECS.items()
    }
    return {
        "schema_version": "stock_cube_lambda_sweep_prospective_v1",
        "run_id": RUN_ID,
        "source_run_id": SOURCE_RUN_ID,
        "gate_status": GATE_STATUS_EXPLORATORY,
        "statement": (
            "Exploratory lambda-transfer sweep: no FM-4 gate applies, no PASS/FAIL "
            "prediction is made, no confirmatory family exists, and no existing "
            "claim can change; descriptive A192 readout errors are republished "
            "from the imported sealed caches of the source run."
        ),
        "prediction_cells": list(CELL_SPECS),
        "cells": cells,
        "descriptive_readout": descriptive,
    }


def publish_prospective_predictions(args: argparse.Namespace) -> dict[str, Any]:
    run_root = Path(args.run_root).resolve()
    if run_root.name != RUN_ID or not run_root.is_dir():
        raise ValueError(f"invalid immutable run root: {run_root}")
    state_root = run_root / "state"
    if any(
        (state_root / name).exists()
        for name in (
            "smoke.attempted.json",
            "smoke.launched.json",
            "full.attempted.json",
            "full.launched.json",
        )
    ):
        raise RuntimeError("prospective predictions cannot be published after planning is claimed")
    if list((run_root / "smoke").glob("cube_*")) or list(
        (run_root / "full").glob("cube_*")
    ):
        raise RuntimeError("prospective predictions cannot be published after planning output exists")
    summary_path = run_root / "probes" / "probe_fit_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("n_cache_artifacts") != 30:
        raise ValueError("fit summary cache count drift")
    predictions = recompute_prospective_predictions(run_root, summary)
    destination = run_root / "probes" / "prospective_gate_predictions.json"
    atomic_write_json(destination, predictions)
    round_trip = json.loads(destination.read_text(encoding="utf-8"))
    if round_trip != predictions:
        raise RuntimeError("prospective prediction artifact round-trip drift")
    print(f"prospective_predictions_sha256={sha256_file(destination)}")
    return predictions


def _validate_loaded_metadata(
    metadata: dict[str, Any],
    *,
    family: str,
    train_seed: int,
    eval_seed: int,
) -> None:
    expected = {
        "schema_version": SCHEMA_VERSION,
        # imported sealed caches carry the source run identity
        "run_id": SOURCE_RUN_ID,
        "family": family,
        "probe_family": FAMILY_SPECS[family]["probe_family"],
        "target_variant": FAMILY_SPECS[family]["target_variant"],
        "probe_mode": "linear",
        "task": TASK,
        "arm": ARM,
        "train_seed": train_seed,
        "eval_seed": eval_seed,
        "checkpoint_epoch": 10,
        "checkpoint_sha256": CHECKPOINT_SHA256[train_seed],
        "checkpoint_config_sha256": CONFIG_SHA256[train_seed],
        "budgets": list(BUDGETS),
        "probe_train_samples": PROBE_TRAIN_SAMPLES,
        "probe_batch_size": PROBE_BATCH_SIZE,
        "probe_seed": PROBE_SEED,
        "effective_seed": PROBE_SEED + eval_seed,
        "probe_runner_sha256": PROBE_RUNNER_SHA256,
        "probe_common_sha256": PROBE_COMMON_SHA256,
        "budget_runner_sha256": BUDGET_RUNNER_SHA256,
        "task_manifest_sha256": TASK_MANIFEST_SHA256,
        "baseline_manifest_sha256": BASELINE_MANIFEST_SHA256,
        "baseline_eval_arrays_sha256": BASELINE_EVAL_ARRAYS_SHA256,
        "baseline_pair_eval_arrays_sha256": BASELINE_PAIR_EVAL_ARRAYS_SHA256[
            (train_seed, eval_seed)
        ],
        "jepa_wms_base_commit": JEPA_WMS_BASE_COMMIT,
        "stable_worldmodel_commit": STABLE_WORLDMODEL_COMMIT,
        "success_radius_m": SUCCESS_RADIUS_M,
        "pure_gate_threshold_m": PURE_GATE_THRESHOLD_M,
        "hybrid_gate_threshold_m": HYBRID_GATE_THRESHOLD_M,
        "physical_coordinate_unit": "metres (OGBench cube world frame)",
    }
    for name, wanted in expected.items():
        if metadata.get(name) != wanted:
            raise ValueError(
                f"cache metadata mismatch {name}: expected={wanted!r} actual={metadata.get(name)!r}"
            )
    _validate_dataset_unit_attestation(metadata.get("dataset_unit_attestation"))
    _assert_cube_target_contract(metadata.get("budget_metadata", {}))


def _reconstruct_linear(common: Any, metadata: dict[str, Any], arrays: dict[str, np.ndarray]) -> dict[int, Any]:
    probes: dict[int, Any] = {}
    for budget in BUDGETS:
        prefix = f"b{budget}_"
        budget_meta = metadata["budget_metadata"][str(budget)]
        probes[budget] = common.LinearProbeBundle(
            task=TASK,
            budget=budget,
            x_mean=arrays[prefix + "x_mean"],
            x_scale=arrays[prefix + "x_scale"],
            y_mean=arrays[prefix + "y_mean"],
            y_scale=arrays[prefix + "y_scale"],
            coef=arrays[prefix + "coef"],
            intercept=arrays[prefix + "intercept"],
            target_dim=int(budget_meta["target_dim"]),
            target_names=tuple(budget_meta["target_names"]),
            target_blocks=tuple(tuple(x) for x in budget_meta["target_blocks"]),
            val_rmse_z=arrays[prefix + "val_rmse_z"],
            whitened_q95=float(budget_meta["whitened_q95"]),
        )
    return probes


def _reconstruct_mlp(
    runner: Any,
    metadata: dict[str, Any],
    arrays: dict[str, np.ndarray],
    *,
    model: Any,
    device: str,
) -> dict[int, Any]:
    import torch

    probes: dict[int, Any] = {}
    dtype = next(model.parameters()).dtype
    if dtype != torch.float32:
        raise ValueError(
            f"cached S1 planning requires canonical float32 model parameters, found {dtype}"
        )
    dev = torch.device(device)
    for budget in BUDGETS:
        prefix = f"b{budget}_"
        budget_meta = metadata["budget_metadata"][str(budget)]
        net = runner._MLPNet(budget, int(budget_meta["target_dim"]))
        state_dict = {
            name: torch.as_tensor(arrays[array_name])
            for name, array_name in budget_meta["state_keys"].items()
        }
        net.load_state_dict(state_dict, strict=True)
        probes[budget] = runner.TorchMLPProbe(
            budget,
            net,
            arrays[prefix + "x_mean"],
            arrays[prefix + "x_scale"],
            tuple(tuple(x) for x in budget_meta["target_blocks"]),
            tuple(budget_meta["target_names"]),
            int(budget_meta["target_dim"]),
            float(budget_meta["whitened_q95"]),
            dev,
            dtype,
        )
    return probes


def _same_array(left: Any, right: np.ndarray) -> bool:
    return np.array_equal(np.asarray(left), np.asarray(right))


def _validate_planning_ordering_state(
    run_root: Path,
    *,
    mode: str,
    cell: str,
    baseline_sha256: str,
    prediction_sha256: str,
    fit_summary: dict[str, Any],
) -> None:
    """Make the prospective publication order mandatory even for direct calls."""
    provenance = run_root / "provenance"
    recorded_baseline = (provenance / "baseline_manifest_sha256.txt").read_text(
        encoding="utf-8"
    ).strip()
    recorded_prediction = (
        provenance / "prospective_gate_predictions_sha256.txt"
    ).read_text(encoding="utf-8").strip()
    if recorded_baseline != baseline_sha256 or recorded_prediction != prediction_sha256:
        raise ValueError("planning provenance is not bound to the baseline/prediction hashes")

    records = fit_summary.get("records")
    if not isinstance(records, list) or len(records) != 30:
        raise ValueError("fit summary lacks the exact 30-cache record matrix")
    record_hashes = {
        str(record["path"]): _require_sha256(record["sha256"], "cache artifact SHA256")
        for record in records
    }
    record_hashes = {path: record_hashes[path] for path in sorted(record_hashes)}
    expected_paths = {
        _cache_relpath(family, train_seed, eval_seed).as_posix()
        for family in FAMILY_SPECS
        for train_seed in TRAIN_SEEDS
        for eval_seed in EVAL_SEEDS
    }
    if (
        set(record_hashes) != expected_paths
        or fit_summary.get("cache_artifact_sha256") != record_hashes
    ):
        raise ValueError("fit-summary cache artifact map drift")

    cache_validated_path = run_root / "cache_validated.json"
    cache_validated = json.loads(cache_validated_path.read_text(encoding="utf-8"))
    expected_cache_validated = {
        "run_id": RUN_ID,
        "state": "CACHE_AND_PREDICTIONS_VALIDATED",
        "n_cache_artifacts": 30,
        "fit_summary_sha256": sha256_file(
            run_root / "probes" / "probe_fit_summary.json"
        ),
        "cache_artifact_set_sha256": sha256_json(record_hashes),
        "prospective_gate_predictions_sha256": prediction_sha256,
        "baseline_manifest_sha256": baseline_sha256,
        "prediction_cells": list(CELL_SPECS),
    }
    if cache_validated != expected_cache_validated:
        raise ValueError("planning cache_validated hash-DAG payload drift")

    attempted_path = run_root / "state" / f"{mode}.attempted.json"
    attempted = json.loads(attempted_path.read_text(encoding="utf-8"))
    attempted_exact = {
        "state": f"{mode.upper()}_ATTEMPTED",
        "cells": list(CELL_SPECS),
        "cache_validated_sha256": sha256_file(cache_validated_path),
        "prospective_gate_predictions_sha256": prediction_sha256,
        "baseline_manifest_sha256": baseline_sha256,
    }
    for name, expected in attempted_exact.items():
        if attempted.get(name) != expected:
            raise ValueError(f"planning attempted-state hash-DAG drift: {name}")
    if not isinstance(attempted.get("launch_token"), str) or not attempted["launch_token"]:
        raise ValueError("planning attempted state lacks a launch token")
    attempted_dataset_sha = _require_sha256(
        attempted.get("dataset_sha256", ""), "planning dataset SHA256"
    )
    recorded_dataset_sha = _require_sha256(
        (provenance / "cube_dataset_sha256.txt").read_text(encoding="utf-8").strip(),
        "recorded planning dataset SHA256",
    )
    if attempted_dataset_sha != recorded_dataset_sha:
        raise ValueError("planning attempted state is not bound to the recorded dataset")
    if not isinstance(attempted.get("at"), str) or not attempted["at"]:
        raise ValueError("planning attempted state lacks a timestamp")
    smoke_attestation = attempted.get("smoke_validation_attestation_sha256")
    if mode == "smoke":
        if smoke_attestation != "":
            raise ValueError("smoke attempted state must precede smoke validation")
    else:
        frozen_smoke_attestation = _require_sha256(
            smoke_attestation, "full-launch smoke attestation SHA256"
        )
        if frozen_smoke_attestation != sha256_file(
            run_root / "smoke_validation_attestation.json"
        ):
            raise ValueError("full planning is not bound to the smoke validation attestation")

    cell_attempted = json.loads(
        (run_root / "state" / f"{mode}_{cell}.attempted.json").read_text(
            encoding="utf-8"
        )
    )
    expected_gpu = list(CELL_SPECS).index(cell)
    if (
        cell_attempted.get("state") != f"{mode.upper()}_CELL_ATTEMPTED"
        or cell_attempted.get("cell") != cell
        or int(cell_attempted.get("gpu", -1)) != expected_gpu
    ):
        raise ValueError("planning cell-attempted state drift")


def run_cached(args: argparse.Namespace) -> dict[str, Any]:
    run_root = Path(args.run_root).resolve()
    out_root = Path(args.out_root).resolve()
    if run_root.name != RUN_ID:
        raise ValueError(f"run root basename must be {RUN_ID}: {run_root}")
    if out_root.exists():
        raise FileExistsError(f"planning output already exists: {out_root}")
    if args.cell not in CELL_SPECS or args.mode not in {"smoke", "full"}:
        raise ValueError(f"unsupported cell/mode: {args.cell}/{args.mode}")
    if not HEX40.fullmatch(args.jepa_commit):
        raise ValueError(f"invalid experiment commit: {args.jepa_commit}")
    if args.stable_commit != STABLE_WORLDMODEL_COMMIT:
        raise ValueError("stable-worldmodel commit drift")
    baseline_sha = _require_sha256(args.baseline_manifest_sha256, "baseline manifest SHA256")
    if baseline_sha != BASELINE_MANIFEST_SHA256:
        raise ValueError("baseline-manifest hash differs from the preregistered implementation")
    prediction_sha = _require_sha256(args.prediction_sha256, "prediction artifact SHA256")

    common, runner = load_frozen_modules(args)
    fit_summary_path = run_root / "probes" / "probe_fit_summary.json"
    fit_summary = json.loads(fit_summary_path.read_text(encoding="utf-8"))
    if fit_summary.get("n_cache_artifacts") != 30:
        raise ValueError("fit summary cache count drift")
    if fit_summary.get("source_sha256", {}).get("baseline_manifest") != baseline_sha:
        raise ValueError("fit summary baseline-manifest binding drift")
    _validate_dataset_unit_attestation(fit_summary.get("dataset_unit_attestation"))
    _validate_planning_ordering_state(
        run_root,
        mode=args.mode,
        cell=args.cell,
        baseline_sha256=baseline_sha,
        prediction_sha256=prediction_sha,
        fit_summary=fit_summary,
    )
    lookup = _record_lookup(fit_summary)
    prediction_path = run_root / "probes" / "prospective_gate_predictions.json"
    if not prediction_path.is_file() or sha256_file(prediction_path) != prediction_sha:
        raise ValueError("immutable prospective prediction artifact is absent or hash-mismatched")
    published_predictions = json.loads(prediction_path.read_text(encoding="utf-8"))
    recomputed_predictions = recompute_prospective_predictions(run_root, fit_summary)
    if published_predictions != recomputed_predictions:
        raise ValueError("published prospective predictions differ from the sealed caches")

    cell_spec = CELL_SPECS[args.cell]
    family = str(cell_spec["family"])
    family_spec = FAMILY_SPECS[family]
    train_seeds = [42] if args.mode == "smoke" else list(TRAIN_SEEDS)
    eval_seeds = [42] if args.mode == "smoke" else list(EVAL_SEEDS)
    num_eval = 2 if args.mode == "smoke" else 50
    smoke_manifest = run_root / "probes" / "smoke_eval_seed42.json"
    eval_manifest = str(smoke_manifest) if args.mode == "smoke" else ""
    if args.mode == "smoke" and not smoke_manifest.is_file():
        raise FileNotFoundError(smoke_manifest)

    manifest = common.load_manifest(args.task_manifest)
    task_spec = manifest["tasks"][TASK]
    expected_dataset = Path(task_spec["dataset"]).resolve()
    expected_models = {f"stock_seed{seed}" for seed in TRAIN_SEEDS}
    if set(task_spec["models"]) != expected_models:
        raise ValueError("stock task-manifest model set drift")

    context: dict[str, Any] = {}
    loaded: dict[tuple[int, int], dict[str, Any]] = {}
    runtime_checkpoint_hashes: dict[int, str] = {}
    original_parse_model_name = common.parse_model_name
    original_import_budget_runner = common.import_budget_runner
    original_linear_fit = common.fit_probe_bundles
    original_mlp_fit = runner.fit_mlp_probes

    def tracked_parse_model_name(model_name: str) -> tuple[str, int]:
        match = STOCK_MODEL_RE.fullmatch(model_name)
        if match is None:
            raise ValueError(f"planning requested a non-stock model: {model_name!r}")
        train_seed = int(match.group("seed"))
        context.clear()
        context.update({"arm": ARM, "train_seed": train_seed, "model_name": model_name})
        return ARM, train_seed

    def tracked_import_budget_runner(path: Path) -> Any:
        base = original_import_budget_runner(path)
        original_prepare_common = base.prepare_common

        def tracked_prepare_common(
            task: str,
            run: str,
            epoch: int,
            device: str,
            requested_num_eval: int,
            eval_seed: int,
            requested_manifest: str | None = None,
        ) -> tuple[dict[str, Any], dict[str, Any]]:
            if not context or int(context["train_seed"]) not in train_seeds:
                raise RuntimeError("prepare_common called without a bound canonical stock seed")
            if (
                task != TASK
                or epoch != 10
                or requested_num_eval != num_eval
                or int(eval_seed) not in eval_seeds
            ):
                raise ValueError("frozen planning preparation arguments drifted")
            if args.mode == "full" and requested_manifest:
                raise ValueError("full run must not use an evaluation manifest")
            if args.mode == "smoke" and Path(str(requested_manifest)).resolve() != smoke_manifest.resolve():
                raise ValueError("smoke did not use the cache-derived evaluation manifest")
            run_spec, prepared = original_prepare_common(
                task, run, epoch, device, requested_num_eval, eval_seed, requested_manifest
            )
            train_seed = int(context["train_seed"])
            expected_checkpoint = Path(task_spec["models"][f"stock_seed{train_seed}"]).resolve()
            actual_checkpoint = Path(run_spec["checkpoint"]).resolve()
            if actual_checkpoint != expected_checkpoint:
                raise ValueError(
                    f"prepare_common checkpoint drift: {actual_checkpoint} != {expected_checkpoint}"
                )
            if Path(run_spec["dataset"]).resolve() != expected_dataset:
                raise ValueError("prepare_common Cube dataset path drift")
            _assert_plain_stock_model(prepared["model"])
            if args.mode == "full":
                _assert_prepared_eval_identity(prepared, train_seed, int(eval_seed))
            prepared["model"] = _make_stock_full_a192_shim(prepared["model"])
            context.update(
                {
                    "eval_seed": int(eval_seed),
                    "prepared": prepared,
                    "run": run,
                    "run_spec": run_spec,
                }
            )
            return run_spec, prepared

        base.prepare_common = tracked_prepare_common
        return base

    def cache_loader(
        base: Any,
        task: str,
        model: Any,
        dataset: Path,
        *,
        budgets: Iterable[int],
        exclude_episodes: set[int],
        n_train: int,
        seed: int,
        batch_size: int,
        device: str,
        **kwargs: Any,
    ) -> dict[int, Any]:
        if not context or "prepared" not in context:
            raise RuntimeError("cache load called outside a tracked planning pair")
        train_seed = int(context["train_seed"])
        eval_seed = int(context["eval_seed"])
        if seed != PROBE_SEED + eval_seed:
            raise ValueError("effective probe seed drift")
        if (
            task != TASK
            or tuple(int(x) for x in budgets) != BUDGETS
            or n_train != PROBE_TRAIN_SAMPLES
            or batch_size != PROBE_BATCH_SIZE
        ):
            raise ValueError("frozen cache-loader arguments drifted")
        if not bool(getattr(model, "_stock_full_planning_shim", False)):
            raise TypeError("planning cache loader did not receive the A192 stock compatibility shim")
        if family == "s0":
            if kwargs.get("probe_mode") != "linear":
                raise ValueError("S0 cache loader received non-linear probe_mode")
            if kwargs.get("target_variant") != family_spec["target_variant"]:
                raise ValueError("S0 cache target variant drift")
        elif kwargs.get("epochs") != MLP_EPOCHS:
            raise ValueError("S1 cache loader received a non-frozen epoch count")

        key = (family, train_seed, eval_seed)
        record = lookup.get(key)
        if record is None:
            raise KeyError(f"fit summary has no cache record {key}")
        path = run_root / record["path"]
        metadata, arrays = load_npz_cache(path, expected_sha256=record["sha256"])
        _validate_loaded_metadata(metadata, family=family, train_seed=train_seed, eval_seed=eval_seed)
        dataset_path = Path(dataset).resolve()
        if dataset_path != expected_dataset or dataset_path != Path(metadata["dataset_path"]).resolve():
            raise ValueError("planning dataset differs from the cache/manifest dataset")
        dataset_stat = dataset_path.stat()
        if metadata["dataset_stat"] != {
            "size": dataset_stat.st_size,
            "mtime_ns": dataset_stat.st_mtime_ns,
        }:
            raise ValueError("planning dataset identity changed after cache fitting")
        expected_run = Path(metadata["checkpoint_path"]).parent.name
        if context.get("run") != expected_run:
            raise ValueError("planning checkpoint run differs from the cache")
        actual_checkpoint = Path(context["run_spec"]["checkpoint"]).resolve()
        if actual_checkpoint != Path(metadata["checkpoint_path"]).resolve():
            raise ValueError("planning checkpoint path differs from the cache")
        if train_seed not in runtime_checkpoint_hashes:
            _validate_checkpoint(actual_checkpoint, train_seed)
            runtime_checkpoint_hashes[train_seed] = CHECKPOINT_SHA256[train_seed]

        prepared = context["prepared"]
        runtime_excluded = {int(x) for x in exclude_episodes}
        cached_excluded = {int(x) for x in metadata["exclude_episodes"]}
        if args.mode == "full":
            if runtime_excluded != cached_excluded:
                raise ValueError("full runtime exclusion set differs from cached exclusion set")
            comparisons = {
                "planning_eval_indices": prepared["eval_indices"],
                "planning_eval_episodes": prepared["eval_episodes"],
                "planning_eval_start_idx": prepared["eval_start_idx"],
            }
            for name, runtime_value in comparisons.items():
                if not _same_array(runtime_value, arrays[name]):
                    raise ValueError(f"full runtime evaluation array differs from cache: {name}")
        else:
            if not runtime_excluded or not runtime_excluded.issubset(cached_excluded):
                raise ValueError("smoke evaluation episodes are not a subset of full exclusions")
            smoke_payload = json.loads(smoke_manifest.read_text(encoding="utf-8"))
            if set(smoke_payload["planning_eval_episode_ids"]) != cached_excluded:
                raise ValueError("smoke manifest contents differ from the cache exclusion set")
        pair = (train_seed, eval_seed)
        if pair in loaded:
            raise RuntimeError(f"cache loaded more than once for planning pair {pair}")
        loaded[pair] = {
            "path": record["path"],
            "sha256": record["sha256"],
            "exclude_episodes_sha256": metadata["exclude_episodes_sha256"],
        }
        if family == "s1":
            return _reconstruct_mlp(runner, metadata, arrays, model=model, device=device)
        return _reconstruct_linear(common, metadata, arrays)

    def forbidden_fit(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("unselected/original probe fit route was invoked during cached planning")

    common.parse_model_name = tracked_parse_model_name
    common.import_budget_runner = tracked_import_budget_runner
    if family == "s1":
        runner.fit_mlp_probes = cache_loader
        common.fit_probe_bundles = forbidden_fit
    else:
        common.fit_probe_bundles = cache_loader
        runner.fit_mlp_probes = forbidden_fit

    runner_args = argparse.Namespace(
        task_manifest=str(Path(args.task_manifest).resolve()),
        budget_runner=str(Path(args.budget_runner).resolve()),
        out_root=str(out_root),
        tasks=[TASK],
        arms=[ARM],
        train_seeds=train_seeds,
        eval_seeds=eval_seeds,
        num_eval=num_eval,
        methods=["fixed_full"],
        budgets=list(BUDGETS),
        device=args.device,
        eval_episode_manifest=eval_manifest,
        probe_train_samples=PROBE_TRAIN_SAMPLES,
        probe_batch_size=PROBE_BATCH_SIZE,
        probe_seed=PROBE_SEED,
        probe_budget=192,
        hybrid_lambda=float(cell_spec["hybrid_lambda"]),
        beta_tr=0.0,
        probe_mode="linear",
        target_variant=family_spec["target_variant"],
        probe_family=family_spec["probe_family"],
        mlp_epochs=MLP_EPOCHS,
    )
    try:
        runner.run_all(runner_args)
    finally:
        common.parse_model_name = original_parse_model_name
        common.import_budget_runner = original_import_budget_runner
        common.fit_probe_bundles = original_linear_fit
        runner.fit_mlp_probes = original_mlp_fit

    expected_pairs = {(ts, es) for ts in train_seeds for es in eval_seeds}
    if set(loaded) != expected_pairs:
        raise RuntimeError(
            f"expected one cache load for each planning pair: expected={expected_pairs} actual={set(loaded)}"
        )
    metadata_path = out_root / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "run_id": RUN_ID,
            "cell": args.cell,
            "mode": args.mode,
            "probe_cache_family": family,
            "precomputed_probes": True,
            "planning_probe_fit_calls": 0,
            "planning_probe_cache_load_calls": len(loaded),
            "probe_artifact_sha256": {
                f"train_seed{ts}_eval_seed{es}": loaded[(ts, es)]["sha256"]
                for ts, es in sorted(loaded)
            },
            "probe_cache_paths": {
                f"train_seed{ts}_eval_seed{es}": loaded[(ts, es)]["path"]
                for ts, es in sorted(loaded)
            },
            "cache_exclusion_basis": "full_n50",
            "runtime_smoke_exclusion_subset": args.mode == "smoke",
            "probe_batch_size": PROBE_BATCH_SIZE,
            "probe_seed": PROBE_SEED,
            "target_variant": family_spec["target_variant"],
            "checkpoint_epoch": 10,
            "checkpoint_sha256": {str(seed): CHECKPOINT_SHA256[seed] for seed in train_seeds},
            "checkpoint_config_sha256": {str(seed): CONFIG_SHA256[seed] for seed in train_seeds},
            "probe_runner_sha256": PROBE_RUNNER_SHA256,
            "probe_common_sha256": PROBE_COMMON_SHA256,
            "budget_runner_sha256": BUDGET_RUNNER_SHA256,
            "task_manifest_sha256": TASK_MANIFEST_SHA256,
            "baseline_manifest_sha256": baseline_sha,
            "prospective_gate_predictions_sha256": prediction_sha,
            "prospective_gate_cell_prediction": published_predictions["cells"][args.cell],
            "stock_full_a192_planning_shim": True,
            "stock_full_a192_lower_budget_hard_reject": True,
            "jepa_wms_commit": args.jepa_commit,
            "jepa_wms_base_commit": JEPA_WMS_BASE_COMMIT,
            "stable_worldmodel_commit": STABLE_WORLDMODEL_COMMIT,
            "worktree_clean": True,
            "smoke_eval_manifest_sha256": sha256_file(smoke_manifest) if args.mode == "smoke" else "",
        }
    )
    atomic_write_json(metadata_path, metadata, replace=True)
    return metadata


def add_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--probe-runner", required=True)
    parser.add_argument("--probe-common", required=True)
    parser.add_argument("--budget-runner", required=True)
    parser.add_argument("--task-manifest", required=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    imp = subparsers.add_parser(
        "import-caches",
        help="import and hash-verify the sealed caches of the source run (no refit)",
    )
    imp.add_argument("--source-root", required=True)
    imp.add_argument("--run-root", required=True)

    publish = subparsers.add_parser(
        "publish-predictions",
        help="atomically publish the prospective FM-4 predictions before planning",
    )
    publish.add_argument("--run-root", required=True)

    cached = subparsers.add_parser(
        "run-cached", help="run one frozen planning cell using only sealed caches"
    )
    add_source_arguments(cached)
    cached.add_argument("--run-root", required=True)
    cached.add_argument("--out-root", required=True)
    cached.add_argument("--cell", required=True, choices=sorted(CELL_SPECS))
    cached.add_argument("--mode", required=True, choices=("smoke", "full"))
    cached.add_argument("--jepa-commit", required=True)
    cached.add_argument("--stable-commit", required=True)
    cached.add_argument("--baseline-manifest-sha256", required=True)
    cached.add_argument("--prediction-sha256", required=True)
    cached.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "import-caches":
        import_caches(args)
    elif args.command == "publish-predictions":
        publish_prospective_predictions(args)
    elif args.command == "run-cached":
        run_cached(args)
    else:  # pragma: no cover - argparse enforces this
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
