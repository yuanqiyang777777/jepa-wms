#!/usr/bin/env python3
"""Fit immutable stock-LeWM PushT probe caches and run from them.

This is the only probe implementation for the preregistered 2026-07-14 stock
PushT prospective-gate experiment.  Its result-bearing operations are:

``fit-cache``
    Reproduce the frozen S0/S1 fits for all 15 train/evaluation pairs and write
    30 immutable, numeric-only ``.npz`` artifacts.

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
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np


RUN_ID = "stock_pusht_probe_20260714_retry1"
TASK = "pusht"
ARM = "stock"
TRAIN_SEEDS = (42, 43, 44)
EVAL_SEEDS = (42, 43, 44, 45, 46)
BUDGETS = (24, 48, 96, 192)
PROBE_SEED = 1703
PROBE_TRAIN_SAMPLES = 3000
PROBE_BATCH_SIZE = 128
MLP_EPOCHS = 200
# PushT has NO preregistered per-variable success radius: the environment's
# success condition couples agent and block positions in one joint norm and an
# angle threshold, while the probe target mixes pixel positions with angle
# sin/cos.  The prospective FM-4 artifact is therefore frozen as not evaluable
# and publishes native-unit RMSE only.  No scalar radius or conversion may be
# introduced after fits or outcomes are seen.
GATE_STATUS = "NOT_EVALUABLE_NO_PER_VARIABLE_RADIUS"
TARGET_NAMES = ["agent_location", "block_location", "block_angle_sincos"]
TARGET_BLOCKS = [[0, 2], [2, 4], [4, 6]]
TARGET_DIM = 6
TARGET_NATIVE_UNITS = {
    "agent_location": "PushT environment pixels (512-frame)",
    "block_location": "PushT environment pixels (512-frame)",
    "block_angle_sincos": "dimensionless sin/cos",
}

PROBE_RUNNER_SHA256 = "564d7469840a38bf6015149d9d61191694beef9e0f5ed6610c5db9934b7dbdc8"
PROBE_COMMON_SHA256 = "9c57a82343277e3378f24df8ca38e5f9a4dcc53a4fead909e2a6126001d1a89c"
BUDGET_RUNNER_SHA256 = "cefd2b267601377aa1201a5302828807f4189ca11f123c1128746938f51dfd8f"
TASK_MANIFEST_SHA256 = "7673dc4389b5cb0e9b4eb2082b3fbb1152c8307b790380eefdeee3c451c37ce3"
BASELINE_MANIFEST_SHA256 = "9017654b0b159521dbd93e274e8dbd1e37cfcdfc3aa66eaaee59530883a328e5"
BASELINE_EVAL_ARRAYS_SHA256 = "e6dfc3ae7debeaad80b29c4171121f14fbee2bbfb31d5d12b97427b9f330752e"
BASELINE_PAIR_EVAL_ARRAYS_SHA256 = {
    (42, 42): "a3556b665c62f293eab03db6c68f54778397d99d14b6d17ad664a3a592112263",
    (42, 43): "2febe1b604fbaecdb9d26ebedbc61579e988f93605e0515a4fd066cc1da38ed8",
    (42, 44): "5bf96ab7d91bf9eff0f76bd59d59cd4101c0b08203636c317bc3474083c66756",
    (42, 45): "5a75b4bd3e1403933c67d4227a9c9180fefc55896c59e43ed20343142485e166",
    (42, 46): "0d60b66dfc88aab3548f7c56233c9f59090967da3c87eb769e019b7ba1d5e854",
    (43, 42): "aa9a07e9e51e9ac1c34fa05334f83010d1f2e65cdd9d1c3a030947a8237cfabe",
    (43, 43): "61b55978c19ddd6f261ad64975a7e8327ff7933ed8498f9cb3f56bae463d5c3f",
    (43, 44): "1f1fb6d05373d15222fef304f1ad7d365f1aa63b1adbd610dc0cc87f0e6d9d46",
    (43, 45): "c922164b2e19cd8c7ca22c791d7894877280d3a575557b0577e1cba6e0be447c",
    (43, 46): "bc166cc0d10a18215d341b96e4208f9b4275eb966ac0621b1d292afd34a6eee6",
    (44, 42): "92c299d096fcabe2043554c904ded1935c93549b8d4a4cbca40639cf435d6d88",
    (44, 43): "3deb80ea039324a237f91fb15bf40bc4623499b3c411a0e4d8710dc4e7520dfd",
    (44, 44): "ef5136b9929aa21639ee618baf0d8930819b5c69f2a6c36985d5175decbea47b",
    (44, 45): "70f08bb9a31620c989766774f80378d541a2ffd9e4ebc386640c3267a222842b",
    (44, 46): "15900dce22cda8fa13d0eeb51618968ca675b212e0ee76d1c5c0643813eaf0f6",
}
JEPA_WMS_BASE_COMMIT = "21389db737ddf1bcbc8dc695d27ca8f7a8b77cb0"
STABLE_WORLDMODEL_COMMIT = "314201d0347baf61304c7558a98463c6a01080f0"
CHECKPOINT_SHA256 = {
    42: "e94142169195edfb2fc7691a424d36b63d036722a5402aefdfa6ce1bd361b1d4",
    43: "28c3a43e8604856cf92c50b18163b03ba4c56dc4f2be1e7ccf17b01ad40267fd",
    44: "90d78f11ec51271e81ba140a55303560c6dc31fe3f919b3b47f26ec15c34a98d",
}
CONFIG_SHA256 = {
    42: "54fa420fb93bb253065408a976513c9b5ae4ef45ab8c60eeb12d6b324021b566",
    43: "c385ab390e84d5f28e75e47b198047912b7b10244a5e9e84300f41d9a63ff2a1",
    44: "1b1bee67c98cc38950db6d53918f8ed71a809ae9b3f0af86ff08aa9a637262fb",
}

FAMILY_SPECS: dict[str, dict[str, str]] = {
    "s0": {"probe_family": "linear", "target_variant": "success"},
    "s1": {"probe_family": "mlp", "target_variant": "success"},
}

CELL_SPECS: dict[str, dict[str, Any]] = {
    "p1_linear_pure": {"family": "s0", "hybrid_lambda": 1.0, "mode": "pure"},
    "p2_linear_hybrid_l05": {"family": "s0", "hybrid_lambda": 0.5, "mode": "hybrid"},
    "p3_mlp_hybrid_l05": {"family": "s1", "hybrid_lambda": 0.5, "mode": "hybrid"},
    "p4_mlp_pure": {"family": "s1", "hybrid_lambda": 1.0, "mode": "pure"},
}

SCHEMA_VERSION = "stock_pusht_probe_cache_v1"
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")
STOCK_MODEL_RE = re.compile(r"stock_seed(?P<seed>42|43|44)")

DATASET_UNIT_CONTRACT = {
    "coordinate_frame": "PushT environment pixels (512-frame); angle stored in radians",
    "state_field": "state",
    "state_columns": [
        "agent_x",
        "agent_y",
        "block_x",
        "block_y",
        "block_angle_rad",
        "agent_vel_x",
        "agent_vel_y",
    ],
    "proprio_field": "proprio",
    "episode_field": "episode_idx",
    "state_proprio_atol": 1.0e-6,
    "position_envelope_px": [-512.0, 1024.0],
    "angle_range_rad": [0.0, 6.283185307179587],
    "angle_atol_rad": 1.0e-4,
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
    """Prove the PushT state columns carry the units the probe target assumes.

    The probe target reads ``state`` columns 0-1 (agent xy), 2-3 (block xy) and
    the sin/cos of column 4 (block angle).  This attestation verifies the
    column semantics against the redundant ``proprio`` field (agent xy and
    velocity, written by the same environment step), the 512-frame pixel
    envelope, and the ``angle % 2*pi`` storage convention — without inventing
    any per-variable success radius (gate is NOT_EVALUABLE by preregistration).
    """
    import h5py

    required = {"state", "proprio", "episode_idx", "ep_offset", "ep_len"}
    chunk = 250_000
    maximum_agent_delta = 0.0
    maximum_velocity_delta = 0.0
    minimum_position = math.inf
    maximum_position = -math.inf
    minimum_angle = math.inf
    maximum_angle = -math.inf
    with h5py.File(dataset, "r") as handle:
        missing = sorted(required.difference(handle.keys()))
        if missing:
            raise ValueError(f"PushT dataset lacks unit-contract fields: {missing}")
        state = handle["state"]
        proprio = handle["proprio"]
        if state.ndim != 2 or state.shape[1] != 7:
            raise ValueError(f"PushT state must have shape (N,7), got {state.shape}")
        if proprio.ndim != 2 or proprio.shape[1] != 4:
            raise ValueError(f"PushT proprio must have shape (N,4), got {proprio.shape}")
        n_rows = int(state.shape[0])
        if (
            int(proprio.shape[0]) != n_rows
            or int(handle["episode_idx"].shape[0]) != n_rows
        ):
            raise ValueError("PushT unit-contract fields have different row counts")
        for start in range(0, n_rows, chunk):
            stop = min(start + chunk, n_rows)
            st = np.asarray(state[start:stop], dtype=np.float64)
            pr = np.asarray(proprio[start:stop], dtype=np.float64)
            if not (np.isfinite(st).all() and np.isfinite(pr).all()):
                raise ValueError("PushT state/proprio fields contain NaN/Inf")
            maximum_agent_delta = max(
                maximum_agent_delta, float(np.max(np.abs(st[:, :2] - pr[:, :2])))
            )
            maximum_velocity_delta = max(
                maximum_velocity_delta, float(np.max(np.abs(st[:, 5:7] - pr[:, 2:4])))
            )
            minimum_position = min(minimum_position, float(st[:, :4].min()))
            maximum_position = max(maximum_position, float(st[:, :4].max()))
            minimum_angle = min(minimum_angle, float(st[:, 4].min()))
            maximum_angle = max(maximum_angle, float(st[:, 4].max()))
    atol = DATASET_UNIT_CONTRACT["state_proprio_atol"]
    if maximum_agent_delta > atol:
        raise ValueError(
            "PushT state agent xy and proprio agent xy differ: "
            f"max_abs_delta={maximum_agent_delta}"
        )
    if maximum_velocity_delta > atol:
        raise ValueError(
            "PushT state velocity and proprio velocity differ: "
            f"max_abs_delta={maximum_velocity_delta}"
        )
    envelope_low, envelope_high = DATASET_UNIT_CONTRACT["position_envelope_px"]
    if minimum_position < envelope_low or maximum_position > envelope_high:
        raise ValueError(
            "PushT position columns escape the 512-frame pixel envelope "
            f"[{envelope_low}, {envelope_high}]: observed "
            f"[{minimum_position}, {maximum_position}]"
        )
    angle_low, angle_high = DATASET_UNIT_CONTRACT["angle_range_rad"]
    angle_atol = DATASET_UNIT_CONTRACT["angle_atol_rad"]
    if minimum_angle < angle_low - angle_atol or maximum_angle > angle_high + angle_atol:
        raise ValueError(
            "PushT block angle is not stored as radians mod 2*pi: observed "
            f"[{minimum_angle}, {maximum_angle}]"
        )
    payload = {
        **DATASET_UNIT_CONTRACT,
        "n_rows_checked": n_rows,
        "position_min_px": minimum_position,
        "position_max_px": maximum_position,
        "angle_min_rad": minimum_angle,
        "angle_max_rad": maximum_angle,
        "max_abs_state_agent_minus_proprio_px": maximum_agent_delta,
        "max_abs_state_velocity_minus_proprio": maximum_velocity_delta,
        "status": "PASS",
    }
    payload["attestation_sha256"] = sha256_json(payload)
    return payload


def _validate_dataset_unit_attestation(payload: Any) -> None:
    if not isinstance(payload, dict):
        raise ValueError("cache lacks the PushT dataset-unit attestation")
    actual = dict(payload)
    recorded = actual.pop("attestation_sha256", None)
    if recorded != sha256_json(actual):
        raise ValueError("PushT dataset-unit attestation hash drift")
    for name, expected in DATASET_UNIT_CONTRACT.items():
        if actual.get(name) != expected:
            raise ValueError(f"PushT unit-contract field drift: {name}")
    if actual.get("status") != "PASS" or int(actual.get("n_rows_checked", 0)) <= 0:
        raise ValueError("PushT dataset-unit attestation is not a positive PASS")
    atol = float(DATASET_UNIT_CONTRACT["state_proprio_atol"])
    if float(actual["max_abs_state_agent_minus_proprio_px"]) > atol:
        raise ValueError("PushT agent-column attestation exceeds tolerance")
    if float(actual["max_abs_state_velocity_minus_proprio"]) > atol:
        raise ValueError("PushT velocity-column attestation exceeds tolerance")
    envelope_low, envelope_high = DATASET_UNIT_CONTRACT["position_envelope_px"]
    if (
        float(actual["position_min_px"]) < float(envelope_low)
        or float(actual["position_max_px"]) > float(envelope_high)
    ):
        raise ValueError("PushT position-envelope attestation exceeds tolerance")
    angle_low, angle_high = DATASET_UNIT_CONTRACT["angle_range_rad"]
    angle_atol = float(DATASET_UNIT_CONTRACT["angle_atol_rad"])
    if (
        float(actual["angle_min_rad"]) < float(angle_low) - angle_atol
        or float(actual["angle_max_rad"]) > float(angle_high) + angle_atol
    ):
        raise ValueError("PushT angle-range attestation exceeds tolerance")


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
        n_rows = int(handle["episode_idx"].shape[0])
        if int(indices.min()) < 0 or int(indices.max()) >= n_rows:
            raise ValueError("probe sample index is outside the PushT dataset")
        sample_episodes = np.asarray(handle["episode_idx"][indices], dtype=np.int64)
    overlap = sorted(set(sample_episodes.tolist()).intersection(exclude_episodes))
    if overlap:
        raise ValueError(f"probe fit leaked planning-evaluation episodes: {overlap[:10]}")


def _assert_pusht_target_contract(budget_meta: dict[str, Any]) -> None:
    for budget in BUDGETS:
        metadata = budget_meta[str(budget)]
        if (
            int(metadata.get("target_dim", -1)) != TARGET_DIM
            or list(metadata.get("target_names", [])) != TARGET_NAMES
            or [list(block) for block in metadata.get("target_blocks", [])] != TARGET_BLOCKS
        ):
            raise ValueError(
                "PushT probe must decode agent_location xy + block_location xy + "
                "block_angle_sincos (6 dims, blocks (0,2),(2,4),(4,6)): "
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
        "gate_status": GATE_STATUS,
        "target_native_units": TARGET_NATIVE_UNITS,
        "physical_coordinate_unit": (
            "native units per block: positions in PushT 512-frame pixels, "
            "block_angle_sincos dimensionless"
        ),
        "dataset_unit_attestation": dataset_unit_attestation,
        "mlp_epochs": MLP_EPOCHS if family == "s1" else None,
        "created_unix": time.time(),
    }


def _cache_relpath(family: str, train_seed: int, eval_seed: int) -> Path:
    return Path("probe_caches") / family / f"ts{train_seed}_es{eval_seed}.npz"


def _validate_checkpoint(checkpoint: Path, train_seed: int) -> None:
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    expected_name = f"stock_lewm_pusht_matched3_seed{train_seed}_20260615"
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


def fit_cache(args: argparse.Namespace) -> dict[str, Any]:
    run_root = Path(args.run_root).resolve()
    if run_root.name != RUN_ID:
        raise ValueError(f"run root basename must be {RUN_ID}: {run_root}")
    if not run_root.is_dir():
        raise FileNotFoundError(run_root)
    cache_root = run_root / "probe_caches"
    probes_root = run_root / "probes"
    if cache_root.exists() or probes_root.exists():
        raise FileExistsError("cache/probe output already exists; immutable retry requires a new run id")
    cache_root.mkdir()
    probes_root.mkdir()

    common, runner = load_frozen_modules(args)
    base = common.import_budget_runner(Path(args.budget_runner))
    manifest = common.load_manifest(args.task_manifest)
    task_spec = manifest["tasks"][TASK]
    dataset = Path(task_spec["dataset"])
    if not dataset.is_file():
        raise FileNotFoundError(dataset)
    dataset_unit_attestation = _dataset_unit_attestation(dataset)
    _validate_dataset_unit_attestation(dataset_unit_attestation)
    expected_models = {f"stock_seed{seed}" for seed in TRAIN_SEEDS}
    if set(task_spec["models"]) != expected_models:
        raise ValueError(f"canonical manifest model set drift: {sorted(task_spec['models'])}")

    records: list[dict[str, Any]] = []
    all_gate_rows: list[dict[str, Any]] = []
    all_eval_rows: list[list[int]] = []
    started = time.perf_counter()
    original_encode_latents = common.encode_latents

    def stock_aware_encode(
        imported_base: Any,
        model: Any,
        h5_path: Path,
        indices: np.ndarray,
        *,
        batch_size: int,
        device: str,
    ) -> np.ndarray:
        return _encode_stock_aware_latents(
            common,
            imported_base,
            model,
            h5_path,
            indices,
            batch_size=batch_size,
            device=device,
        )

    common.encode_latents = stock_aware_encode
    for train_seed in TRAIN_SEEDS:
        model_name = f"stock_seed{train_seed}"
        checkpoint = Path(task_spec["models"][model_name])
        _validate_checkpoint(checkpoint, train_seed)
        run_name = common.checkpoint_to_run(str(checkpoint))
        for eval_seed in EVAL_SEEDS:
            print(f"[cache-pair] train={train_seed} eval={eval_seed}", flush=True)
            run_spec, prepared = base.prepare_common(
                TASK,
                run_name,
                10,
                args.device,
                50,
                eval_seed,
                None,
            )
            actual_checkpoint = Path(run_spec["checkpoint"]).resolve()
            if actual_checkpoint != checkpoint.resolve():
                raise ValueError(
                    "prepare_common loaded a different checkpoint than the hash-verified manifest "
                    f"path: {actual_checkpoint} != {checkpoint.resolve()}"
                )
            _assert_plain_stock_model(prepared["model"])
            all_eval_rows.extend(
                _assert_prepared_eval_identity(prepared, train_seed, eval_seed)
            )
            exclude_episodes = {int(x) for x in prepared["eval_episodes"].tolist()}
            effective_seed = PROBE_SEED + eval_seed
            sample_indices = common.make_probe_indices(
                dataset,
                exclude_episodes=exclude_episodes,
                n_train=PROBE_TRAIN_SAMPLES,
                seed=effective_seed,
            )
            _assert_probe_partition(
                dataset,
                sample_indices,
                exclude_episodes,
                prepared["eval_episodes"],
            )
            val_mask = _val_mask(len(sample_indices), effective_seed)

            for family in ("s0", "s1"):
                spec = FAMILY_SPECS[family]
                if family == "s1":
                    probes, details, mlp_indices, mlp_val_mask = _fit_mlp_probes_with_metrics(
                        common,
                        runner,
                        base,
                        prepared["model"],
                        dataset,
                        exclude_episodes=exclude_episodes,
                        seed=effective_seed,
                        device=args.device,
                    )
                    if not np.array_equal(sample_indices, mlp_indices) or not np.array_equal(
                        val_mask, mlp_val_mask
                    ):
                        raise RuntimeError("MLP fit split drifted from frozen common split")
                    arrays, budget_meta = _mlp_cache_payload(
                        details, prepared, sample_indices, val_mask
                    )
                else:
                    probes = common.fit_probe_bundles(
                        base,
                        TASK,
                        prepared["model"],
                        dataset,
                        budgets=BUDGETS,
                        exclude_episodes=exclude_episodes,
                        n_train=PROBE_TRAIN_SAMPLES,
                        seed=effective_seed,
                        batch_size=PROBE_BATCH_SIZE,
                        device=args.device,
                        probe_mode="linear",
                        target_variant=spec["target_variant"],
                    )
                    arrays, budget_meta = _linear_cache_payload(
                        probes, prepared, sample_indices, val_mask
                    )
                _assert_pusht_target_contract(budget_meta)
                metadata = _base_cache_metadata(
                    family=family,
                    train_seed=train_seed,
                    eval_seed=eval_seed,
                    checkpoint=checkpoint,
                    dataset=dataset,
                    exclude_episodes=exclude_episodes,
                    budget_meta=budget_meta,
                    dataset_unit_attestation=dataset_unit_attestation,
                )
                metadata["gate_rows"] = _gate_rows(
                    family, train_seed, eval_seed, budget_meta, arrays
                )
                relative = _cache_relpath(family, train_seed, eval_seed)
                destination = run_root / relative
                artifact_sha = atomic_write_npz(destination, metadata, arrays)
                loaded_meta, _ = load_npz_cache(destination, expected_sha256=artifact_sha)
                if loaded_meta["exclude_episodes_sha256"] != metadata["exclude_episodes_sha256"]:
                    raise RuntimeError(f"cache round-trip metadata mismatch: {destination}")
                record = {
                    "family": family,
                    "train_seed": train_seed,
                    "eval_seed": eval_seed,
                    "path": relative.as_posix(),
                    "sha256": artifact_sha,
                    "exclude_episodes_sha256": metadata["exclude_episodes_sha256"],
                    "checkpoint_sha256": metadata["checkpoint_sha256"],
                    "gate_rows": metadata["gate_rows"],
                }
                records.append(record)
                all_gate_rows.extend(metadata["gate_rows"])
                print(f"[cached] {relative} sha256={artifact_sha}", flush=True)

    common.encode_latents = original_encode_latents
    if len(all_eval_rows) != 750 or sha256_json(all_eval_rows) != BASELINE_EVAL_ARRAYS_SHA256:
        raise RuntimeError("global prepared-evaluation digest differs from true-stock baseline")
    expected_keys = {
        (family, train_seed, eval_seed)
        for family in FAMILY_SPECS
        for train_seed in TRAIN_SEEDS
        for eval_seed in EVAL_SEEDS
    }
    actual_keys = {
        (row["family"], row["train_seed"], row["eval_seed"]) for row in records
    }
    if actual_keys != expected_keys or len(records) != 30:
        raise RuntimeError(f"cache matrix incomplete: {len(records)} records")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "run_id": RUN_ID,
        "n_cache_artifacts": len(records),
        "records": sorted(
            records, key=lambda row: (row["family"], row["train_seed"], row["eval_seed"])
        ),
        "cache_artifact_sha256": {
            row["path"]: row["sha256"] for row in sorted(records, key=lambda x: x["path"])
        },
        "prospective_gate_input": {
            "gate_status": GATE_STATUS,
            "aggregation": (
                "per-family per-coordinate and per-block A192 RMSE in native units; "
                "no threshold, no PASS/FAIL prediction"
            ),
            "target_native_units": TARGET_NATIVE_UNITS,
            "publication_policy": "validate-cache publishes the immutable prediction artifact before planning",
        },
        "source_sha256": {
            "probe_runner": PROBE_RUNNER_SHA256,
            "probe_common": PROBE_COMMON_SHA256,
            "budget_runner": BUDGET_RUNNER_SHA256,
            "task_manifest": TASK_MANIFEST_SHA256,
            "baseline_manifest": BASELINE_MANIFEST_SHA256,
            "baseline_eval_arrays": BASELINE_EVAL_ARRAYS_SHA256,
        },
        "checkpoint_sha256": {str(k): v for k, v in CHECKPOINT_SHA256.items()},
        "checkpoint_config_sha256": {str(k): v for k, v in CONFIG_SHA256.items()},
        "dataset_unit_attestation": dataset_unit_attestation,
        "elapsed_sec": time.perf_counter() - started,
    }
    atomic_write_json(probes_root / "probe_fit_summary.json", summary)
    smoke_record = next(
        row
        for row in records
        if row["family"] == "s0" and row["train_seed"] == 42 and row["eval_seed"] == 42
    )
    smoke_meta, _ = load_npz_cache(
        run_root / smoke_record["path"], expected_sha256=smoke_record["sha256"]
    )
    smoke_manifest = {
        "planning_eval_episode_ids": smoke_meta["exclude_episodes"],
        "source_cache_path": smoke_record["path"],
        "source_cache_sha256": smoke_record["sha256"],
        "purpose": "two-episode wiring smoke drawn only from the full-n50 exclusion set",
    }
    atomic_write_json(probes_root / "smoke_eval_seed42.json", smoke_manifest)
    print(json.dumps(summary["prospective_gate_input"], indent=2, allow_nan=False))
    return summary


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
    run_root: Path, summary: dict[str, Any]
) -> dict[str, Any]:
    """Verify all caches and recompute the frozen not-evaluable FM-4 artifact.

    PushT publishes per-coordinate and per-block A192 RMSE in native units for
    both families and makes NO PASS/FAIL prediction (gate_status
    NOT_EVALUABLE_NO_PER_VARIABLE_RADIUS).  Every planning cell runs
    regardless; introducing any radius or threshold here after fits are seen
    is a preregistration violation.
    """
    lookup = _record_lookup(summary)
    family_results: dict[str, Any] = {}
    for family in ("s0", "s1"):
        per_cache_rows: list[dict[str, Any]] = []
        per_coordinate_values: list[list[float]] = [[] for _ in range(TARGET_DIM)]
        for train_seed in TRAIN_SEEDS:
            for eval_seed in EVAL_SEEDS:
                record = lookup[(family, train_seed, eval_seed)]
                metadata, arrays = load_npz_cache(
                    run_root / record["path"], expected_sha256=record["sha256"]
                )
                _validate_loaded_metadata(
                    metadata,
                    family=family,
                    train_seed=train_seed,
                    eval_seed=eval_seed,
                )
                budget_meta = metadata["budget_metadata"]["192"]
                coordinates = arrays["b192_val_rmse_phys"].astype(float).tolist()
                if (
                    int(budget_meta["target_dim"]) != TARGET_DIM
                    or budget_meta["target_names"] != TARGET_NAMES
                    or len(coordinates) != TARGET_DIM
                    or not all(math.isfinite(x) for x in coordinates)
                ):
                    raise ValueError(
                        "PushT A192 RMSE coordinates invalid for "
                        f"family={family} train={train_seed} eval={eval_seed}"
                    )
                per_block = {
                    name: max(coordinates[start:stop])
                    for name, (start, stop) in zip(
                        TARGET_NAMES, [tuple(block) for block in TARGET_BLOCKS]
                    )
                }
                for index, value in enumerate(coordinates):
                    per_coordinate_values[index].append(value)
                per_cache_rows.append(
                    {
                        "train_seed": train_seed,
                        "eval_seed": eval_seed,
                        "a192_val_rmse_native_per_coordinate": coordinates,
                        "a192_val_rmse_native_per_block": per_block,
                    }
                )
        if len(per_cache_rows) != 15:
            raise ValueError(
                f"family {family} expected 15 A192 caches, found {len(per_cache_rows)}"
            )
        block_aggregates = {}
        for name, block in zip(TARGET_NAMES, TARGET_BLOCKS):
            block_values = [row["a192_val_rmse_native_per_block"][name] for row in per_cache_rows]
            block_aggregates[name] = {
                "native_unit": TARGET_NATIVE_UNITS[name],
                "n_caches": len(block_values),
                "rmse_native_mean": statistics.fmean(block_values),
                "rmse_native_min": min(block_values),
                "rmse_native_max": max(block_values),
            }
        family_results[family] = {
            "probe_family": FAMILY_SPECS[family]["probe_family"],
            "gate_status": GATE_STATUS,
            "aggregation": (
                "per-coordinate and per-block (max over block coordinates) A192 "
                "held-out RMSE in native units; descriptive only, no threshold"
            ),
            "target_names": list(TARGET_NAMES),
            "target_dim": TARGET_DIM,
            "target_blocks": [list(block) for block in TARGET_BLOCKS],
            "n_caches": 15,
            "per_cache": per_cache_rows,
            "per_coordinate_max": [max(values) for values in per_coordinate_values],
            "per_block": block_aggregates,
        }
    cell_predictions: dict[str, Any] = {}
    for cell, spec in CELL_SPECS.items():
        cell_predictions[cell] = {
            "family": str(spec["family"]),
            "mode": str(spec["mode"]),
            "gate_status": GATE_STATUS,
            "prediction": "NONE_GATE_NOT_EVALUABLE",
            "launch_policy": "RUN_REGARDLESS_OF_GATE_VERDICT",
        }
    return {
        "schema_version": "stock_pusht_prospective_gate_v1",
        "run_id": RUN_ID,
        "state": "PROSPECTIVE_PREDICTIONS_FROZEN",
        "task": TASK,
        "arm": ARM,
        "planning_budget": 192,
        "gate_status": GATE_STATUS,
        "success_definition": {
            "per_variable_radius": None,
            "source_commit": STABLE_WORLDMODEL_COMMIT,
            "source_file": "stable_worldmodel/envs/pusht/env.py",
            "condition": (
                "eval_state: success = ||goal_state[:4] - state[:4]|| < 20 px "
                "AND |delta angle| < pi/9; couples agent+block in one joint norm, "
                "so no per-variable radius exists and none may be invented post hoc"
            ),
        },
        "prediction_scoring": {
            "policy": (
                "not scoreable: no PASS/FAIL prediction is made; native-unit RMSE "
                "is published for transparency only"
            ),
            "realized_test": "two-sided exact McNemar raw alpha=0.05; H1 handled by the global Holm m=6 family",
        },
        "fit_summary_sha256": sha256_file(run_root / "probes" / "probe_fit_summary.json"),
        "baseline_manifest_sha256": BASELINE_MANIFEST_SHA256,
        "families": family_results,
        "cells": cell_predictions,
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
    if list((run_root / "smoke").glob("pusht_*")) or list(
        (run_root / "full").glob("pusht_*")
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
        "run_id": RUN_ID,
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
        "gate_status": GATE_STATUS,
        "target_native_units": TARGET_NATIVE_UNITS,
        "physical_coordinate_unit": (
            "native units per block: positions in PushT 512-frame pixels, "
            "block_angle_sincos dimensionless"
        ),
    }
    for name, wanted in expected.items():
        if metadata.get(name) != wanted:
            raise ValueError(
                f"cache metadata mismatch {name}: expected={wanted!r} actual={metadata.get(name)!r}"
            )
    _validate_dataset_unit_attestation(metadata.get("dataset_unit_attestation"))
    _assert_pusht_target_contract(metadata.get("budget_metadata", {}))


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
        (provenance / "pusht_dataset_sha256.txt").read_text(encoding="utf-8").strip(),
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
                raise ValueError("prepare_common PushT dataset path drift")
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

    fit = subparsers.add_parser("fit-cache", help="fit and seal the 30 immutable caches")
    add_source_arguments(fit)
    fit.add_argument("--run-root", required=True)
    fit.add_argument("--device", default="cuda")

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
    if args.command == "fit-cache":
        fit_cache(args)
    elif args.command == "publish-predictions":
        publish_prospective_predictions(args)
    elif args.command == "run-cached":
        run_cached(args)
    else:  # pragma: no cover - argparse enforces this
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
