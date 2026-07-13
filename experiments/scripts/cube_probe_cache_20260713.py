#!/usr/bin/env python3
"""Fit immutable Cube probe caches and run the frozen planner from them.

This is the only probe implementation for the preregistered 2026-07-13 Cube
C2 closure.  It deliberately exposes two modes only:

``fit-cache``
    Reproduce the frozen in-cell P0/P1/P2 fits for all 15 train/evaluation
    pairs and write 45 immutable, numeric-only ``.npz`` artifacts.

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


RUN_ID = "adadim_cube_probe_closure_20260713"
TASK = "cube"
ARM = "initialpartial"
TRAIN_SEEDS = (42, 43, 44)
EVAL_SEEDS = (42, 43, 44, 45, 46)
BUDGETS = (24, 48, 96, 192)
PROBE_SEED = 1703
PROBE_TRAIN_SAMPLES = 3000
PROBE_BATCH_SIZE = 128
MLP_EPOCHS = 200
SUCCESS_RADIUS_M = 0.04
P1_GATE_THRESHOLD_M = SUCCESS_RADIUS_M / 3.0

PROBE_RUNNER_SHA256 = "564d7469840a38bf6015149d9d61191694beef9e0f5ed6610c5db9934b7dbdc8"
PROBE_COMMON_SHA256 = "9c57a82343277e3378f24df8ca38e5f9a4dcc53a4fead909e2a6126001d1a89c"
BUDGET_RUNNER_SHA256 = "cefd2b267601377aa1201a5302828807f4189ca11f123c1128746938f51dfd8f"
TASK_MANIFEST_SHA256 = "74e723c932c0320c5bde2117d496ad4f0c2e6a14c0163e69486d9e5d2ccd8c2f"
JEPA_WMS_BASE_COMMIT = "7dbbf8277a3f4c9071ba9d1eb0e11f23c8657f4d"
STABLE_WORLDMODEL_COMMIT = "314201d0347baf61304c7558a98463c6a01080f0"
CHECKPOINT_SHA256 = {
    42: "81f2b3023f57299cb2d0504ec168bdf92d827d0d54b65a05718b4c9b492ba3e7",
    43: "0ccce2664439ac5a286bfcfd485f30d78c41e0d54de181de717e09657b510b0d",
    44: "c16a351e13b294c809ca6ebe4ac00278287ac5e0e97be68d0b5980851fe1600d",
}

FAMILY_SPECS: dict[str, dict[str, str]] = {
    "p0": {"probe_family": "linear", "target_variant": "success"},
    "p1": {"probe_family": "mlp", "target_variant": "success"},
    "p2": {"probe_family": "linear", "target_variant": "decoy"},
}

CELL_SPECS: dict[str, dict[str, Any]] = {
    "linear_l03": {"family": "p0", "hybrid_lambda": 0.3},
    "linear_l04": {"family": "p0", "hybrid_lambda": 0.4},
    "mlp_l05": {"family": "p1", "hybrid_lambda": 0.5},
    "mlp_pure": {"family": "p1", "hybrid_lambda": 1.0},
    "decoy_pure": {"family": "p2", "hybrid_lambda": 1.0},
    "decoy_l05": {"family": "p2", "hybrid_lambda": 0.5},
}

SCHEMA_VERSION = "cube_probe_cache_v1"
HEX40 = re.compile(r"[0-9a-f]{40}")


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


def _val_mask(length: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 991)
    mask = rng.random(length) < 0.2
    if mask.all() or not mask.any():
        mask = np.zeros(length, dtype=bool)
        mask[: max(1, length // 5)] = True
    return mask


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
            f"frozen P1 fit requires canonical float32 model parameters, found {dtype}"
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


def _base_cache_metadata(
    *,
    family: str,
    train_seed: int,
    eval_seed: int,
    checkpoint: Path,
    dataset: Path,
    exclude_episodes: Iterable[int],
    budget_meta: dict[str, Any],
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
        "jepa_wms_base_commit": JEPA_WMS_BASE_COMMIT,
        "stable_worldmodel_commit": STABLE_WORLDMODEL_COMMIT,
        "mlp_epochs": MLP_EPOCHS if family == "p1" else None,
        "created_unix": time.time(),
    }


def _cache_relpath(family: str, train_seed: int, eval_seed: int) -> Path:
    return Path("probe_caches") / family / f"ts{train_seed}_es{eval_seed}.npz"


def _validate_checkpoint(checkpoint: Path, train_seed: int) -> None:
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    expected_name = f"adadim_cube_matched3_lbproj_seed{train_seed}_20260615"
    if checkpoint.name != "weights_epoch_10.pt" or checkpoint.parent.name != expected_name:
        raise ValueError(f"canonical checkpoint lineage drift: {checkpoint}")
    actual = sha256_file(checkpoint)
    if actual != CHECKPOINT_SHA256[train_seed]:
        raise ValueError(
            f"checkpoint hash mismatch seed={train_seed}: expected={CHECKPOINT_SHA256[train_seed]} actual={actual}"
        )


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
    expected_models = {f"initialpartial_seed{seed}" for seed in TRAIN_SEEDS}
    if set(task_spec["models"]) != expected_models:
        raise ValueError(f"canonical manifest model set drift: {sorted(task_spec['models'])}")

    records: list[dict[str, Any]] = []
    all_gate_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for train_seed in TRAIN_SEEDS:
        model_name = f"initialpartial_seed{train_seed}"
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
            exclude_episodes = {int(x) for x in prepared["eval_episodes"].tolist()}
            effective_seed = PROBE_SEED + eval_seed
            sample_indices = common.make_probe_indices(
                dataset,
                exclude_episodes=exclude_episodes,
                n_train=PROBE_TRAIN_SAMPLES,
                seed=effective_seed,
            )
            val_mask = _val_mask(len(sample_indices), effective_seed)

            for family in ("p0", "p1", "p2"):
                spec = FAMILY_SPECS[family]
                if family == "p1":
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
                metadata = _base_cache_metadata(
                    family=family,
                    train_seed=train_seed,
                    eval_seed=eval_seed,
                    checkpoint=checkpoint,
                    dataset=dataset,
                    exclude_episodes=exclude_episodes,
                    budget_meta=budget_meta,
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

    expected_keys = {
        (family, train_seed, eval_seed)
        for family in FAMILY_SPECS
        for train_seed in TRAIN_SEEDS
        for eval_seed in EVAL_SEEDS
    }
    actual_keys = {
        (row["family"], row["train_seed"], row["eval_seed"]) for row in records
    }
    if actual_keys != expected_keys or len(records) != 45:
        raise RuntimeError(f"cache matrix incomplete: {len(records)} records")
    p1_a192 = [
        coordinate
        for row in all_gate_rows
        if row["family"] == "p1" and row["budget"] == 192
        for coordinate in row["val_rmse_phys_per_coordinate"]
    ]
    if len(p1_a192) != 15 * 3 or not all(math.isfinite(x) for x in p1_a192):
        gate_pass = False
        gate_value = math.inf
    else:
        gate_value = max(p1_a192)
        gate_pass = gate_value <= P1_GATE_THRESHOLD_M
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
        "mlp_hybrid_gate": {
            "aggregation": "strict max over all 15 P1 A192 caches and every target coordinate",
            "threshold_m": P1_GATE_THRESHOLD_M,
            "worst_rmse_phys_m": gate_value,
            "n_coordinates": len(p1_a192),
            "pass": gate_pass,
        },
        "source_sha256": {
            "probe_runner": PROBE_RUNNER_SHA256,
            "probe_common": PROBE_COMMON_SHA256,
            "budget_runner": BUDGET_RUNNER_SHA256,
            "task_manifest": TASK_MANIFEST_SHA256,
        },
        "checkpoint_sha256": {str(k): v for k, v in CHECKPOINT_SHA256.items()},
        "elapsed_sec": time.perf_counter() - started,
    }
    atomic_write_json(probes_root / "probe_fit_summary.json", summary)
    smoke_record = next(
        row
        for row in records
        if row["family"] == "p0" and row["train_seed"] == 42 and row["eval_seed"] == 42
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
    if not gate_pass:
        atomic_write_json(
            probes_root / "B3_GATE_BARRED.json",
            {
                "cell": "mlp_l05",
                "reason": "strict preregistered P1 gate failed",
                "h1_fixed_p": 1.0,
                "h1_family_size": 3,
                "gate": summary["mlp_hybrid_gate"],
            },
        )
    print(json.dumps(summary["mlp_hybrid_gate"], indent=2, allow_nan=False))
    return summary


def _record_lookup(summary: dict[str, Any]) -> dict[tuple[str, int, int], dict[str, Any]]:
    records = summary.get("records")
    if not isinstance(records, list) or len(records) != 45:
        raise ValueError("probe_fit_summary does not contain exactly 45 cache records")
    lookup: dict[tuple[str, int, int], dict[str, Any]] = {}
    for record in records:
        key = (str(record["family"]), int(record["train_seed"]), int(record["eval_seed"]))
        if key in lookup:
            raise ValueError(f"duplicate cache record {key}")
        lookup[key] = record
    return lookup


def recompute_strict_p1_gate(run_root: Path, summary: dict[str, Any]) -> dict[str, Any]:
    """Verify every P1 artifact and independently recompute the frozen gate."""
    lookup = _record_lookup(summary)
    values: list[float] = []
    for train_seed in TRAIN_SEEDS:
        for eval_seed in EVAL_SEEDS:
            record = lookup[("p1", train_seed, eval_seed)]
            metadata, arrays = load_npz_cache(
                run_root / record["path"], expected_sha256=record["sha256"]
            )
            _validate_loaded_metadata(
                metadata,
                family="p1",
                train_seed=train_seed,
                eval_seed=eval_seed,
            )
            coordinates = arrays["b192_val_rmse_phys"].astype(float).tolist()
            if len(coordinates) != 3 or not all(math.isfinite(x) for x in coordinates):
                raise ValueError(
                    f"P1 A192 gate coordinates invalid for train={train_seed} eval={eval_seed}"
                )
            values.extend(coordinates)
    if len(values) != 45:
        raise ValueError(f"strict P1 gate expected 45 coordinates, found {len(values)}")
    worst = max(values)
    passed = worst <= P1_GATE_THRESHOLD_M
    frozen = summary.get("mlp_hybrid_gate", {})
    expected_aggregation = (
        "strict max over all 15 P1 A192 caches and every target coordinate"
    )
    if frozen.get("aggregation") != expected_aggregation:
        raise ValueError(f"strict P1 aggregation label drift: {frozen}")
    if frozen.get("threshold_m") != P1_GATE_THRESHOLD_M:
        raise ValueError(f"strict P1 threshold drift: {frozen}")
    if frozen.get("n_coordinates") != 45:
        raise ValueError(f"strict P1 coordinate count drift: {frozen}")
    if not math.isclose(
        float(frozen.get("worst_rmse_phys_m")), worst, rel_tol=0.0, abs_tol=1e-15
    ):
        raise ValueError(
            f"strict P1 summary value differs from caches: summary={frozen.get('worst_rmse_phys_m')} caches={worst}"
        )
    if frozen.get("pass") is not passed:
        raise ValueError(
            f"strict P1 summary verdict differs from caches: summary={frozen.get('pass')} caches={passed}"
        )
    return {
        "aggregation": expected_aggregation,
        "threshold_m": P1_GATE_THRESHOLD_M,
        "worst_rmse_phys_m": worst,
        "n_coordinates": 45,
        "pass": passed,
    }


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
        "budgets": list(BUDGETS),
        "probe_train_samples": PROBE_TRAIN_SAMPLES,
        "probe_batch_size": PROBE_BATCH_SIZE,
        "probe_seed": PROBE_SEED,
        "effective_seed": PROBE_SEED + eval_seed,
        "probe_runner_sha256": PROBE_RUNNER_SHA256,
        "probe_common_sha256": PROBE_COMMON_SHA256,
        "budget_runner_sha256": BUDGET_RUNNER_SHA256,
        "task_manifest_sha256": TASK_MANIFEST_SHA256,
        "jepa_wms_base_commit": JEPA_WMS_BASE_COMMIT,
        "stable_worldmodel_commit": STABLE_WORLDMODEL_COMMIT,
    }
    for name, wanted in expected.items():
        if metadata.get(name) != wanted:
            raise ValueError(
                f"cache metadata mismatch {name}: expected={wanted!r} actual={metadata.get(name)!r}"
            )


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
            f"cached P1 planning requires canonical float32 model parameters, found {dtype}"
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


def run_cached(args: argparse.Namespace) -> dict[str, Any]:
    run_root = Path(args.run_root).resolve()
    out_root = Path(args.out_root).resolve()
    if run_root.name != RUN_ID:
        raise ValueError(f"run root basename must be {RUN_ID}: {run_root}")
    if out_root.exists():
        raise FileExistsError(f"planning output already exists: {out_root}")
    if args.cell not in CELL_SPECS:
        raise ValueError(f"unknown cell {args.cell}")
    if args.mode not in {"smoke", "full"}:
        raise ValueError(f"unsupported mode {args.mode}")
    if not HEX40.fullmatch(args.jepa_commit):
        raise ValueError(f"invalid experiment commit: {args.jepa_commit}")
    if args.stable_commit != STABLE_WORLDMODEL_COMMIT:
        raise ValueError("stable-worldmodel commit drift")

    common, runner = load_frozen_modules(args)
    fit_summary_path = run_root / "probes" / "probe_fit_summary.json"
    fit_summary = json.loads(fit_summary_path.read_text(encoding="utf-8"))
    if fit_summary.get("n_cache_artifacts") != 45:
        raise ValueError("fit summary cache count drift")
    lookup = _record_lookup(fit_summary)
    strict_gate = recompute_strict_p1_gate(run_root, fit_summary)
    cell_spec = CELL_SPECS[args.cell]
    if args.cell == "mlp_l05" and strict_gate["pass"] is not True:
        raise RuntimeError("B3 is gate-barred by the frozen strict P1 gate")
    family = str(cell_spec["family"])
    family_spec = FAMILY_SPECS[family]
    train_seeds = [42] if args.mode == "smoke" else list(TRAIN_SEEDS)
    eval_seeds = [42] if args.mode == "smoke" else list(EVAL_SEEDS)
    num_eval = 2 if args.mode == "smoke" else 50
    smoke_manifest = run_root / "probes" / "smoke_eval_seed42.json"
    eval_manifest = str(smoke_manifest) if args.mode == "smoke" else ""
    if args.mode == "smoke" and not smoke_manifest.is_file():
        raise FileNotFoundError(smoke_manifest)

    context: dict[str, Any] = {}
    loaded: dict[tuple[int, int], dict[str, Any]] = {}
    runtime_checkpoint_hashes: dict[int, str] = {}
    original_parse_model_name = common.parse_model_name
    original_import_budget_runner = common.import_budget_runner
    original_linear_fit = common.fit_probe_bundles
    original_mlp_fit = runner.fit_mlp_probes

    def tracked_parse_model_name(model_name: str) -> tuple[str, int]:
        arm, train_seed = original_parse_model_name(model_name)
        context.clear()
        context.update({"arm": arm, "train_seed": int(train_seed), "model_name": model_name})
        return arm, train_seed

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
                raise RuntimeError("prepare_common called without a bound canonical train seed")
            if (
                task != TASK
                or epoch != 10
                or requested_num_eval != num_eval
                or int(eval_seed) not in eval_seeds
            ):
                raise ValueError("frozen planning preparation arguments drifted")
            if args.mode == "full" and requested_manifest:
                raise ValueError("full run must not use an evaluation manifest")
            if args.mode == "smoke":
                if Path(str(requested_manifest)).resolve() != smoke_manifest.resolve():
                    raise ValueError("smoke did not use the cache-derived evaluation manifest")
            result = original_prepare_common(
                task,
                run,
                epoch,
                device,
                requested_num_eval,
                eval_seed,
                requested_manifest,
            )
            run_spec, prepared = result
            context["eval_seed"] = int(eval_seed)
            context["prepared"] = prepared
            context["run"] = run
            context["run_spec"] = run_spec
            return result

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
        if family != "p1":
            if kwargs.get("probe_mode") != "linear":
                raise ValueError("linear cache loader received non-linear probe_mode")
            if kwargs.get("target_variant") != family_spec["target_variant"]:
                raise ValueError("linear cache target variant drift")
        elif kwargs.get("epochs") != MLP_EPOCHS:
            raise ValueError("MLP cache loader received a non-frozen epoch count")
        key = (family, train_seed, eval_seed)
        record = lookup.get(key)
        if record is None:
            raise KeyError(f"fit summary has no cache record {key}")
        path = run_root / record["path"]
        metadata, arrays = load_npz_cache(path, expected_sha256=record["sha256"])
        _validate_loaded_metadata(
            metadata,
            family=family,
            train_seed=train_seed,
            eval_seed=eval_seed,
        )
        dataset_path = Path(dataset).resolve()
        cached_dataset = Path(metadata["dataset_path"]).resolve()
        if dataset_path != cached_dataset:
            raise ValueError(
                f"planning dataset differs from cached fit dataset: {dataset_path} != {cached_dataset}"
            )
        dataset_stat = dataset_path.stat()
        if metadata["dataset_stat"] != {
            "size": dataset_stat.st_size,
            "mtime_ns": dataset_stat.st_mtime_ns,
        }:
            raise ValueError("planning dataset identity changed after cache fitting")
        expected_run = Path(metadata["checkpoint_path"]).parent.name
        if context.get("run") != expected_run:
            raise ValueError(
                f"planning checkpoint run differs from cache: {context.get('run')} != {expected_run}"
            )
        actual_checkpoint = Path(context["run_spec"]["checkpoint"]).resolve()
        cached_checkpoint = Path(metadata["checkpoint_path"]).resolve()
        if actual_checkpoint != cached_checkpoint:
            raise ValueError(
                f"planning checkpoint path differs from cache: {actual_checkpoint} != {cached_checkpoint}"
            )
        if train_seed not in runtime_checkpoint_hashes:
            runtime_checkpoint_hashes[train_seed] = sha256_file(actual_checkpoint)
        if runtime_checkpoint_hashes[train_seed] != CHECKPOINT_SHA256[train_seed]:
            raise ValueError(
                f"planning checkpoint hash differs from frozen seed {train_seed} checkpoint"
            )
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
                raise ValueError("smoke evaluation episodes are not a subset of full cache exclusions")
            smoke_payload = json.loads(smoke_manifest.read_text(encoding="utf-8"))
            if set(smoke_payload["planning_eval_episode_ids"]) != cached_excluded:
                raise ValueError("smoke manifest contents differ from cache exclusion set")
        pair = (train_seed, eval_seed)
        if pair in loaded:
            raise RuntimeError(f"cache loaded more than once for planning pair {pair}")
        loaded[pair] = {
            "path": record["path"],
            "sha256": record["sha256"],
            "exclude_episodes_sha256": metadata["exclude_episodes_sha256"],
        }
        if family == "p1":
            return _reconstruct_mlp(
                runner, metadata, arrays, model=model, device=device
            )
        return _reconstruct_linear(common, metadata, arrays)

    def forbidden_fit(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("unselected/original probe fit route was invoked during cached planning")

    common.parse_model_name = tracked_parse_model_name
    common.import_budget_runner = tracked_import_budget_runner
    if family == "p1":
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
            "mode": args.mode,
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
            "checkpoint_sha256": {
                str(seed): CHECKPOINT_SHA256[seed] for seed in train_seeds
            },
            "probe_runner_sha256": PROBE_RUNNER_SHA256,
            "probe_common_sha256": PROBE_COMMON_SHA256,
            "budget_runner_sha256": BUDGET_RUNNER_SHA256,
            "task_manifest_sha256": TASK_MANIFEST_SHA256,
            "jepa_wms_commit": args.jepa_commit,
            "jepa_wms_base_commit": JEPA_WMS_BASE_COMMIT,
            "stable_worldmodel_commit": STABLE_WORLDMODEL_COMMIT,
            "worktree_clean": True,
            "smoke_eval_manifest_sha256": (
                sha256_file(smoke_manifest) if args.mode == "smoke" else ""
            ),
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

    fit = subparsers.add_parser("fit-cache", help="fit and seal the 45 immutable caches")
    add_source_arguments(fit)
    fit.add_argument("--run-root", required=True)
    fit.add_argument("--device", default="cuda")

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
    cached.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "fit-cache":
        fit_cache(args)
    elif args.command == "run-cached":
        run_cached(args)
    else:  # pragma: no cover - argparse enforces this
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
