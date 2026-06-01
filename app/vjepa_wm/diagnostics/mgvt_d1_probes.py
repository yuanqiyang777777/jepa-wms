"""Stage-D1 post-hoc diagnostics for the dynamics trend bottleneck.

The CLI intentionally supports lightweight JSON/NPZ bundles so smoke scripts
can verify probe artifact plumbing before full GPU scans. The heavier probe
training can reuse the same helpers after frozen ``d_h`` dumps exist.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _as_2d(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    return array.reshape(array.shape[0], -1)


def ridge_r2(x: np.ndarray, y: np.ndarray, alpha: float = 1.0e-3) -> float:
    x = _as_2d(x)
    y = _as_2d(y)
    x = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
    x_rows = x.tolist()
    y_rows = y.tolist()
    n = len(x_rows)
    in_dim = len(x_rows[0])
    out_dim = len(y_rows[0])
    coef = [[0.0 for _ in range(out_dim)] for _ in range(in_dim)]
    # Use a small deterministic optimizer instead of LAPACK-backed solves so
    # Windows/MKL instability cannot abort smoke tests.
    lipschitz = float((x * x).sum(axis=1).max() + alpha + 1.0e-6)
    lr = 0.2 / lipschitz
    for _ in range(1000):
        grad = [[0.0 for _ in range(out_dim)] for _ in range(in_dim)]
        for row, target in zip(x_rows, y_rows):
            pred_row = [sum(row[k] * coef[k][j] for k in range(in_dim)) for j in range(out_dim)]
            err = [pred_row[j] - target[j] for j in range(out_dim)]
            for k in range(in_dim):
                for j in range(out_dim):
                    grad[k][j] += row[k] * err[j] / n
        for k in range(in_dim - 1):
            for j in range(out_dim):
                grad[k][j] += alpha * coef[k][j]
        for k in range(in_dim):
            for j in range(out_dim):
                coef[k][j] -= lr * grad[k][j]

    pred_rows = [
        [sum(row[k] * coef[k][j] for k in range(in_dim)) for j in range(out_dim)]
        for row in x_rows
    ]
    ss_res = 0.0
    means = [sum(row[j] for row in y_rows) / n for j in range(out_dim)]
    ss_tot = 0.0
    for pred_row, target in zip(pred_rows, y_rows):
        for j in range(out_dim):
            ss_res += (target[j] - pred_row[j]) ** 2
            ss_tot += (target[j] - means[j]) ** 2
    return 0.0 if ss_tot <= 0.0 else 1.0 - ss_res / ss_tot


def nearest_centroid_accuracy(x: np.ndarray, labels: np.ndarray) -> float:
    x = _as_2d(x)
    labels = np.asarray(labels)
    centroids = []
    classes = np.unique(labels)
    for cls in classes:
        centroids.append(x[labels == cls].mean(axis=0))
    centroid_matrix = np.stack(centroids, axis=0)
    dists = ((x[:, None, :] - centroid_matrix[None, :, :]) ** 2).sum(axis=-1)
    pred = classes[dists.argmin(axis=1)]
    return float((pred == labels).mean())


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = _as_2d(a)
    b = _as_2d(b)
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1.0e-12)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1.0e-12)
    return (a * b).sum(axis=1)


def cross_position_margin(
    d_h: np.ndarray,
    positive_pairs: np.ndarray,
    negative_pairs: np.ndarray,
) -> dict[str, float]:
    d_h = _as_2d(d_h)
    positive_pairs = np.asarray(positive_pairs, dtype=np.int64)
    negative_pairs = np.asarray(negative_pairs, dtype=np.int64)
    pos = cosine_similarity(d_h[positive_pairs[:, 0]], d_h[positive_pairs[:, 1]])
    neg = cosine_similarity(d_h[negative_pairs[:, 0]], d_h[negative_pairs[:, 1]])
    return {
        "positive_similarity_mean": float(pos.mean()) if pos.size else 0.0,
        "negative_similarity_mean": float(neg.mean()) if neg.size else 0.0,
        "cross_position_margin": float(pos.mean() - neg.mean()) if pos.size and neg.size else 0.0,
        "positive_pairs": int(pos.size),
        "negative_pairs": int(neg.size),
    }


def summarize_probe_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    if "d_h" in bundle and "relative_motion" in bundle:
        summary["relative_motion_r2"] = ridge_r2(bundle["d_h"], bundle["relative_motion"])
    if "d_h" in bundle and "future_z" in bundle:
        summary["future_z_r2"] = ridge_r2(bundle["d_h"], bundle["future_z"])
    if "d_h" in bundle and "absolute_region" in bundle:
        summary["absolute_region_accuracy"] = nearest_centroid_accuracy(bundle["d_h"], bundle["absolute_region"])
    if "d_h" in bundle and "task_id" in bundle:
        summary["task_id_accuracy"] = nearest_centroid_accuracy(bundle["d_h"], bundle["task_id"])
    if {"d_h", "positive_pairs", "negative_pairs"}.issubset(bundle):
        summary.update(cross_position_margin(bundle["d_h"], bundle["positive_pairs"], bundle["negative_pairs"]))
    return summary


def _load_npz(path: Path) -> dict[str, Any]:
    data = np.load(path, allow_pickle=False)
    return {key: data[key] for key in data.files}


def _summary_from_skill_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "source": str(path),
        "model": data.get("model"),
        "task": data.get("task"),
        "seed": data.get("seed"),
        "moved_region_h4": data.get("moved_region_change_skill_by_horizon", {}).get("4"),
        "skill_h4": data.get("skill_by_horizon", {}).get("4"),
        "change_skill_h4": data.get("change_skill_by_horizon", {}).get("4"),
        "probe_status": "skill-json-only",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", default=None, help="NPZ file with frozen d_h/probe arrays.")
    parser.add_argument("--skill-json", default=None, help="Fallback skill_score.json to create smoke probe metadata.")
    parser.add_argument("--output", required=True, help="Output JSON path or directory.")
    args = parser.parse_args()

    if args.bundle:
        summary = summarize_probe_bundle(_load_npz(Path(args.bundle)))
        summary["source"] = str(Path(args.bundle).resolve())
    elif args.skill_json:
        summary = _summary_from_skill_json(Path(args.skill_json))
    else:
        raise SystemExit("Pass either --bundle or --skill-json")

    output = Path(args.output).expanduser().resolve()
    if output.suffix.lower() != ".json":
        output.mkdir(parents=True, exist_ok=True)
        output = output / "mgvt_d1_probes.json"
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
