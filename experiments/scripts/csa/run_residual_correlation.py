#!/usr/bin/env python
"""Run the Exp C residual-correlation analysis for one env's dumps.

For one env, loads:
  * existing per-episode dumps (csa_diag_dump.pt files);
  * raw support tensors (``<env>_states.pt`` / ``<env>_actions.pt``) -- M_real;
  * raw residual tensors (``<env>_residual_{states,actions,errors}.pt``) --
    M_resid produced by ``encode_residual_data.py``.

Builds matched (support, residual) memories under each of the v3-defined
correlation variants (raw + PCA-64) and writes a single JSON report describing
per-depth + per-stratum Pearson / Spearman correlations of (r, U_state,
U_action|x, U_SA) against the realized rollout error from the dumps.

DOES NOT touch the existing diag-0-6 reports, the geometry-sensitivity reports,
or the raw eval dumps -- writes only to ``--output`` (default
``<env>_residual_correlation_report.json``).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals.simu_env_planning.planning.planning.csa.diagnostic_recorder import (
    load_episode_dumps,
)
from evals.simu_env_planning.planning.planning.csa.residual_correlation import (
    run_residual_correlation,
)
from evals.simu_env_planning.planning.planning.csa.residual_scorer import (
    default_variants_for_correlation,
)


def _load_tensor(path: str) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        for k in ("tensor", "states", "actions", "errors", "state_features"):
            if k in obj:
                return obj[k]
        raise ValueError(f"could not unwrap tensor dict at {path}: keys={list(obj.keys())}")
    return obj


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", required=True, help="env tag, stamped onto dumps that lack one")
    parser.add_argument("--dump-dir", required=True, help="directory with per-episode csa_diag_dump.pt files")
    parser.add_argument("--raw-support-states", required=True, help="<env>_states.pt (M_real states)")
    parser.add_argument("--raw-support-actions", required=True, help="<env>_actions.pt (M_real actions)")
    parser.add_argument("--raw-residual-states", required=True, help="<env>_residual_states.pt (M_resid states)")
    parser.add_argument("--raw-residual-actions", required=True, help="<env>_residual_actions.pt (M_resid actions)")
    parser.add_argument("--raw-residual-errors", required=True, help="<env>_residual_errors.pt (M_resid e_i)")
    parser.add_argument("--output", required=True, help="output JSON path")
    parser.add_argument("--state-k", type=int, default=64)
    parser.add_argument("--action-k-support", type=int, default=64,
                        help="action_k for the coverage scorer (matches diag-0-6 default).")
    parser.add_argument("--action-k-residual", type=int, default=8,
                        help="action_k for the residual scorer (smaller so r is local).")
    parser.add_argument("--state-familiar-quantile", type=float, default=0.50,
                        help="state-familiar threshold for the conditioned correlation (diag-2/diag-6 default).")
    parser.add_argument("--pca-fit-samples", type=int, default=50_000)
    parser.add_argument("--device", default="cpu",
                        help="cpu (default) or cuda:N. The sweep is small; CPU usually beats GPU-launch overhead.")
    args = parser.parse_args()

    dumps = load_episode_dumps(args.dump_dir)
    if not dumps:
        sys.stderr.write(f"ERROR: no dumps under {args.dump_dir}\n")
        return 1
    for d in dumps:
        d.setdefault("env", args.env)

    raw_support_states = _load_tensor(args.raw_support_states)
    raw_support_actions = _load_tensor(args.raw_support_actions)
    raw_residual_states = _load_tensor(args.raw_residual_states)
    raw_residual_actions = _load_tensor(args.raw_residual_actions)
    raw_residual_errors = _load_tensor(args.raw_residual_errors)

    report = run_residual_correlation(
        raw_support_states=raw_support_states,
        raw_support_actions=raw_support_actions,
        raw_resid_states=raw_residual_states,
        raw_resid_actions=raw_residual_actions,
        raw_resid_errors=raw_residual_errors,
        dumps=dumps,
        variants=default_variants_for_correlation(),
        state_k=args.state_k,
        action_k_support=args.action_k_support,
        action_k_residual=args.action_k_residual,
        state_familiar_quantile=args.state_familiar_quantile,
        pca_fit_samples=args.pca_fit_samples,
        device=args.device,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(f"Wrote residual-correlation report -> {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
