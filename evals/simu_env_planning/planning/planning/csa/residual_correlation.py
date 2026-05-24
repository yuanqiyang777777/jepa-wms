# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
"""Exp C: local residual calibration vs data-coverage support distance.

Main question (memo v3, §4/C):

    Does local residual risk r(q), estimated from real validation transition
    errors, predict rollout / realized error better than support distance
    U_action|x?

This is the key empirical answer to "support does not equal correctness."

This module:

* Loads existing per-episode dumps (the same ones diagnostics 0-6 and the
  geometry-sensitivity sweep already consumed).
* Loads the raw support tensors ``<env>_states.pt`` / ``<env>_actions.pt``
  (training-split memory M_real) and the raw residual tensors
  ``<env>_residual_{states,actions,errors}.pt`` (val-split memory M_resid
  built by ``encode_residual_data.py``).
* For each geometry variant in ``residual_scorer.default_variants_for_correlation()``
  (raw = PRIMARY, PCA-64 = SENSITIVITY), under matched calibration:
  - computes U_state(q), U_action|x(q), U_SA(q) on M_real;
  - computes r(q) on M_resid (same variant);
  - aggregates Pearson + Spearman correlations of each of those four
    scorers against the dumps' realized rollout error, per env, per
    rollout depth, and stratified by U_state quantile (R4 transfer
    mitigation -- show whether r(q) holds up far from the val distribution).

Output is a single JSON-ready dict + a small per-env verdict block that
``run_residual_correlation.py`` writes to disk. The cross-env roll-up
into a Go/No-Go table is left to ``summarize_residual_correlation.py``.

What this module deliberately does NOT do:
* It does not propose any reranker (that's Exp B, gated on C passing).
* It does not retrain the world model or fit a parametric residual head
  (that's an Exp C fallback in Q7 of the v3 memo).
* It does not claim r(q) is a safety signal or a calibrated uncertainty.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from evals.simu_env_planning.planning.planning.csa.geometry_sensitivity import (
    GeometryVariant,
    _gather_queries_from_dumps,
    build_calibrated_memory,
    score_queries,
)
from evals.simu_env_planning.planning.planning.csa.residual_scorer import (
    build_calibrated_residual_memory,
    default_variants_for_correlation,
    score_residual_queries,
)


# --------------------------------------------------------------------------
# Pearson + Spearman with a NaN-safe interface.
# --------------------------------------------------------------------------

def _pearson(x: torch.Tensor, y: torch.Tensor) -> Optional[float]:
    if x.numel() < 2 or y.numel() < 2:
        return None
    xf = x.to(torch.float32) - x.to(torch.float32).mean()
    yf = y.to(torch.float32) - y.to(torch.float32).mean()
    denom = torch.linalg.vector_norm(xf) * torch.linalg.vector_norm(yf)
    if float(denom.item()) <= 1e-12:
        return None
    return float((xf * yf).sum().item() / float(denom.item()))


def _spearman(x: torch.Tensor, y: torch.Tensor) -> Optional[float]:
    if x.numel() < 2 or y.numel() < 2:
        return None

    def _rank(z: torch.Tensor) -> torch.Tensor:
        order = torch.argsort(z)
        ranks = torch.empty_like(order, dtype=torch.float32)
        ranks[order] = torch.arange(z.numel(), dtype=torch.float32)
        return ranks

    return _pearson(_rank(x), _rank(y))


# --------------------------------------------------------------------------
# Per-(variant, depth, stratum) correlation block.
# --------------------------------------------------------------------------

def _corr_block(
    scorer_values: Dict[str, torch.Tensor],
    error: torch.Tensor,
) -> Dict[str, Any]:
    """Compute Pearson + Spearman of each scorer in ``scorer_values`` against
    ``error``. Returns a dict ``{<scorer_key>: {"pearson": ..., "spearman": ..., "count": ...}}``.
    """
    out: Dict[str, Any] = {"count": int(error.numel())}
    for name, vals in scorer_values.items():
        if vals.numel() != error.numel():
            raise ValueError(
                f"scorer {name!r} length {vals.numel()} != error length {error.numel()}"
            )
        out[name] = {
            "pearson": _pearson(vals, error),
            "spearman": _spearman(vals, error),
        }
    return out


def _strata_by_u_state(
    u_state: torch.Tensor,
    quantiles: Sequence[float] = (0.50,),
) -> Dict[str, torch.Tensor]:
    """Boolean strata masks keyed by descriptive name. ``familiar_qXX``
    selects queries with ``U_state <= XXth percentile`` (state-familiar
    subset). Always includes ``all`` as the no-stratification baseline."""
    masks: Dict[str, torch.Tensor] = {"all": torch.ones_like(u_state, dtype=torch.bool)}
    for q in quantiles:
        thr = float(torch.quantile(u_state, float(q)).item())
        masks[f"familiar_q{int(round(q * 100)):02d}"] = u_state <= thr
    return masks


def _r_beats_summary(per_depth: Dict[str, Any]) -> Dict[str, Any]:
    """Across per-depth blocks, count how often r(q) beats U_action|x and U_SA
    on the |Pearson| with rollout error. Used by the C verdict.

    Beats = stricly larger absolute Pearson on the state-familiar stratum
    (the diag-6 / diag-2 convention).
    """
    depths_with_data = 0
    r_beats_action = 0
    r_beats_csa = 0
    for d_key, d_block in per_depth.items():
        fam = d_block.get("familiar_q50")
        if fam is None or fam.get("count", 0) < 2:
            continue
        r_pearson = fam.get("r", {}).get("pearson")
        ua_pearson = fam.get("u_action", {}).get("pearson")
        ucsa_pearson = fam.get("u_sa", {}).get("pearson")
        if r_pearson is None or ua_pearson is None or ucsa_pearson is None:
            continue
        depths_with_data += 1
        if abs(r_pearson) > abs(ua_pearson):
            r_beats_action += 1
        if abs(r_pearson) > abs(ucsa_pearson):
            r_beats_csa += 1
    return {
        "depths_with_data": depths_with_data,
        "r_beats_u_action_count": r_beats_action,
        "r_beats_u_sa_count": r_beats_csa,
        "r_beats_u_action_fraction": (
            r_beats_action / depths_with_data if depths_with_data > 0 else None
        ),
        "r_beats_u_sa_fraction": (
            r_beats_csa / depths_with_data if depths_with_data > 0 else None
        ),
    }


# --------------------------------------------------------------------------
# Top-level per-env driver.
# --------------------------------------------------------------------------

def run_residual_correlation(
    raw_support_states: torch.Tensor,
    raw_support_actions: torch.Tensor,
    raw_resid_states: torch.Tensor,
    raw_resid_actions: torch.Tensor,
    raw_resid_errors: torch.Tensor,
    dumps: Sequence[Dict[str, Any]],
    *,
    variants: Optional[Sequence[GeometryVariant]] = None,
    state_k: int = 64,
    action_k_support: int = 64,
    action_k_residual: int = 8,
    state_familiar_quantile: float = 0.50,
    pca_fit_samples: Optional[int] = 50_000,
    device: str = "cpu",
) -> Dict[str, Any]:
    """For one env: build matched (M_real, M_resid) memories under each
    variant, score every (episode, replan, depth) query, and report per-depth
    + per-stratum Pearson / Spearman of (r, U_state, U_action|x, U_SA) vs the
    dump's realized rollout error.

    Returns a JSON-ready dict. Variants default to raw (PRIMARY) + PCA-64
    (SENSITIVITY) per the v3 memo.
    """
    if variants is None:
        variants = default_variants_for_correlation()

    queries = _gather_queries_from_dumps(dumps)
    n_queries = int(queries["raw_states"].shape[0])
    n_decision = int((queries["depth"] == 0).sum().item())
    env = dumps[0].get("env", "") if dumps else ""

    out: Dict[str, Any] = {
        "env": env,
        "num_episodes": len(dumps),
        "num_failed_episodes": int(sum(1 for d in dumps if int(d.get("episode_success", 0)) <= 0)),
        "num_queries_scored": n_queries,
        "num_decision_steps": n_decision,
        "raw_support_size": int(raw_support_states.shape[0]),
        "raw_residual_size": int(raw_resid_states.shape[0]),
        "residual_memory_summary": {
            "num_items": int(raw_resid_errors.shape[0]),
            "mean_residual": float(raw_resid_errors.to(torch.float32).mean().item())
            if raw_resid_errors.numel() > 0 else None,
            "std_residual": float(raw_resid_errors.to(torch.float32).std(unbiased=False).item())
            if raw_resid_errors.numel() > 1 else None,
            "p95_residual": (
                float(torch.quantile(raw_resid_errors.to(torch.float32), 0.95).item())
                if raw_resid_errors.numel() > 0 else None
            ),
        },
        "scorer": {
            "state_k": state_k,
            "action_k_support": action_k_support,
            "action_k_residual": action_k_residual,
        },
        "thresholds": {"state_familiar_quantile": state_familiar_quantile},
        "variants": {},
    }

    if n_queries == 0:
        out["status"] = "not_computed"
        out["reason"] = "no queries from dumps"
        return out
    if int(raw_resid_errors.shape[0]) < state_k:
        out["status"] = "not_computed"
        out["reason"] = (
            f"residual memory has only {int(raw_resid_errors.shape[0])} transitions "
            f"(< state_k={state_k}); scoring not meaningful"
        )
        return out

    raw_support_states = raw_support_states.to(device=device, dtype=torch.float32)
    raw_support_actions = raw_support_actions.to(device=device, dtype=torch.float32)
    raw_resid_states = raw_resid_states.to(device=device, dtype=torch.float32)
    raw_resid_actions = raw_resid_actions.to(device=device, dtype=torch.float32)
    raw_resid_errors = raw_resid_errors.to(device=device, dtype=torch.float32)
    q_states = queries["raw_states"].to(device=device, dtype=torch.float32)
    q_actions = queries["raw_actions"].to(device=device, dtype=torch.float32)
    depth = queries["depth"].to(device=device)
    rollout_error = queries["rollout_error"].to(device=device)

    # Only rollout queries (depth >= 1) carry a real rollout_error -- depth 0
    # is the true observation with error = 0 by definition. Diag-2/diag-6 use
    # the same filter.
    rollout_mask_global = (depth >= 1) & torch.isfinite(rollout_error)
    if int(rollout_mask_global.sum().item()) < 2:
        out["status"] = "not_computed"
        out["reason"] = "fewer than 2 rollout queries with finite rollout_error"
        return out

    for variant in variants:
        support_mem = build_calibrated_memory(
            raw_support_states, raw_support_actions, variant,
            pca_fit_samples=pca_fit_samples,
        )
        resid_mem = build_calibrated_residual_memory(
            raw_resid_states, raw_resid_actions, raw_resid_errors,
            variant=variant, pca_fit_samples=pca_fit_samples,
        )

        u_state, u_action = score_queries(
            q_states, q_actions, support_mem,
            state_k=state_k, action_k=action_k_support,
        )
        u_sa = u_state + u_action  # beta=1.0, matches diag-6
        r_q = score_residual_queries(
            q_states, q_actions, resid_mem,
            state_k=state_k, action_k=action_k_residual,
        )

        # Restrict everything to the rollout subset before correlating.
        us = u_state[rollout_mask_global]
        ua = u_action[rollout_mask_global]
        ucsa = u_sa[rollout_mask_global]
        rr = r_q[rollout_mask_global]
        err = rollout_error[rollout_mask_global]
        d_sub = depth[rollout_mask_global]

        scorer_vals_all = {"u_state": us, "u_action": ua, "u_sa": ucsa, "r": rr}

        # State-familiar strata (same threshold as diag-6).
        strata = _strata_by_u_state(us, quantiles=(state_familiar_quantile,))

        # Conditioned-overall block (across all depths in the rollout subset).
        conditioned_overall: Dict[str, Any] = {}
        for stratum_name, mask in strata.items():
            if int(mask.sum().item()) < 2:
                conditioned_overall[stratum_name] = {"count": int(mask.sum().item())}
                continue
            conditioned_overall[stratum_name] = _corr_block(
                {k: v[mask] for k, v in scorer_vals_all.items()},
                err[mask],
            )

        # Per-depth block, stratified.
        per_depth: Dict[str, Any] = {}
        for d in sorted({int(x.item()) for x in d_sub}):
            d_mask = d_sub == int(d)
            depth_block: Dict[str, Any] = {}
            for stratum_name, mask in strata.items():
                combined = d_mask & mask
                if int(combined.sum().item()) < 2:
                    depth_block[stratum_name] = {"count": int(combined.sum().item())}
                    continue
                depth_block[stratum_name] = _corr_block(
                    {k: v[combined] for k, v in scorer_vals_all.items()},
                    err[combined],
                )
            per_depth[f"depth_{int(d)}"] = depth_block

        r_beats = _r_beats_summary(per_depth)

        out["variants"][variant.name] = {
            "variant": {
                "name": variant.name,
                "pca_dim": variant.pca_dim,
                "whiten": bool(variant.whiten),
                "calibration": variant.calibration,
                "is_primary_no_pca_variant": variant.pca_dim is None,
                "description": variant.describe(),
            },
            "memory_summary": {
                "support_num_items": support_mem.num_items,
                "residual_num_items": resid_mem.num_items,
                "state_dim_after_geometry": int(resid_mem.state_features.shape[1]),
                "state_scale": resid_mem.state_scale,
                "action_scale": resid_mem.action_scale,
            },
            "conditioned_overall": conditioned_overall,
            "per_depth": per_depth,
            "r_beats_summary": r_beats,
        }

    out["status"] = "computed"
    out["verdict"] = _per_env_verdict(out)
    return out


# --------------------------------------------------------------------------
# Per-env verdict: did C pass on this env? (advisory, the human reads numbers.)
# --------------------------------------------------------------------------

# Memo v3 §5: C passes on an env if r(q) beats U_action|x on |Pearson| in
# >= a majority of (env, depth) cells on the state-familiar stratum at
# depths 1-3, under raw geometry.
C_PASS_MAJORITY_FRACTION = 0.5
C_PRIMARY_VARIANT = "raw_recipRMS"
C_PRIMARY_DEPTHS = (1, 2, 3)


def _per_env_verdict(report: Dict[str, Any]) -> Dict[str, Any]:
    """Return a per-env verdict block. Variant-specific: surfaces the primary
    raw verdict and the PCA-64 sensitivity verdict side-by-side."""
    verdict: Dict[str, Any] = {
        "primary_variant": C_PRIMARY_VARIANT,
        "primary_depths": list(C_PRIMARY_DEPTHS),
        "pass_majority_fraction": C_PASS_MAJORITY_FRACTION,
    }
    variants = report.get("variants", {})
    for variant_name, vblock in variants.items():
        per_depth = vblock.get("per_depth", {})
        primary_cells = [
            per_depth.get(f"depth_{d}", {}).get("familiar_q50", {})
            for d in C_PRIMARY_DEPTHS
        ]
        n_with_data = 0
        n_beats_action = 0
        n_beats_csa = 0
        for cell in primary_cells:
            r = cell.get("r", {}).get("pearson") if cell.get("count", 0) >= 2 else None
            ua = cell.get("u_action", {}).get("pearson") if cell.get("count", 0) >= 2 else None
            uc = cell.get("u_sa", {}).get("pearson") if cell.get("count", 0) >= 2 else None
            if r is None or ua is None or uc is None:
                continue
            n_with_data += 1
            if abs(r) > abs(ua):
                n_beats_action += 1
            if abs(r) > abs(uc):
                n_beats_csa += 1
        if n_with_data == 0:
            verdict[variant_name] = {"status": "insufficient_depths"}
            continue
        frac_action = n_beats_action / n_with_data
        frac_csa = n_beats_csa / n_with_data
        verdict[variant_name] = {
            "depths_with_data": n_with_data,
            "r_beats_u_action_fraction": frac_action,
            "r_beats_u_sa_fraction": frac_csa,
            "passes_vs_u_action": frac_action > C_PASS_MAJORITY_FRACTION,
            "passes_vs_u_sa": frac_csa > C_PASS_MAJORITY_FRACTION,
            "passes_c": bool(
                frac_action > C_PASS_MAJORITY_FRACTION
                and frac_csa > C_PASS_MAJORITY_FRACTION
            ),
        }
    # Overall C verdict for this env: anchored on the PRIMARY raw variant.
    primary = verdict.get(C_PRIMARY_VARIANT, {})
    verdict["env_c_pass_under_raw"] = bool(primary.get("passes_c", False))
    return verdict
