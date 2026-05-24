# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
"""Support-geometry sensitivity analysis for the CSA diagnostic phase.

Diagnostics 0-6 use a single fixed support estimator (PCA-64 + whitening +
reciprocal-RMS modality calibration). A No-Go from those is, strictly,
a No-Go *for that scorer geometry*. PCA does not collapse the underlying
JEPA latent -- it can however collapse the kNN support metric:

* PCA on the *pooled* visual latent can drop low-variance but task-critical
  directions (wall / contact / boundary indicators, etc.) that the encoder
  itself still encodes.
* Whitening on the top-k singular values can amplify noisy directions to the
  same scale as informative ones.
* Reciprocal-RMS modality calibration mixes the state and action distance
  contributions; an "identity" baseline is needed to verify the calibration
  isn't responsible for the wrong-signed signals seen on Wall.

This module:

* Defines a small enumerable family of (PCA dim, whiten, calibration)
  geometry variants.
* Builds a fresh support memory per env per variant from the raw 384-d pooled
  visual + raw action tensors (no eval re-run needed).
* Re-scores existing v1 dumps (the ones already written under jepawm_logs)
  against each variant.
* Reports the full Q1-Q4 quadrant census (state familiar x action familiar
  cross-cut) per outcome and per rollout depth, with enrichment ratios and
  mean rollout error per quadrant.

The four quadrants (renamed per reviewer spec):

* **Q1**  state familiar + action familiar    -- in-distribution
* **Q2**  state familiar + action unfamiliar  -- **CSA KEY** (memo's premise)
* **Q3**  state unfamiliar + action familiar  -- state OOD only
* **Q4**  state unfamiliar + action unfamiliar -- generic OOD / hallucination

* state familiar  = U_state low
* action familiar = U_action low

The interpretation framework (advisory only, the human reads the numbers):

* **Q2 enrichment in failures** -> CSA-MPC premise plausible on this env.
* **Q4 enrichment**             -> generic OOD; a CSA penalty would not help.
* **Q3 enrichment**             -> state OOD; not action-support driven.
* **Q1 enrichment**             -> failures land inside the familiar region;
                                   a support penalty is unlikely to help.

Outputs are written to NEW filenames (``*_geometry_sensitivity_report.json``)
and never touch the existing diag 0-6 reports or raw dumps.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from evals.simu_env_planning.planning.planning.csa.feature_adapter import (
    PCAWhiteningAdapter,
)


# --------------------------------------------------------------------------
# Variant catalogue.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class GeometryVariant:
    """One (PCA dim, whitening, modality-calibration) scorer geometry.

    pca_dim=None means "no PCA projection" -- the raw pooled-visual 384-d
    state vectors (plus raw actions) are used directly. whiten only applies
    when pca_dim is set.
    """
    name: str
    pca_dim: Optional[int]
    whiten: bool
    calibration: str  # "reciprocal_rms" | "identity"

    def describe(self) -> str:
        geom = "raw" if self.pca_dim is None else f"pca{self.pca_dim}{'_white' if self.whiten else '_nowhite'}"
        return f"{geom}__{self.calibration}"


# The six variants the diagnostic phase compares. Variant 2 reproduces the
# current default; variants 1 and 6 are the primary "minimum-distortion"
# diagnostics (no PCA); variants 3 / 4 sweep PCA dim and whitening; variant 5
# isolates the modality calibration on the current geometry.
DEFAULT_VARIANTS: Tuple[GeometryVariant, ...] = (
    GeometryVariant("raw_recipRMS",            pca_dim=None, whiten=False, calibration="reciprocal_rms"),
    GeometryVariant("pca64_white_recipRMS",    pca_dim=64,   whiten=True,  calibration="reciprocal_rms"),
    GeometryVariant("pca128_nowhite_recipRMS", pca_dim=128,  whiten=False, calibration="reciprocal_rms"),
    GeometryVariant("pca128_white_recipRMS",   pca_dim=128,  whiten=True,  calibration="reciprocal_rms"),
    GeometryVariant("pca64_white_identity",    pca_dim=64,   whiten=True,  calibration="identity"),
    GeometryVariant("raw_identity",            pca_dim=None, whiten=False, calibration="identity"),
)

PRIMARY_VARIANT_NAMES: Tuple[str, ...] = ("raw_recipRMS", "raw_identity")
"""Variants used as the *primary* (no-PCA) Go/No-Go evidence. The others are
sensitivity checks; the final verdict is anchored on these two."""


# --------------------------------------------------------------------------
# Calibrated memory + scoring.
# --------------------------------------------------------------------------

@dataclass
class CalibratedMemory:
    """One env's support memory under one geometry variant. Lives in RAM only --
    these are rebuildable in seconds from the raw .pt files, so we don't save
    them to disk during the sensitivity sweep."""
    variant: GeometryVariant
    state_features: torch.Tensor   # (N, D_state) post-geometry, post-calibration
    actions: torch.Tensor          # (N, D_action) post-calibration
    state_adapter: Optional[PCAWhiteningAdapter]
    state_scale: float
    action_scale: float

    @property
    def num_items(self) -> int:
        return int(self.state_features.shape[0])


def _reciprocal_rms(x: torch.Tensor, eps: float = 1e-8) -> float:
    """Per-modality reciprocal-RMS scalar (matches build_support_memory's
    default modality calibration)."""
    rms = float(x.to(torch.float32).pow(2).mean().sqrt().item())
    return 1.0 / max(rms, eps)


def build_calibrated_memory(
    raw_states: torch.Tensor,
    raw_actions: torch.Tensor,
    variant: GeometryVariant,
    pca_fit_samples: Optional[int] = None,
) -> CalibratedMemory:
    """Build the (state_features, actions) memory tensors for ``variant``.

    Steps:
      1. Optional PCA + (optional) whitening on raw_states.
      2. Per-modality reciprocal-RMS or identity calibration.

    pca_fit_samples optionally caps the rows used to fit PCA (full memory is
    still projected); this matches build_support_memory's --pca-fit-samples
    knob and keeps SVD cheap on the largest envs.
    """
    raw_states = raw_states.to(torch.float32).contiguous()
    raw_actions = raw_actions.to(torch.float32).contiguous()
    if raw_states.shape[0] != raw_actions.shape[0]:
        raise ValueError(
            f"raw_states ({raw_states.shape[0]}) and raw_actions "
            f"({raw_actions.shape[0]}) must have matching row counts"
        )

    if variant.pca_dim is None:
        state_adapter = None
        state_proj = raw_states
    else:
        fit_x = raw_states
        if pca_fit_samples is not None and pca_fit_samples < raw_states.shape[0]:
            idx = torch.randperm(raw_states.shape[0])[:pca_fit_samples]
            fit_x = raw_states[idx]
        state_adapter = PCAWhiteningAdapter.fit(
            fit_x, dim=int(variant.pca_dim), whiten=bool(variant.whiten),
        )
        state_proj = state_adapter.transform(raw_states)

    if variant.calibration == "reciprocal_rms":
        state_scale = _reciprocal_rms(state_proj)
        action_scale = _reciprocal_rms(raw_actions)
    elif variant.calibration == "identity":
        state_scale = 1.0
        action_scale = 1.0
    else:
        raise ValueError(
            f"Unknown calibration '{variant.calibration}'. "
            "Expected 'reciprocal_rms' or 'identity'."
        )

    return CalibratedMemory(
        variant=variant,
        state_features=(state_proj * state_scale).contiguous(),
        actions=(raw_actions * action_scale).contiguous(),
        state_adapter=state_adapter,
        state_scale=state_scale,
        action_scale=action_scale,
    )


def _transform_query_states(
    raw_query: torch.Tensor, memory: CalibratedMemory
) -> torch.Tensor:
    if memory.state_adapter is None:
        proj = raw_query.to(torch.float32)
    else:
        proj = memory.state_adapter.transform(raw_query)
    return proj * memory.state_scale


def _transform_query_actions(
    raw_action: torch.Tensor, memory: CalibratedMemory
) -> torch.Tensor:
    return raw_action.to(torch.float32) * memory.action_scale


def score_queries(
    raw_query_states: torch.Tensor,
    raw_query_actions: torch.Tensor,
    memory: CalibratedMemory,
    state_k: int = 64,
    action_k: int = 64,
    chunk_size: int = 8192,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exact-kNN (U_state, U_action|x) on a batch of queries.

    Mirrors the chunked / state-conditioned action lookup in
    ConditionalSupportScorer so the sensitivity sweep uses the same scoring
    semantics as diagnostics 0-6 -- only the underlying geometry changes.
    """
    qs = _transform_query_states(raw_query_states, memory)
    qa = _transform_query_actions(raw_query_actions, memory)
    n_mem = memory.num_items
    sk = min(int(state_k), n_mem)
    ak = min(int(action_k), sk)

    # Chunked top-sk state distances (matches ConditionalSupportScorer).
    best_dists: Optional[torch.Tensor] = None
    best_idxs: Optional[torch.Tensor] = None
    for start in range(0, n_mem, chunk_size):
        end = min(start + chunk_size, n_mem)
        d = torch.cdist(qs, memory.state_features[start:end])
        vals, idx = torch.topk(d, k=min(sk, d.shape[1]), largest=False, dim=1)
        idx = idx + start
        if best_dists is None:
            best_dists, best_idxs = vals, idx
        else:
            merged_d = torch.cat([best_dists, vals], dim=1)
            merged_i = torch.cat([best_idxs, idx], dim=1)
            vals, order = torch.topk(merged_d, k=sk, largest=False, dim=1)
            best_dists = vals
            best_idxs = torch.gather(merged_i, 1, order)
    assert best_dists is not None and best_idxs is not None
    u_state = best_dists.mean(dim=1)

    # Conditional action distances among the state-NN's actions.
    neighbor_actions = memory.actions[best_idxs]
    action_dists = torch.linalg.vector_norm(neighbor_actions - qa.unsqueeze(1), dim=-1)
    ak_eff = min(ak, action_dists.shape[1])
    u_action = torch.topk(action_dists, k=ak_eff, largest=False, dim=1).values.mean(dim=1)
    return u_state, u_action


# --------------------------------------------------------------------------
# Q1-Q4 quadrant census.
# --------------------------------------------------------------------------

QUADRANT_LABELS: Tuple[str, str, str, str] = (
    "Q1_familiar_familiar",
    "Q2_familiar_unfamiliar",        # CSA KEY QUADRANT
    "Q3_unfamiliar_familiar",
    "Q4_unfamiliar_unfamiliar",
)


def _quadrant_assignment(
    u_state: torch.Tensor,
    u_action: torch.Tensor,
    state_familiar_quantile: float,
    action_unsupported_quantile: float,
) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
    """Return (state_familiar mask, action_familiar mask, state_thr, action_thr)."""
    state_thr = float(torch.quantile(u_state, state_familiar_quantile).item())
    action_thr = float(torch.quantile(u_action, action_unsupported_quantile).item())
    state_familiar = u_state <= state_thr
    action_familiar = u_action <= action_thr  # familiar = LOW U_action
    return state_familiar, action_familiar, state_thr, action_thr


def _quadrant_stats(
    quadrant_mask: torch.Tensor,
    is_failure: torch.Tensor,
    rollout_error: Optional[torch.Tensor],
) -> Dict[str, Any]:
    """Count + probability + failure rate + mean rollout error (NaN-safe)."""
    n = int(quadrant_mask.sum().item())
    if n == 0:
        return {"count": 0, "fraction": 0.0, "failure_rate": None, "mean_error": None}
    fraction = n / int(quadrant_mask.numel())
    failure_rate = float(is_failure[quadrant_mask].to(torch.float32).mean().item())
    mean_error: Optional[float] = None
    if rollout_error is not None:
        err_slice = rollout_error[quadrant_mask]
        finite = torch.isfinite(err_slice)
        if bool(finite.any()):
            mean_error = float(err_slice[finite].mean().item())
    return {
        "count": n,
        "fraction": fraction,
        "failure_rate": failure_rate,
        "mean_error": mean_error,
    }


def _slice_quadrant_distribution(
    slice_mask: torch.Tensor,
    state_familiar: torch.Tensor,
    action_familiar: torch.Tensor,
) -> Dict[str, Any]:
    """For one slice (e.g. failed-only or one rollout depth), report the
    probability mass per quadrant plus the slice count. The probabilities are
    computed with the *global* thresholds passed in via the masks so the four
    slices share a single reference distribution (diag-0 semantics)."""
    n = int(slice_mask.sum().item())
    if n == 0:
        return {
            "count": 0,
            "Q1_familiar_familiar": 0.0,
            "Q2_familiar_unfamiliar": 0.0,
            "Q3_unfamiliar_familiar": 0.0,
            "Q4_unfamiliar_unfamiliar": 0.0,
        }
    sf = state_familiar[slice_mask]
    af = action_familiar[slice_mask]
    return {
        "count": n,
        "Q1_familiar_familiar": float((sf & af).to(torch.float32).mean().item()),
        "Q2_familiar_unfamiliar": float((sf & ~af).to(torch.float32).mean().item()),
        "Q3_unfamiliar_familiar": float((~sf & af).to(torch.float32).mean().item()),
        "Q4_unfamiliar_unfamiliar": float((~sf & ~af).to(torch.float32).mean().item()),
    }


def quadrant_census(
    u_state: torch.Tensor,
    u_action: torch.Tensor,
    is_failure: torch.Tensor,
    rollout_error: Optional[torch.Tensor] = None,
    depth: Optional[torch.Tensor] = None,
    *,
    state_familiar_quantile: float = 0.50,
    action_unsupported_quantile: float = 0.75,
) -> Dict[str, Any]:
    """Full Q1-Q4 census across all decision steps, plus per-outcome slices,
    enrichment ratios, mean-error ratios vs Q1, and per-depth slices when
    ``depth`` is provided.
    """
    sf, af, state_thr, action_thr = _quadrant_assignment(
        u_state, u_action, state_familiar_quantile, action_unsupported_quantile
    )
    quadrants = {
        "Q1_familiar_familiar":   sf & af,
        "Q2_familiar_unfamiliar": sf & ~af,
        "Q3_unfamiliar_familiar": ~sf & af,
        "Q4_unfamiliar_unfamiliar": ~sf & ~af,
    }
    overall = {
        name: _quadrant_stats(mask, is_failure, rollout_error)
        for name, mask in quadrants.items()
    }

    succ_dist = _slice_quadrant_distribution(~is_failure, sf, af)
    fail_dist = _slice_quadrant_distribution(is_failure, sf, af)
    all_dist = _slice_quadrant_distribution(torch.ones_like(is_failure), sf, af)

    enrichment: Dict[str, Optional[float]] = {}
    for q in QUADRANT_LABELS:
        ps = succ_dist[q] if succ_dist["count"] > 0 else None
        pf = fail_dist[q] if fail_dist["count"] > 0 else None
        if ps is not None and pf is not None and ps > 0:
            enrichment[q] = pf / ps
        else:
            enrichment[q] = None

    # Mean-error ratios vs Q1 (the in-distribution baseline).
    q1_err = overall["Q1_familiar_familiar"]["mean_error"]
    error_ratio_vs_q1: Dict[str, Optional[float]] = {"Q1_familiar_familiar": 1.0 if q1_err else None}
    for q in QUADRANT_LABELS[1:]:
        qe = overall[q]["mean_error"]
        error_ratio_vs_q1[q] = qe / q1_err if (q1_err and qe) else None

    by_depth: Dict[str, Any] = {}
    if depth is not None:
        for d in sorted({int(x.item()) for x in depth}):
            d_mask = depth == d
            by_depth[f"depth_{d}"] = _slice_quadrant_distribution(d_mask, sf, af)
            # add per-quadrant error stats just for the depth slice
            d_q_stats = {
                name: _quadrant_stats(mask & d_mask, is_failure, rollout_error)
                for name, mask in quadrants.items()
            }
            by_depth[f"depth_{d}"]["per_quadrant_stats"] = d_q_stats

    # Dominant failure quadrant -- highest enrichment among the 4.
    dom_q = None
    dom_enrich = None
    for q in QUADRANT_LABELS:
        e = enrichment[q]
        if e is None:
            continue
        if dom_enrich is None or e > dom_enrich:
            dom_enrich = e
            dom_q = q

    return {
        "thresholds": {
            "state_familiar_quantile": state_familiar_quantile,
            "action_unsupported_quantile": action_unsupported_quantile,
            "state_threshold": state_thr,
            "action_threshold": action_thr,
        },
        "per_quadrant_all_decision_steps": overall,
        "by_outcome": {
            "all_decision_steps": all_dist,
            "successful_episodes": succ_dist,
            "failed_episodes": fail_dist,
        },
        "enrichment_failed_over_successful": enrichment,
        "mean_error_ratio_vs_Q1": error_ratio_vs_q1,
        "by_depth": by_depth,
        "dominant_failure_quadrant": dom_q,
        "dominant_failure_quadrant_enrichment": dom_enrich,
    }


# --------------------------------------------------------------------------
# Diagnostic 6 (scorer-validity probes) re-run inside each geometry variant.
# --------------------------------------------------------------------------

_NOISE_PEARSON_THRESHOLD = 0.10


def _pearson(x: torch.Tensor, y: torch.Tensor) -> Optional[float]:
    if x.numel() < 2 or y.numel() < 2:
        return None
    x = x.to(torch.float32) - x.to(torch.float32).mean()
    y = y.to(torch.float32) - y.to(torch.float32).mean()
    denom = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    if denom.item() <= 1e-12:
        return None
    return float((x * y).sum().item() / denom.item())


def _spearman(x: torch.Tensor, y: torch.Tensor) -> Optional[float]:
    if x.numel() < 2 or y.numel() < 2:
        return None

    def _rank(z: torch.Tensor) -> torch.Tensor:
        order = torch.argsort(z)
        ranks = torch.empty_like(order, dtype=torch.float32)
        ranks[order] = torch.arange(z.numel(), dtype=torch.float32)
        return ranks

    return _pearson(_rank(x), _rank(y))


def scorer_validity_probes(
    u_state: torch.Tensor,
    u_action: torch.Tensor,
    rollout_error: torch.Tensor,
    depth: torch.Tensor,
    *,
    beta: float = 1.0,
    state_familiar_quantile: float = 0.50,
) -> Dict[str, Any]:
    """Diagnostic-6-shaped per-modality, per-rollout-depth Pearson + Spearman
    of (U_state, U_action, U_SA) against rollout_error, conditioned on the
    state-familiar subset (same quantile as diag-2 primary). Returns a verdict
    block that the per-variant report can use to flag PCA-style sign artifacts.

    Use this on the rollout-only subset (depth >= 1, finite rollout_error)
    so it's directly comparable to the analysis.py diag-6 output.
    """
    finite = torch.isfinite(rollout_error)
    rollout_mask = (depth >= 1) & finite
    if int(rollout_mask.sum().item()) < 2:
        return {"status": "not_computed", "reason": "fewer than 2 imagined-rollout records with finite error"}

    us = u_state[rollout_mask]
    ua = u_action[rollout_mask]
    ucsa = us + beta * ua
    err = rollout_error[rollout_mask]
    d_sub = depth[rollout_mask]

    state_thr = float(torch.quantile(us, state_familiar_quantile).item())
    familiar_mask = us <= state_thr

    def _corrs(s, a, c, e):
        if s.numel() < 2:
            return {
                "pearson_state": None, "spearman_state": None,
                "pearson_action": None, "spearman_action": None,
                "pearson_csa": None, "spearman_csa": None,
            }
        return {
            "pearson_state": _pearson(s, e),
            "spearman_state": _spearman(s, e),
            "pearson_action": _pearson(a, e),
            "spearman_action": _spearman(a, e),
            "pearson_csa": _pearson(c, e),
            "spearman_csa": _spearman(c, e),
        }

    unconditioned = _corrs(us, ua, ucsa, err)
    if int(familiar_mask.sum().item()) >= 2:
        conditioned = _corrs(us[familiar_mask], ua[familiar_mask], ucsa[familiar_mask], err[familiar_mask])
    else:
        conditioned = _corrs(torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0))

    by_depth: Dict[str, Any] = {}
    for d in sorted({int(x.item()) for x in d_sub}):
        d_mask = (d_sub == d) & familiar_mask
        count = int(d_mask.sum().item())
        if count < 2:
            by_depth[f"depth_{d}"] = {"count": count}
            continue
        by_depth[f"depth_{d}"] = {
            "count": count,
            **_corrs(us[d_mask], ua[d_mask], ucsa[d_mask], err[d_mask]),
        }

    def _summary(metric_key: str) -> Dict[str, Any]:
        vals = [
            d_data[metric_key]
            for d_data in by_depth.values()
            if d_data.get(metric_key) is not None
        ]
        if not vals:
            return {
                "depths_with_data": 0,
                "all_positive": None,
                "all_negative": None,
                "all_same_sign": None,
                "values": [],
            }
        return {
            "depths_with_data": len(vals),
            "all_positive": bool(all(v > 0 for v in vals)),
            "all_negative": bool(all(v < 0 for v in vals)),
            "all_same_sign": bool(all(v > 0 for v in vals) or all(v < 0 for v in vals)),
            "values": [round(float(v), 4) for v in vals],
        }

    sign_consistency = {
        "pearson_state": _summary("pearson_state"),
        "pearson_action": _summary("pearson_action"),
        "pearson_csa": _summary("pearson_csa"),
    }

    def _wrong(mkey: str) -> bool:
        s = sign_consistency[mkey]
        return bool(s["depths_with_data"] >= 2 and s["all_negative"])

    noise_only = True
    for mkey in ("pearson_state", "pearson_action", "pearson_csa"):
        for v in sign_consistency[mkey]["values"]:
            if abs(v) >= _NOISE_PEARSON_THRESHOLD:
                noise_only = False
                break
        if not noise_only:
            break

    return {
        "status": "computed",
        "state_familiar_quantile": state_familiar_quantile,
        "state_threshold": state_thr,
        "subset_size_familiar": int(familiar_mask.sum().item()),
        "subset_size_total": int(us.numel()),
        "unconditioned": unconditioned,
        "conditioned_overall": conditioned,
        "by_depth": by_depth,
        "sign_consistency": sign_consistency,
        "verdict": {
            "scorer_wrong_signed_csa": _wrong("pearson_csa"),
            "scorer_wrong_signed_action": _wrong("pearson_action"),
            "scorer_wrong_signed_state": _wrong("pearson_state"),
            "scorer_noise_only": noise_only,
            "noise_pearson_threshold": _NOISE_PEARSON_THRESHOLD,
        },
    }


# --------------------------------------------------------------------------
# Top-level: run the sweep over variants for one env's data.
# --------------------------------------------------------------------------

def _gather_queries_from_dumps(
    dumps: Sequence[Dict[str, Any]],
) -> Dict[str, torch.Tensor]:
    """Build flat per-query tensors (state, action, episode_success, depth,
    rollout_error) from raw dump payloads -- mirrors analysis.score_dumps but
    keeps everything in raw (pre-scorer) space."""
    states: List[torch.Tensor] = []
    actions: List[torch.Tensor] = []
    succ: List[int] = []
    depth: List[int] = []
    err: List[float] = []
    for dump in dumps:
        ep_succ = int(dump.get("episode_success", 0))
        for step in dump.get("steps", []):
            sel_actions = step.get("selected_actions")
            decision_state = step.get("decision_state")
            imagined = step.get("imagined_states")
            real = step.get("real_states")
            if sel_actions is None or decision_state is None:
                continue
            n_actions = int(sel_actions.shape[0]) if sel_actions.ndim == 2 else 0
            n_imag = int(imagined.shape[0]) if (imagined is not None and imagined.ndim == 2) else 0
            n_real = int(real.shape[0]) if (real is not None and real.ndim == 2) else 0
            for d in range(n_actions):
                if d == 0:
                    state_vec = decision_state
                    err_d = 0.0
                else:
                    if d - 1 >= n_imag:
                        break
                    state_vec = imagined[d - 1]
                    if d - 1 < n_real:
                        err_d = float(((imagined[d - 1] - real[d - 1]) ** 2).mean().item())
                    else:
                        err_d = float("nan")
                states.append(state_vec.reshape(-1).to(torch.float32))
                actions.append(sel_actions[d].reshape(-1).to(torch.float32))
                succ.append(ep_succ)
                depth.append(d)
                err.append(err_d)
    if not states:
        return {
            "raw_states": torch.empty(0, 0),
            "raw_actions": torch.empty(0, 0),
            "is_failure": torch.empty(0, dtype=torch.bool),
            "depth": torch.empty(0, dtype=torch.long),
            "rollout_error": torch.empty(0),
        }
    return {
        "raw_states": torch.stack(states),
        "raw_actions": torch.stack(actions),
        "is_failure": torch.tensor(succ).bool().logical_not(),  # failure = success <= 0
        "depth": torch.tensor(depth, dtype=torch.long),
        "rollout_error": torch.tensor(err, dtype=torch.float32),
    }


def run_geometry_sensitivity(
    raw_support_states: torch.Tensor,
    raw_support_actions: torch.Tensor,
    dumps: Sequence[Dict[str, Any]],
    *,
    variants: Sequence[GeometryVariant] = DEFAULT_VARIANTS,
    state_k: int = 64,
    action_k: int = 64,
    state_familiar_quantile: float = 0.50,
    action_unsupported_quantile: float = 0.75,
    pca_fit_samples: Optional[int] = 50_000,
    device: str = "cpu",
) -> Dict[str, Any]:
    """For one env: rebuild support memory under each ``variant``, re-score
    every (episode, replan, depth) query from ``dumps``, and emit the Q1-Q4
    census per variant. Returns a single dict ready to JSON-dump."""
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
        "raw_state_dim": int(raw_support_states.shape[1]) if raw_support_states.ndim == 2 else None,
        "raw_action_dim": int(raw_support_actions.shape[1]) if raw_support_actions.ndim == 2 else None,
        "thresholds": {
            "state_familiar_quantile": state_familiar_quantile,
            "action_unsupported_quantile": action_unsupported_quantile,
        },
        "scorer": {"state_k": state_k, "action_k": action_k},
        "primary_variant_names": list(PRIMARY_VARIANT_NAMES),
        "quadrant_labels": list(QUADRANT_LABELS),
        "quadrant_interpretation_note": (
            "Q1 enrichment = failures inside the familiar region (support penalty unlikely to help). "
            "Q2 enrichment = state-familiar but action-unfamiliar -> supports the CSA-MPC premise. "
            "Q3 enrichment = state OOD only. "
            "Q4 enrichment = generic OOD / hallucination, not CSA-specific."
        ),
        "variants": {},
    }

    if n_queries == 0:
        out["status"] = "not_computed"
        out["reason"] = "no queries from dumps"
        return out

    raw_support_states = raw_support_states.to(device=device, dtype=torch.float32)
    raw_support_actions = raw_support_actions.to(device=device, dtype=torch.float32)
    q_states = queries["raw_states"].to(device=device, dtype=torch.float32)
    q_actions = queries["raw_actions"].to(device=device, dtype=torch.float32)
    is_failure = queries["is_failure"].to(device=device)
    depth = queries["depth"].to(device=device)
    rollout_error = queries["rollout_error"].to(device=device)

    decision_mask = depth == 0
    rollout_mask = depth > 0

    for variant in variants:
        memory = build_calibrated_memory(
            raw_support_states, raw_support_actions, variant,
            pca_fit_samples=pca_fit_samples,
        )
        u_state, u_action = score_queries(
            q_states, q_actions, memory, state_k=state_k, action_k=action_k,
        )

        # Census #1: ALL queries (decision + rollout). The full distribution.
        full = quadrant_census(
            u_state, u_action, is_failure, rollout_error, depth=depth,
            state_familiar_quantile=state_familiar_quantile,
            action_unsupported_quantile=action_unsupported_quantile,
        )

        # Census #2: decision-step queries only (depth=0) -- the diag-0
        # comparable census on the executed plan's selected first action.
        decision = quadrant_census(
            u_state[decision_mask], u_action[decision_mask],
            is_failure[decision_mask], rollout_error[decision_mask],
            depth=None,
            state_familiar_quantile=state_familiar_quantile,
            action_unsupported_quantile=action_unsupported_quantile,
        )

        # Census #3: rollout-only queries (depth>=1) -- the in-imagination
        # support distribution along the executed plan.
        if int(rollout_mask.sum().item()) >= 2:
            rollout = quadrant_census(
                u_state[rollout_mask], u_action[rollout_mask],
                is_failure[rollout_mask], rollout_error[rollout_mask],
                depth=depth[rollout_mask],
                state_familiar_quantile=state_familiar_quantile,
                action_unsupported_quantile=action_unsupported_quantile,
            )
        else:
            rollout = {"status": "not_computed", "reason": "no rollout records"}

        # Diag-6-style scorer validity probes under THIS variant's geometry.
        # Runs on the rollout subset (depth>=1, finite error); reports
        # per-modality Pearson/Spearman per depth + a sign-consistency
        # verdict. Lets us cross-check whether the diag-6 wrong-signed CSA
        # on Wall was a PCA-64 artifact.
        validity_probes = scorer_validity_probes(
            u_state=u_state, u_action=u_action,
            rollout_error=rollout_error, depth=depth,
            state_familiar_quantile=state_familiar_quantile,
        )

        out["variants"][variant.name] = {
            "variant": {
                "name": variant.name,
                "pca_dim": variant.pca_dim,
                "whiten": bool(variant.whiten),
                "calibration": variant.calibration,
                "is_primary_no_pca_variant": variant.name in PRIMARY_VARIANT_NAMES,
                "description": variant.describe(),
            },
            "memory_summary": {
                "num_items": memory.num_items,
                "state_dim_after_geometry": int(memory.state_features.shape[1]),
                "state_scale": memory.state_scale,
                "action_scale": memory.action_scale,
            },
            "census_all_queries": full,
            "census_decision_steps_only": decision,
            "census_rollout_only": rollout,
            "scorer_validity_probes": validity_probes,
        }

    out["status"] = "computed"
    return out
