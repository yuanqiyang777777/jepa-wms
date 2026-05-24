# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
"""Local residual-calibration scorer for the CSA-MPC reframing (memo v3 / Exp C).

The data-coverage scorer ``U_action|x`` measures how *near* the training data a
``(x, a)`` query is. The residual scorer measures something different: the
*model error* at that query, estimated from real validation transitions.

Build pipeline (mirrors M_real, but on the held-out val split):

    M_resid = { (x_i, a_i, e_i) }
        where e_i = mean( (f_theta(x_i, a_i) - x_{i+1})**2 )  # pooled-latent MSE,
                                                                # SAME metric the
                                                                # diagnostic dumps
                                                                # use for rollout_error

(See ``experiments/scripts/csa/encode_residual_data.py``.)

Query path (online or offline analysis):

    r(q) = mean( e_i  :  (x_i, a_i) in  state-then-action kNN_k(q) over M_resid )

The kNN structure mirrors :class:`ConditionalSupportScorer`: first take the
state_k nearest neighbours in calibrated state space, then narrow to the
action_k nearest by action distance among those, and return the mean residual
``e_i`` over those final neighbours. This keeps ``r(q)`` genuinely q-dependent
(not only x-dependent) and inherits the geometry / calibration choices the
geometry-sensitivity sweep already exposes (raw, PCA-64-whitened, etc).

NOTE on what r(q) is and is NOT:

* r(q) is an empirical estimate of latent-rollout prediction error at q,
  conditioned on local val transitions. It is NOT a calibrated uncertainty
  estimate, NOT a probability, NOT a safety bound.
* r(q) is bounded above by the worst residual in the local neighbourhood and
  bounded below by the best -- it is a *local average*, not a worst-case bound.
* r(q) inherits the val split's coverage limits: queries far from any val
  transition fall back to whichever neighbours happen to be nearest, which
  may not be informative. The correlation analysis (residual_correlation.py)
  stratifies by ``U_state`` quantile to surface this.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch

from evals.simu_env_planning.planning.planning.csa.feature_adapter import (
    PCAWhiteningAdapter,
)
from evals.simu_env_planning.planning.planning.csa.geometry_sensitivity import (
    CalibratedMemory,
    GeometryVariant,
    _reciprocal_rms,
    _transform_query_actions,
    _transform_query_states,
)


# --------------------------------------------------------------------------
# Residual memory: (calibrated state, calibrated action, raw residual error).
# --------------------------------------------------------------------------

@dataclass
class CalibratedResidualMemory:
    """One env's residual memory under one geometry variant.

    Same calibration pipeline as :class:`CalibratedMemory` but additionally
    carries the per-transition residual error ``e_i`` (the value r(q) averages
    over its kNN).
    """
    variant: GeometryVariant
    state_features: torch.Tensor   # (N, D_state) post-geometry, post-calibration
    actions: torch.Tensor          # (N, D_action) post-calibration
    residual_errors: torch.Tensor  # (N,) raw e_i values
    state_adapter: Optional[PCAWhiteningAdapter]
    state_scale: float
    action_scale: float

    @property
    def num_items(self) -> int:
        return int(self.state_features.shape[0])


def build_calibrated_residual_memory(
    raw_states: torch.Tensor,
    raw_actions: torch.Tensor,
    raw_residual_errors: torch.Tensor,
    variant: GeometryVariant,
    pca_fit_samples: Optional[int] = None,
) -> CalibratedResidualMemory:
    """Build a residual memory for ``variant`` from raw aligned tensors.

    The state + action calibration pipeline matches
    :func:`geometry_sensitivity.build_calibrated_memory` exactly; the only
    difference is we additionally store the per-transition residual errors.
    Re-using the same PCA / scaling means queries computed via either scorer
    live in identical metric spaces -- so the (r vs U_action|x) correlation
    comparison is apples-to-apples.
    """
    raw_states = raw_states.to(torch.float32).contiguous()
    raw_actions = raw_actions.to(torch.float32).contiguous()
    raw_residual_errors = raw_residual_errors.to(torch.float32).contiguous()

    n_states = int(raw_states.shape[0])
    n_actions = int(raw_actions.shape[0])
    n_errors = int(raw_residual_errors.shape[0])
    if not (n_states == n_actions == n_errors):
        raise ValueError(
            f"raw_states ({n_states}) / raw_actions ({n_actions}) / raw_residual_errors "
            f"({n_errors}) row-count mismatch; all three must be aligned per-transition."
        )

    if variant.pca_dim is None:
        state_adapter = None
        state_proj = raw_states
    else:
        fit_x = raw_states
        if pca_fit_samples is not None and pca_fit_samples < n_states:
            idx = torch.randperm(n_states)[:pca_fit_samples]
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

    return CalibratedResidualMemory(
        variant=variant,
        state_features=(state_proj * state_scale).contiguous(),
        actions=(raw_actions * action_scale).contiguous(),
        residual_errors=raw_residual_errors.clone(),
        state_adapter=state_adapter,
        state_scale=state_scale,
        action_scale=action_scale,
    )


# --------------------------------------------------------------------------
# Scoring: r(q) via state-then-action kNN, returning the mean residual.
# --------------------------------------------------------------------------

def _state_topk(
    query_states: torch.Tensor,
    memory_states: torch.Tensor,
    k: int,
    chunk_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Chunked top-k state distances. Mirrors ConditionalSupportScorer._state_scores."""
    n_mem = int(memory_states.shape[0])
    k_eff = min(int(k), n_mem)
    best_dists: Optional[torch.Tensor] = None
    best_idxs: Optional[torch.Tensor] = None
    for start in range(0, n_mem, chunk_size):
        end = min(start + chunk_size, n_mem)
        d = torch.cdist(query_states, memory_states[start:end])
        vals, idx = torch.topk(d, k=min(k_eff, d.shape[1]), largest=False, dim=1)
        idx = idx + start
        if best_dists is None:
            best_dists, best_idxs = vals, idx
        else:
            merged_d = torch.cat([best_dists, vals], dim=1)
            merged_i = torch.cat([best_idxs, idx], dim=1)
            vals, order = torch.topk(merged_d, k=k_eff, largest=False, dim=1)
            best_dists = vals
            best_idxs = torch.gather(merged_i, 1, order)
    assert best_dists is not None and best_idxs is not None
    return best_dists, best_idxs


def score_residual_queries(
    raw_query_states: torch.Tensor,
    raw_query_actions: torch.Tensor,
    memory: CalibratedResidualMemory,
    state_k: int = 64,
    action_k: int = 8,
    chunk_size: int = 8192,
) -> torch.Tensor:
    """Compute r(q) for a batch of queries against a calibrated residual memory.

    Same state-then-action kNN structure as :class:`ConditionalSupportScorer`,
    but the final reduction returns the mean residual error ``e_i`` over the
    inner action-kNN bucket. Default ``action_k=8`` (smaller than the
    coverage scorer's 64) so the residual average isn't dominated by far
    neighbours in action space; the correlation analysis can sweep this.

    Returns a Tensor of shape ``(B,)`` with one r(q) per query.
    """
    qs = _transform_query_states(raw_query_states, memory)  # type: ignore[arg-type]
    qa = _transform_query_actions(raw_query_actions, memory)  # type: ignore[arg-type]

    _, state_nn_idx = _state_topk(
        qs, memory.state_features, k=state_k, chunk_size=chunk_size
    )  # (B, state_k)

    # Restrict to the action-kNN among the state-NN, then average e_i.
    neighbour_actions = memory.actions[state_nn_idx]              # (B, state_k, A)
    action_dists = torch.linalg.vector_norm(
        neighbour_actions - qa.unsqueeze(1), dim=-1
    )                                                              # (B, state_k)
    ak = min(int(action_k), action_dists.shape[1])
    action_topk_vals, action_topk_idx = torch.topk(
        action_dists, k=ak, largest=False, dim=1
    )                                                              # both (B, ak)

    # Gather the residuals at the surviving (state-then-action) neighbours.
    neighbour_residuals = memory.residual_errors[state_nn_idx]    # (B, state_k)
    final_residuals = torch.gather(neighbour_residuals, 1, action_topk_idx)  # (B, ak)
    return final_residuals.mean(dim=1)                            # (B,)


# --------------------------------------------------------------------------
# Convenience: combined scoring (r, U_state, U_action|x) on a single memory pair.
# --------------------------------------------------------------------------

def score_residual_and_support(
    raw_query_states: torch.Tensor,
    raw_query_actions: torch.Tensor,
    support_memory: "CalibratedMemory",
    residual_memory: CalibratedResidualMemory,
    state_k: int = 64,
    action_k_support: int = 64,
    action_k_residual: int = 8,
    chunk_size: int = 8192,
) -> dict:
    """Compute U_state, U_action|x, U_SA, and r(q) for the same queries.

    The support and residual memories must be calibrated under the SAME
    geometry variant -- otherwise the comparison "r better than U_action|x"
    crosses scorer geometries and isn't meaningful. We assert that here.

    Returns a dict of four ``(B,)`` tensors plus the variant name actually
    used. ``U_SA = U_state + U_action|x`` (beta=1.0 by convention; the
    diag-6 module uses the same).
    """
    sv = support_memory.variant
    rv = residual_memory.variant
    if (sv.name != rv.name
            or sv.pca_dim != rv.pca_dim
            or bool(sv.whiten) != bool(rv.whiten)
            or sv.calibration != rv.calibration):
        raise ValueError(
            "support_memory and residual_memory must be built under the SAME variant; "
            f"got support={sv.describe()!r}, residual={rv.describe()!r}"
        )

    # U_state, U_action|x via the geometry-sensitivity scorer for parity.
    from evals.simu_env_planning.planning.planning.csa.geometry_sensitivity import (
        score_queries as _score_support,
    )
    u_state, u_action = _score_support(
        raw_query_states, raw_query_actions, support_memory,
        state_k=state_k, action_k=action_k_support, chunk_size=chunk_size,
    )
    r_q = score_residual_queries(
        raw_query_states, raw_query_actions, residual_memory,
        state_k=state_k, action_k=action_k_residual, chunk_size=chunk_size,
    )
    return {
        "u_state": u_state,
        "u_action": u_action,
        "u_sa": u_state + u_action,  # beta=1.0 (matches diag-6)
        "r": r_q,
        "variant_name": sv.name,
    }


# Re-export the variant list so callers can iterate without two imports.
def default_variants_for_correlation() -> Tuple[GeometryVariant, ...]:
    """Return the two variants used by the C correlation analysis:
    raw (primary, no PCA) and PCA-64 (sensitivity check). The full 6-variant
    sweep is reserved for the geometry-sensitivity diagnostic, not the C
    correlation comparison."""
    from evals.simu_env_planning.planning.planning.csa.geometry_sensitivity import (
        DEFAULT_VARIANTS,
    )
    by_name = {v.name: v for v in DEFAULT_VARIANTS}
    return (by_name["raw_recipRMS"], by_name["pca64_white_recipRMS"])
