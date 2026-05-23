# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
"""Offline CSA-MPC diagnostics 0-4 (memo sas_mpc_research_memo_zh.tex section 6).

This module consumes the per-episode ``.pt`` dumps written by
``DiagnosticRecorder`` during instrumented vanilla CEM/MPC eval, runs the
``ConditionalSupportScorer`` along each imagined rollout, and computes:

* Diagnostic 0  -- failure-mode census / Go-No-Go gate.
* Diagnostic 1  -- exploitation gap ``G = C_goal_real - C_goal_pred``
                   (both terms in the planner's objective units).
* Diagnostic 2  -- support-score vs rollout-error correlation, conditioned on
                   a state-familiar subset, swept over the familiar threshold.
* Diagnostic 3-H1 -- four-bin ``(U_state hi/lo) x (U_SA hi/lo)`` matrix.
* Diagnostic 4  -- false-negative rate (support low but rollout error high) as a
                   function of imagined-rollout depth.

Everything runs offline, so the support memory, k, beta and every diagnostic
threshold can be re-tuned without re-running eval.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import torch

from evals.simu_env_planning.planning.planning.csa.diagnostic_recorder import load_episode_dumps
from evals.simu_env_planning.planning.planning.csa.support_memory import SupportMemory
from evals.simu_env_planning.planning.planning.csa.support_scorer import ConditionalSupportScorer


# --------------------------------------------------------------------------
# Scoring: turn per-episode dumps into a flat per-(episode, step, depth) table.
# --------------------------------------------------------------------------

def score_dumps(
    dumps: List[Dict[str, Any]],
    memory: SupportMemory,
    state_k: int = 64,
    action_k: int = 64,
    beta: float = 1.0,
    chunk_size: int = 8192,
) -> List[Dict[str, Any]]:
    """Score every (episode, replan step, rollout depth) query against ``memory``.

    Depth 0 is the strictly-reliable t=0 query (real observation). Depth d>=1
    queries the imagined state ``x_hat_d`` the planner committed to; its
    ``rollout_error`` is the pooled-latent MSE between that imagined state and
    the real state the environment actually reached (NaN if the episode
    terminated before depth d).
    """
    scorer = ConditionalSupportScorer(
        memory=memory, state_k=state_k, action_k=action_k, beta=beta, chunk_size=chunk_size
    )
    states: List[torch.Tensor] = []
    actions: List[torch.Tensor] = []
    meta: List[Dict[str, Any]] = []

    for dump in dumps:
        episode = int(dump.get("episode_id", -1))
        episode_success = int(dump.get("episode_success", 0))
        for step in dump.get("steps", []):
            replan_idx = int(step.get("replan_idx", -1))
            decision_state = step["decision_state"]
            imagined = step["imagined_states"]
            real = step["real_states"]
            sel_actions = step["selected_actions"]
            n_actions = int(sel_actions.shape[0]) if sel_actions is not None and sel_actions.ndim == 2 else 0
            n_imagined = int(imagined.shape[0]) if imagined is not None and imagined.ndim == 2 else 0
            n_real = int(real.shape[0]) if real is not None and real.ndim == 2 else 0
            pred_cost = float(step.get("predicted_terminal_cost", float("nan")))
            real_cost = float(step.get("real_terminal_cost", float("nan")))

            for depth in range(n_actions):
                if depth == 0:
                    state_vec = decision_state
                    rollout_error = 0.0
                else:
                    if depth - 1 >= n_imagined:
                        break
                    state_vec = imagined[depth - 1]
                    if depth - 1 < n_real:
                        rollout_error = float(((imagined[depth - 1] - real[depth - 1]) ** 2).mean().item())
                    else:
                        rollout_error = float("nan")
                if state_vec is None:
                    continue
                states.append(state_vec.reshape(-1))
                actions.append(sel_actions[depth].reshape(-1))
                meta.append(
                    {
                        "episode": episode,
                        "episode_success": episode_success,
                        "replan_idx": replan_idx,
                        "depth": depth,
                        "is_decision_step": depth == 0,
                        "rollout_error": rollout_error,
                        "predicted_terminal_cost": pred_cost,
                        "real_terminal_cost": real_cost,
                        "step_success": int(step.get("step_success", 0)),
                    }
                )

    if not meta:
        return []

    scores = scorer.score(torch.stack(states), torch.stack(actions), raw_state=True)
    state_score = scores["state_score"].detach().cpu()
    action_score = scores["action_score"].detach().cpu()
    csa_score = scores["csa_score"].detach().cpu()
    for i, record in enumerate(meta):
        record["state_score"] = float(state_score[i].item())
        record["action_score"] = float(action_score[i].item())
        record["csa_score"] = float(csa_score[i].item())
    return meta


# --------------------------------------------------------------------------
# Top-level analysis.
# --------------------------------------------------------------------------

def analyze(
    dumps: List[Dict[str, Any]],
    memory: SupportMemory,
    state_k: int = 64,
    action_k: int = 64,
    beta: float = 1.0,
    chunk_size: int = 8192,
    gate_min_ratio: float = 0.10,
    state_familiar_quantile: float = 0.50,
    action_unsupported_quantile: float = 0.75,
) -> Dict[str, Any]:
    """Run diagnostics 0-4 over ``dumps`` scored against ``memory``."""
    records = score_dumps(
        dumps, memory, state_k=state_k, action_k=action_k, beta=beta, chunk_size=chunk_size
    )
    env = dumps[0].get("env", "") if dumps else ""
    num_failed = sum(1 for d in dumps if int(d.get("episode_success", 0)) <= 0)

    report: Dict[str, Any] = {
        "env": env,
        "num_episodes": len(dumps),
        "num_failed_episodes": num_failed,
        "num_scored_queries": len(records),
        "support_memory": {
            "num_items": int(getattr(memory, "num_items", 0)),
            "pca_dim": int(memory.metadata.get("pca_dim", -1)) if memory.metadata else -1,
            "state_reduction": memory.metadata.get("state_reduction", "") if memory.metadata else "",
        },
        "scorer": {"state_k": state_k, "action_k": action_k, "beta": beta},
        "thresholds": {
            "state_familiar_quantile": state_familiar_quantile,
            "action_unsupported_quantile": action_unsupported_quantile,
            "gate_min_ratio": gate_min_ratio,
        },
    }
    if not records:
        report.update(_empty_diagnostics(gate_min_ratio))
        return report

    diag0 = _diagnostic_0(records, state_familiar_quantile, action_unsupported_quantile)
    diag1 = _diagnostic_1(dumps)
    diag2 = _diagnostic_2(records)
    diag3 = _diagnostic_3(records, state_familiar_quantile)
    diag4 = _diagnostic_4(records)

    report["diagnostic_0_failure_mode"] = diag0
    report["diagnostic_1_exploitation_gap"] = diag1
    report["diagnostic_2_conditioned_correlation"] = diag2
    report["diagnostic_3_buckets"] = diag3
    report["diagnostic_4_false_negative_by_depth"] = diag4
    report["go_no_go"] = _go_no_go(diag0, diag1, diag2, diag3, gate_min_ratio)
    return report


def analyze_dump_dir(
    dump_dir: str | Path,
    memory_path: str | Path,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Convenience: load every episode dump under ``dump_dir`` + the support
    memory at ``memory_path``, then run :func:`analyze`."""
    dumps = load_episode_dumps(dump_dir)
    memory = SupportMemory.load(memory_path)
    return analyze(dumps, memory, **kwargs)


def write_report_json(report: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")


# --------------------------------------------------------------------------
# Diagnostic 0 -- failure-mode census / gate.
# --------------------------------------------------------------------------

def _diagnostic_0(
    records: List[Dict[str, Any]],
    state_familiar_quantile: float,
    action_unsupported_quantile: float,
) -> Dict[str, Any]:
    """Census of the 'familiar state + unsupported action' failure mode.

    Thresholds are defined on *all* decision steps (the unconditional reference
    distribution); the census then reports what fraction of failed-episode
    decision steps land in the quadrant, alongside the successful-episode and
    overall base rates so enrichment -- not just prevalence -- is visible.
    """
    decision = [r for r in records if r["is_decision_step"]]
    if not decision:
        return {"status": "not_computed", "reason": "no decision-step records"}

    state_scores = torch.tensor([r["state_score"] for r in decision])
    action_scores = torch.tensor([r["action_score"] for r in decision])
    failed = torch.tensor([r["episode_success"] <= 0 for r in decision])

    sweep = []
    for s_q in (0.40, 0.50, 0.60):
        for a_q in (0.60, 0.75, 0.90):
            sweep.append(
                _quadrant_census(state_scores, action_scores, failed, s_q, a_q)
            )
    primary = _quadrant_census(
        state_scores, action_scores, failed, state_familiar_quantile, action_unsupported_quantile
    )
    return {
        "status": "computed",
        "num_decision_steps": len(decision),
        "num_failed_decision_steps": int(failed.sum().item()),
        "primary": primary,
        "threshold_sweep": sweep,
    }


def _quadrant_census(
    state_scores: torch.Tensor,
    action_scores: torch.Tensor,
    failed: torch.Tensor,
    state_q: float,
    action_q: float,
) -> Dict[str, Any]:
    state_thr = torch.quantile(state_scores, state_q).item()
    action_thr = torch.quantile(action_scores, action_q).item()
    in_quadrant = (state_scores <= state_thr) & (action_scores >= action_thr)
    failed_mask = failed.bool()
    succ_mask = ~failed_mask
    n_failed = int(failed_mask.sum().item())
    n_succ = int(succ_mask.sum().item())
    return {
        "state_familiar_quantile": state_q,
        "action_unsupported_quantile": action_q,
        "state_threshold": state_thr,
        "action_threshold": action_thr,
        "ratio_failed": _safe_ratio((in_quadrant & failed_mask).sum().item(), n_failed),
        "ratio_successful": _safe_ratio((in_quadrant & succ_mask).sum().item(), n_succ),
        "ratio_all": _safe_ratio(in_quadrant.sum().item(), in_quadrant.numel()),
        "num_failed_in_quadrant": int((in_quadrant & failed_mask).sum().item()),
    }


# --------------------------------------------------------------------------
# Diagnostic 1 -- exploitation gap (same units).
# --------------------------------------------------------------------------

def _diagnostic_1(dumps: List[Dict[str, Any]]) -> Dict[str, Any]:
    """G = C_goal_real - C_goal_pred per replan step; positive => model was
    over-optimistic (predicted goal cost lower than the real executed cost)."""
    gaps = []
    for dump in dumps:
        for step in dump.get("steps", []):
            pred = step.get("predicted_terminal_cost", float("nan"))
            real = step.get("real_terminal_cost", float("nan"))
            if pred is None or real is None:
                continue
            gap = float(real) - float(pred)
            if gap == gap:  # filter NaN
                gaps.append(gap)
    if not gaps:
        return {"status": "not_computed", "reason": "no finite predicted/real terminal costs"}
    t = torch.tensor(gaps, dtype=torch.float32)
    return {
        "status": "computed",
        "num_replan_steps": int(t.numel()),
        "mean_gap": float(t.mean().item()),
        "median_gap": float(torch.median(t).item()),
        "std_gap": float(t.std(unbiased=False).item()),
        "p90_gap": float(torch.quantile(t, 0.90).item()),
        "fraction_real_worse_than_pred": float((t > 0).to(torch.float32).mean().item()),
    }


# --------------------------------------------------------------------------
# Diagnostic 2 -- conditioned support/error correlation.
# --------------------------------------------------------------------------

def _diagnostic_2(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Within state-familiar subsets, correlate action/CSA support with the real
    rollout error. Swept over the familiar threshold to expose threshold
    sensitivity (the conclusion must not be an artefact of one cutoff)."""
    rollout = [r for r in records if not r["is_decision_step"] and _finite(r["rollout_error"])]
    if len(rollout) < 2:
        return {"status": "not_computed", "reason": "fewer than 2 imagined-rollout records with finite error"}
    state_scores = torch.tensor([r["state_score"] for r in rollout])
    action_scores = torch.tensor([r["action_score"] for r in rollout])
    csa_scores = torch.tensor([r["csa_score"] for r in rollout])
    errors = torch.tensor([r["rollout_error"] for r in rollout])

    by_threshold = []
    for q in (0.25, 0.40, 0.50, 0.60, 0.75):
        thr = torch.quantile(state_scores, q).item()
        mask = state_scores <= thr
        if int(mask.sum().item()) < 2:
            by_threshold.append({"state_familiar_quantile": q, "subset_size": int(mask.sum().item()),
                                  "pearson_action": None, "spearman_action": None, "pearson_csa": None})
            continue
        by_threshold.append(
            {
                "state_familiar_quantile": q,
                "state_threshold": thr,
                "subset_size": int(mask.sum().item()),
                "pearson_action": _pearson(action_scores[mask], errors[mask]),
                "spearman_action": _spearman(action_scores[mask], errors[mask]),
                "pearson_csa": _pearson(csa_scores[mask], errors[mask]),
            }
        )
    return {
        "status": "computed",
        "num_rollout_records": len(rollout),
        "unconditioned_pearson_action": _pearson(action_scores, errors),
        "by_threshold": by_threshold,
    }


# --------------------------------------------------------------------------
# Diagnostic 3-H1 -- four-bin (U_state, U_SA) matrix.
# --------------------------------------------------------------------------

def _diagnostic_3(records: List[Dict[str, Any]], state_familiar_quantile: float) -> Dict[str, Any]:
    rollout = [r for r in records if not r["is_decision_step"] and _finite(r["rollout_error"])]
    if not rollout:
        return {"status": "not_computed", "reason": "no imagined-rollout records with finite error"}
    state_scores = torch.tensor([r["state_score"] for r in rollout])
    csa_scores = torch.tensor([r["csa_score"] for r in rollout])
    errors = torch.tensor([r["rollout_error"] for r in rollout])
    failed = torch.tensor([r["episode_success"] <= 0 for r in rollout])

    state_thr = torch.quantile(state_scores, state_familiar_quantile).item()
    csa_thr = torch.quantile(csa_scores, 0.50).item()
    state_low = state_scores <= state_thr
    csa_high = csa_scores >= csa_thr

    buckets = {
        "state_low_csa_low": _summarize_mask(state_low & ~csa_high, errors, failed),
        "state_low_csa_high": _summarize_mask(state_low & csa_high, errors, failed),
        "state_high_csa_low": _summarize_mask(~state_low & ~csa_high, errors, failed),
        "state_high_csa_high": _summarize_mask(~state_low & csa_high, errors, failed),
    }
    key = buckets["state_low_csa_high"]
    baseline = buckets["state_low_csa_low"]
    h1_supported = (
        key["mean_error"] is not None
        and baseline["mean_error"] is not None
        and key["mean_error"] > baseline["mean_error"]
    )
    return {
        "status": "computed",
        "state_threshold": state_thr,
        "csa_threshold": csa_thr,
        "buckets": buckets,
        "h1_key_bucket_worse_than_baseline": bool(h1_supported),
        "h1_error_ratio": (
            key["mean_error"] / baseline["mean_error"]
            if (key["mean_error"] and baseline["mean_error"])
            else None
        ),
    }


# --------------------------------------------------------------------------
# Diagnostic 4 -- false-negative rate vs imagined-rollout depth.
# --------------------------------------------------------------------------

def _diagnostic_4(
    records: List[Dict[str, Any]],
    support_low_quantile: float = 0.50,
    error_high_quantile: float = 0.50,
) -> Dict[str, Any]:
    """A false negative is a query the support score deems supported (low CSA)
    whose imagined state is in fact far from reality (high rollout error). The
    memo predicts this rate rises with rollout depth -- the structural risk of
    scoring imagined states (hard wound #1)."""
    rollout = [r for r in records if not r["is_decision_step"] and _finite(r["rollout_error"])]
    if not rollout:
        return {"status": "not_computed", "reason": "no imagined-rollout records with finite error"}
    csa_scores = torch.tensor([r["csa_score"] for r in rollout])
    errors = torch.tensor([r["rollout_error"] for r in rollout])
    depths = [int(r["depth"]) for r in rollout]

    csa_thr = torch.quantile(csa_scores, support_low_quantile).item()
    error_thr = torch.quantile(errors, error_high_quantile).item()
    false_negative = (csa_scores <= csa_thr) & (errors >= error_thr)

    by_depth = {}
    for depth in sorted(set(depths)):
        mask = torch.tensor([d == depth for d in depths])
        count = int(mask.sum().item())
        by_depth[str(depth)] = {
            "count": count,
            "false_negative_count": int((false_negative & mask).sum().item()),
            "false_negative_rate": _safe_ratio((false_negative & mask).sum().item(), count),
            "mean_rollout_error": float(errors[mask].mean().item()) if count else None,
        }
    return {
        "status": "computed",
        "support_low_quantile": support_low_quantile,
        "error_high_quantile": error_high_quantile,
        "csa_threshold": csa_thr,
        "error_threshold": error_thr,
        "by_depth": by_depth,
    }


# --------------------------------------------------------------------------
# Go / No-Go.
# --------------------------------------------------------------------------

def _go_no_go(
    diag0: Dict[str, Any],
    diag1: Dict[str, Any],
    diag2: Dict[str, Any],
    diag3: Dict[str, Any],
    gate_min_ratio: float,
) -> Dict[str, Any]:
    """The recommendation is advisory only -- the memo says the gate threshold
    is not pre-committed; a human reads the diag-0 census + 2D distribution."""
    ratio = None
    enriched = None
    if diag0.get("status") == "computed":
        primary = diag0["primary"]
        ratio = primary["ratio_failed"]
        if primary["ratio_successful"] is not None and ratio is not None:
            enriched = ratio > primary["ratio_successful"]
    corr_positive = None
    if diag2.get("status") == "computed":
        corrs = [t.get("pearson_action") for t in diag2["by_threshold"] if t.get("pearson_action") is not None]
        corr_positive = bool(corrs) and all(c > 0 for c in corrs)
    prevalent = ratio is not None and ratio >= gate_min_ratio
    return {
        "gate_min_ratio": gate_min_ratio,
        "failure_mode_ratio": ratio,
        "failure_mode_prevalent": bool(prevalent),
        "failure_mode_enriched_in_failures": enriched,
        "conditioned_correlation_positive": corr_positive,
        "recommendation": "go" if (prevalent and corr_positive) else "no-go",
        "note": "Advisory only. Inspect the diagnostic_0 threshold_sweep before deciding.",
    }


# --------------------------------------------------------------------------
# Helpers.
# --------------------------------------------------------------------------

def _finite(value: Any) -> bool:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return False
    return f == f and f not in (float("inf"), float("-inf"))


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def _summarize_mask(mask: torch.Tensor, errors: torch.Tensor, failed: torch.Tensor) -> Dict[str, Any]:
    count = int(mask.sum().item())
    if count == 0:
        return {"count": 0, "mean_error": None, "failure_rate": None}
    return {
        "count": count,
        "mean_error": float(errors[mask].mean().item()),
        "failure_rate": float(failed[mask].to(torch.float32).mean().item()),
    }


def _pearson(x: torch.Tensor, y: torch.Tensor) -> float | None:
    if x.numel() < 2 or y.numel() < 2:
        return None
    x = x.to(torch.float32) - x.to(torch.float32).mean()
    y = y.to(torch.float32) - y.to(torch.float32).mean()
    denom = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    if denom.item() <= 1e-12:
        return None
    return float((x * y).sum().item() / denom.item())


def _spearman(x: torch.Tensor, y: torch.Tensor) -> float | None:
    if x.numel() < 2 or y.numel() < 2:
        return None
    return _pearson(_rank(x), _rank(y))


def _rank(x: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(x)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(x.numel(), dtype=torch.float32)
    return ranks


def _empty_diagnostics(gate_min_ratio: float) -> Dict[str, Any]:
    return {
        "diagnostic_0_failure_mode": {"status": "not_computed", "reason": "no scored queries"},
        "diagnostic_1_exploitation_gap": {"status": "not_computed", "reason": "no scored queries"},
        "diagnostic_2_conditioned_correlation": {"status": "not_computed", "reason": "no scored queries"},
        "diagnostic_3_buckets": {"status": "not_computed", "reason": "no scored queries"},
        "diagnostic_4_false_negative_by_depth": {"status": "not_computed", "reason": "no scored queries"},
        "go_no_go": {"gate_min_ratio": gate_min_ratio, "recommendation": "no-go"},
    }
