# MGVT-D Stage-D1 Baseline Freeze

Date: 2026-06-02

Status: `FROZEN FOR D1-A CANDIDATE COMPARISON`, prediction-only. This document freezes the Stage-2c AdaLN(depth=1) baseline under the D1 diagnostic code before any D1-A full confirmation comparison.

## Scope

- Baseline: Stage-2c AdaLN(depth=1), 5 Phase-1 tasks, 3 seeds per task, W=2/H=4 training, H=1..6 diagnostics.
- Re-diagnostic run: `mgvt_d1_baseline_rediag_20260602`.
- Remote root: `/home/ps/Code/yqy/DINO-WM/jepawm_logs/mgvt_d1_baseline_rediag_20260602`.
- Local root: `experiments/results/mgvt_d/d1_20260602/mgvt_d1_baseline_rediag_20260602`.
- Source Stage-2c artifacts: `/home/ps/Code/yqy/DINO-WM/jepawm_logs/mgvt_stage2c_3seed_20260531` and `/home/ps/Code/yqy/DINO-WM/jepawm_checkpoints/mgvt_stage2c_3seed_20260531`.
- Re-diagnostic commit: `6072b00c004e879821b022b966ecfebed402cb2c` (`6072b00`).
- Exclusions: no training, no CEM, no planning eval, no H=8, no Stage-3.

## Integrity

- `skill_score.json`: 15/15 present and parseable.
- Raw diagnostic rows: 90 = 15 runs x 6 horizons.
- Summary rows: 30 = 5 tasks x 6 horizons.
- Scan end: `MGVT-D Stage-D1 baseline re-diagnostics finished 2026-06-01T23:35:32+08:00`.
- Remote worktree was clean at launch; `git_status_start.txt` is empty.
- Log grep found no `traceback`, `runtimeerror`, or standalone `nan` in `driver.nohup` or per-run `skill_score.log`.
- Old Stage-2c `skill`, `change_skill`, and `proprio_skill` reproduce exactly under the new diagnostic path: max absolute drift `0` across 90 run-horizon rows.

## Moved-Region Mask Rule

This is a latent diagnostic mask, not pixel optical flow and not an observation-grounded object mask. Claims must be phrased as moved-region latent prediction unless a later plan introduces an observation/state mask and re-freezes the baseline.

| Field | Frozen value |
| --- | --- |
| `source` | `target_side_persistence_patch_mse` |
| `top_frac` | `0.2` |
| `min_floor` | `0.0` |
| `denominator_floor` | `0.0` |
| `dilation` | `1` |

The denominator floor is intentionally frozen at `0.0` because the current D1 diagnostic code computes skill relative to all target-side persistence motion inside the top-20% latent moved patches. Changing this value makes candidate results incomparable with this baseline.

## H4 Frozen Baseline And Floors

Primary D1-A comparison uses H4 `moved_region_change_skill`. A task-level win requires the candidate mean to clear `baseline_moved_H4_mean + 0.020`. This is an absolute effect floor, not a relative-percent threshold.

| Task | Seeds | Skill mean | Skill std | Change mean | Change std | Moved mean | Moved std | Frozen moved floor | Proprio mean | Proprio std |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PushT | 234 235 236 | 0.558493 | 0.015610 | 0.706201 | 0.011280 | 0.616359 | 0.013988 | 0.636359 | 0.996936 | 0.001371 |
| Wall | 234 235 236 | 0.811236 | 0.003429 | 0.884096 | 0.004993 | 0.840925 | 0.004206 | 0.860925 | 0.941791 | 0.010577 |
| PointMaze | 234 235 236 | 0.343501 | 0.029232 | 0.592846 | 0.033323 | 0.476823 | 0.030233 | 0.496823 | 0.899755 | 0.011974 |
| MW-R | 1 2 3 | 0.659362 | 0.022173 | 0.808101 | 0.013311 | 0.716599 | 0.019808 | 0.736599 | 0.949731 | 0.010425 |
| MW-RW | 1 2 3 | 0.692445 | 0.008304 | 0.836040 | 0.007340 | 0.748568 | 0.008142 | 0.768568 | 0.940865 | 0.017488 |

## Per-Run H4 Baseline

| Run | Task | Seed | Skill H4 | Change H4 | Moved H4 | Proprio H4 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `pusht_stage2c_pred_adaln_depth1_lance_h4_seed234` | PushT | 234 | 0.545442 | 0.700160 | 0.606794 | 0.995876 |
| `pusht_stage2c_pred_adaln_depth1_lance_h4_seed235` | PushT | 235 | 0.575785 | 0.719215 | 0.632413 | 0.996448 |
| `pusht_stage2c_pred_adaln_depth1_lance_h4_seed236` | PushT | 236 | 0.554253 | 0.699227 | 0.609870 | 0.998485 |
| `wall_stage2c_pred_adaln_depth1_lance_h4_seed234` | Wall | 234 | 0.807680 | 0.878428 | 0.836996 | 0.954002 |
| `wall_stage2c_pred_adaln_depth1_lance_h4_seed235` | Wall | 235 | 0.811508 | 0.886015 | 0.840416 | 0.935897 |
| `wall_stage2c_pred_adaln_depth1_lance_h4_seed236` | Wall | 236 | 0.814522 | 0.887844 | 0.845361 | 0.935474 |
| `maze_stage2c_pred_adaln_depth1_lance_h4_seed234` | PointMaze | 234 | 0.374433 | 0.626794 | 0.509001 | 0.894981 |
| `maze_stage2c_pred_adaln_depth1_lance_h4_seed235` | PointMaze | 235 | 0.339735 | 0.591560 | 0.472457 | 0.890904 |
| `maze_stage2c_pred_adaln_depth1_lance_h4_seed236` | PointMaze | 236 | 0.316335 | 0.560185 | 0.449009 | 0.913379 |
| `mw_r_stage2c_pred_adaln_depth1_lance_h4_seed1` | MW-R | 1 | 0.676581 | 0.815086 | 0.732234 | 0.937997 |
| `mw_r_stage2c_pred_adaln_depth1_lance_h4_seed2` | MW-R | 2 | 0.667162 | 0.816467 | 0.723239 | 0.957923 |
| `mw_r_stage2c_pred_adaln_depth1_lance_h4_seed3` | MW-R | 3 | 0.634343 | 0.792751 | 0.694324 | 0.953273 |
| `mw_rw_stage2c_pred_adaln_depth1_lance_h4_seed1` | MW-RW | 1 | 0.701655 | 0.842143 | 0.756748 | 0.952747 |
| `mw_rw_stage2c_pred_adaln_depth1_lance_h4_seed2` | MW-RW | 2 | 0.690151 | 0.838080 | 0.748491 | 0.949065 |
| `mw_rw_stage2c_pred_adaln_depth1_lance_h4_seed3` | MW-RW | 3 | 0.685529 | 0.827896 | 0.740465 | 0.920784 |

## H1-H6 Moved-Region Horizon Decay

H4 is the frozen primary horizon. H1-H3 are supportive; H5-H6 are diagnostic only and must not be used to claim planning readiness.

| Task | H1 | H2 | H3 | H4 | H5 | H6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PushT | 0.449582 | 0.563645 | 0.597784 | 0.616359 | 0.628638 | 0.639990 |
| Wall | 0.750129 | 0.825492 | 0.838577 | 0.840925 | 0.838894 | 0.835623 |
| PointMaze | 0.424505 | 0.517842 | 0.510780 | 0.476823 | 0.436082 | 0.396440 |
| MW-R | 0.433034 | 0.626077 | 0.690066 | 0.716599 | 0.725816 | 0.725952 |
| MW-RW | 0.449508 | 0.648976 | 0.717313 | 0.748568 | 0.763433 | 0.771066 |

## D1-A Candidate Gate

The gate below applies only after the remaining D1-A smokes pass and before any full 75-run confirmation is interpreted. It is frozen here to prevent threshold tuning after seeing candidate results.

- Primary metric: H4 `moved_region_change_skill` using the frozen mask above.
- Matched pairs: same task and same seed as the Stage-2c baseline, yielding 15 task-seed pairs per candidate.
- Absolute effect: all-task paired mean delta on H4 moved-region skill must be at least `+0.020`.
- Task floor: at least 3/5 tasks must clear the per-task frozen moved floor in the H4 table, and no task may have mean moved-region delta below `-0.020`.
- Significance: one-sided paired sign-flip/permutation test over the 15 H4 moved-region deltas; apply Holm-Bonferroni correction across D1-A candidate comparisons. Report paired mean delta, 95% bootstrap CI, uncorrected p, and corrected p.
- Harm guardrail: H4 global `skill` and `proprio_skill` paired mean deltas must each be at least `-0.020` overall and per task. Any H<=4 non-finite metric is an immediate fail.
- Control interpretation: `adaln_param_match` can beat the baseline as a capacity control, but it cannot support the dynamics-decoupling claim. A trend model only supports the core claim if it beats both Stage-2c and the parameter/FLOP-matched AdaLN control under the same metric family.
- Parameter/FLOP control note: `adaln_param_match` is parameter-matched to the Mamba lead candidate at roughly 569k trainable predictor-side parameters; the reported AdaLN FLOPs are an analytic linear-only proxy and must not be used to claim exact FLOP parity.
- Stop condition: if no trend variant (`raw_action_guidance`, `mlp_trend_dim16`, `gru_trend_dim16`, `mamba_trend_dim16`) clears this gate, stop after D1-A and write a neutral/negative D1-A gate. Do not proceed to D1-B/C/D/E by momentum.

## Cross-Position Diagnostic Definition

Cross-position consistency is a diagnostic for reusable action-induced trends; it is not the primary D1-A gate until implemented and reviewed.

- Split: validation only; never use train windows or test/planning rollouts.
- Pairing target: same relative motion, different absolute start. Use task state/proprio deltas where available. Offline target deltas may be used for grouping diagnostics, but not as model input.
- Same-relative-motion pair: cosine similarity of target delta or action-induced proprio delta at least `0.95`, relative magnitude difference at most `20%`, and absolute prefix-state distance above the task median.
- Negative pair: same task and horizon, but cosine similarity below `0.50` or magnitude ratio outside `[0.5, 2.0]`.
- Readout: compare `d_h` or coarse-guidance similarity for positive vs negative pairs and report AUROC / delta-similarity. A positive cross-position diagnostic cannot override a failed H4 moved-region gate.
- Claim wording: until this diagnostic is implemented, D1-A can only claim improved latent prediction under a dynamics-guided architecture, not proven reusable physical primitive discovery.

## Candidate Comparison Checklist

- Run exactly the reviewed D1-A variants and seeds; no silent reroll. Failed runs keep their id; retries use `_retryN`.
- Keep `max_horizon=6`, H4 primary, same Lance stores, same checkpoint policy, same frozen diagnostic script, and same moved-region rule.
- Report `skill`, `change_skill`, `moved_region_change_skill`, `proprio_skill`, horizon decay, params, FLOPs, and train step time.
- Logs must reject `traceback`, `runtimeerror`, and standalone `nan`.
- D1-B remove/shuffle/random `d_h`, D1-C proprio placement, D1-D refiner depth, and D1-E decomposition controls remain blocked until D1-A has a candidate winner.

## Files

- Re-diagnostic raw CSV: `mgvt_d1_baseline_rediag_20260602/summary/mgvt_d1_horizon_raw.csv`.
- Re-diagnostic summary CSV: `mgvt_d1_baseline_rediag_20260602/summary/mgvt_d1_horizon_summary.csv`.
- Re-diagnostic per-run JSON: `mgvt_d1_baseline_rediag_20260602/skill_scores/*/skill_score.json`.
- Frozen comparison source: this file.

