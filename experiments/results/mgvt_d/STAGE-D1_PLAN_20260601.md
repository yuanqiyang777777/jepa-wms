# MGVT-D Stage-D1 Plan

Date: 2026-06-01

Author: Codex draft for Claude review. Claude owns plan ratification.

Branch: `exp/20260530-mgvt-stage2`

Worktree: `C:\Users\17695\.config\superpowers\worktrees\jepa-wms\exp-20260529-mgvt-v1`

Method source: `E:\jepa-wms\notes\mgvt_dynamics_guided_confidence_framework_20260601_v6.tex` and `.pdf` (Claude-reviewed `PASS`).

Frozen baseline source: `E:\jepa-wms\experiments\results\mgvt_stage2\stage2c_3seed_20260531\MGVT_STAGE2C_GATE_20260531.md`.

Stage-2c status: Claude ratified `PASS-WITH-FLAGS` on 2026-06-01; the flags were documentation/provenance readouts, not result defects. The gate doc has been updated with per-seed H1-H4 positivity, H1-H6 decay, and Stage-2b sanity reproduction. AdaLN(depth=1) is the frozen prediction-only D-stage baseline for H<=4.

Claude D1 plan review: first review returned `PASS-WITH-FLAGS` on 2026-06-01. The required edits from that review are folded into this revision: baseline re-diagnostics for the new moved-region metric, `d_h` dimensionality sweep, task-stratified statistics, `delta_p` constraints, complete parameter/FLOP accounting, per-task moved-mask/cross-position definitions, proprio-passthrough gates, and a real-CUDA Mamba execution check.

Claude D1 re-check: second review returned `PASS-WITH-FLAGS (non-blocking for implementation)` on 2026-06-01. It found no remaining blockers. The two small required clarifications from that re-check are folded in: similar-action/different-outcome negative pairs replace stale "mismatched-action" wording, and the PASS rule now disambiguates per-task floors versus the 4/5 cross-task guard.

---

## 0. Status and Authorization Boundary

This plan has Claude non-blocking `PASS-WITH-FLAGS` for implementation planning. It still does not authorize code edits, GPU launch, remote smoke, full scan, CEM, planning evaluation, H=8, Stage-3, Notion updates, or `notes/alignment_log.md` edits by itself.

After Claude review and user approval, this plan authorizes Codex to create or edit **only** the files enumerated in section 5. Any change outside that file list requires a new plan amendment and user approval.

Hard exclusions for Stage-D1:

- no confidence-aware CEM;
- no Terver planning evaluation;
- no online planning success claim;
- no H=8 or training horizon greater than H=4;
- no Stage-3;
- no PointMaze Mamba recheck outside the D1 matrix;
- no registry, alignment log, or Notion update until results exist and Claude ratifies the result gate.

---

## 1. Scientific Goal

Stage-D1 tests one core hypothesis:

> A compact action-induced dynamics trend signal `d_h`, learned by a dedicated dynamics predictor `F_dyn`, can guide a scene-conditioned visual latent refiner `F_refine` to predict future DINO/JEPA latents more efficiently than the implicit coupled AdaLN predictor, especially on changed or moved regions.

The proposed decomposition is:

```text
s_t = Phi_state(z_{t-W+1:t}, p_{t-W+1:t}, a_{t-W:t-1})
d_{1:H}, delta_p_{1:H} = F_dyn(s_t, a_{t:t+H-1})
tilde_z_{t+h} = G_guidance(z_t, d_h)
hat_z_{t+h} = F_refine(z_{t-W+1:t}, p_{t-W+1:t}, tilde_z_{t+h}, d_h)
```

For Stage-D1, `d_h` is not claimed to be a discovered option, skill, or complete physical state. It is a testable bottleneck for **relative motion trend**: direction, displacement, velocity/contact tendency, or a small set of motion-basis coefficients. The visual refiner is still responsible for scene grounding, token-level placement, occlusion effects, contact detail, and residual correction.

Stage-D1 must distinguish three possibilities:

1. the dynamics-guided architecture improves prediction because `d_h` carries reusable relative transition structure;
2. the improvement, if any, is only parameter/FLOP/capacity gain;
3. `d_h` leaks full visual future information or degenerates into another implicit predictor.

Only case 1 supports the D1 claim.

---

## 2. Baseline and Run Constants

Frozen prediction-only baseline:

- model: Stage-2c `AdaLN(depth=1)`;
- tasks: PushT, Wall, PointMaze, MW-R, MW-RW;
- seeds: PushT/Wall/PointMaze `234 235 236`; MW-R/MW-RW `1 2 3`;
- W=2 context, H=4 training, H=1..6 diagnostics;
- frozen DINO encoder, normalized latent targets, rollout stop-gradient, Lance no-fallback;
- no CEM/planning/H8/Stage-3.

Stage-D1 uses the same constants unless explicitly listed as a D1 factor:

- tasks: same five Phase-1 tasks;
- seeds: same seed sets as Stage-2c;
- training horizon: H=4 only;
- diagnostics: H=1..6 report, gate reads H<=4;
- checkpoint policy: `jepa-latest.pth.tar`, no best-test checkpoint selection;
- data backend: Lance with `fall_back_to_raw_if_unsupported=false`;
- remote code root: `/home/ps/Code/yqy/DINO-WM/jepa-wms`;
- conda env: `jepa-wms`;
- GPU default: use GPUs `0-5` or `7`; avoid GPU 6 unless a fresh check shows it is idle.

Proposed D1 run roots:

- smoke run root: `/home/ps/Code/yqy/DINO-WM/jepawm_logs/mgvt_d1_smoke_20260602`;
- smoke checkpoint root: `/home/ps/Code/yqy/DINO-WM/jepawm_checkpoints/mgvt_d1_smoke_20260602`;
- full run root: `/home/ps/Code/yqy/DINO-WM/jepawm_logs/mgvt_d1_20260602`;
- full checkpoint root: `/home/ps/Code/yqy/DINO-WM/jepawm_checkpoints/mgvt_d1_20260602`;
- local artifact root after pull: `E:\jepa-wms\experiments\results\mgvt_d\d1_20260602`.

Run-id shape:

```text
20260602_mgvt_d1_<block>_<task>_<variant>_seed<seed>
```

Examples:

- `20260602_mgvt_d1a_pusht_mamba_trend_seed234`
- `20260602_mgvt_d1b_wall_mamba_trend_shuffle_dh_seed235`
- `20260602_mgvt_d1c_mw_rw_mamba_trend_dyn_refine_prop_seed2`

Retries must append `_retryN`. Failed runs keep their original run directory and registry status; never silently reroll.

---

## 3. D1 Experiment Matrix

The matrix is staged. A later block is launched only if the earlier block passes its smoke and does not violate stop conditions.

### D1-A: Backbone and Guidance Main Test

Purpose: test whether explicit trend guidance helps beyond implicit AdaLN and beyond raw-action conditioning.

Rows:

| Variant | Description | Full runs |
|---|---|---:|
| `stage2c_adaln_depth1` | Frozen Stage-2c training baseline; rerun only the extended D1 diagnostics on the 15 frozen checkpoints to obtain moved-region metrics | 0 new training / 15 baseline re-diagnostics |
| `raw_action_guidance` | `G_guidance/F_refine` receives raw action embedding only; no learned `F_dyn` state | 15 |
| `mlp_trend_dim16` | low-capacity `F_dyn` MLP over compact state/action, default `d_h_dim=16` | 15 |
| `gru_trend_dim16` | recurrent `F_dyn` over action horizon, default `d_h_dim=16` | 15 |
| `mamba_trend_dim16` | Mamba/SSM `F_dyn` over action horizon, default `d_h_dim=16`, with GRU fallback disallowed for final GPU runs | 15 |
| `adaln_param_match` | AdaLN or implicit coupled control matched to best D1 parameter/FLOP budget | 15 |

Expected new D1-A runs: `5 variants x 5 tasks x 3 seeds = 75`.

Primary comparison: each trend variant vs frozen Stage-2c re-diagnostics, and best trend variant vs `adaln_param_match`.

Candidate launch is blocked until the 15 Stage-2c frozen checkpoints are re-diagnosed with the new moved-region metric, the per-task baseline mean/std are written to `D1_BASELINE_FREEZE_20260602.md`, and the numeric absolute floors in section 4 are frozen before any candidate result is available.

### D1-B: Necessity and Leakage of `d_h`

Purpose: verify that the model uses `d_h` as a trend signal, not as dead code or a future-latent leak.

Run only for the best D1-A trend variant unless D1-A has multiple statistically tied winners. If there are tied winners, run D1-B for at most two variants.

Rows per selected trend variant:

| Variant | Meaning | Full runs |
|---|---|---:|
| `dh_dim_sweep` | selected `F_dyn` backbone with `d_h_dim in {4,8,16,32}`; dim16 can reuse D1-A | 45 new + 15 reused |
| `remove_dh` | replace `d_h` input to `G_guidance/F_refine` with zeros at train and eval | 15 |
| `shuffle_dh` | shuffle `d_h` across batch within task/horizon at eval; training unchanged | 15 |
| `random_dh` | feed matched-distribution random `d_h` at eval; training unchanged | 15 |
| `coarse_guidance_readout` | mandatory post-hoc diagnostic probe on frozen `d_h` / guidance intermediates; no gradient into the world model and no extra training row | 0 |
| `leakage_probe` | linear/MLP probes from frozen `d_h`, no new world-model training | 0 |

Pass requirement: a compact dimension must be chosen by the performance/leakage tradeoff, not by performance alone. `remove_dh`, `shuffle_dh`, and `random_dh` must significantly reduce moved-region `change_skill` versus the intact model. If they do not, D1 cannot claim that the trend variable is causally used.

`coarse_guidance_readout` is a post-hoc probe trained after the world model is frozen. It may read `d_h`, `tilde_z`, and pre-refinement intermediates to predict relative moved-region change, but it must not update `F_dyn`, `G_guidance`, or `F_refine`. If a later implementation wants a co-trained coarse readout, that becomes a new ablation because it changes optimization and parameter/FLOP accounting.

### D1-C: Proprioception Placement

Purpose: test where proprioception belongs in the decomposition.

Run for the selected D1-A/B architecture.

Rows:

| Variant | Proprio flow | Full runs |
|---|---|---:|
| `no_proprio` | no proprio in `Phi_state`, `F_dyn`, or `F_refine` | 15 |
| `dyn_only_prop` | proprio enters `Phi_state/F_dyn` only | 15 |
| `refine_only_prop` | proprio enters `F_refine` only | 15 |
| `dyn_refine_prop` | proprio enters both dynamics state and refiner | 15 |
| `old_concat_prop` | old-style implicit concatenation/control, no explicit trend bottleneck | 15 |

Required proprio-passthrough controls:

- `proprio_fixed`: hold proprio constant during evaluation while action/vision vary;
- `visual_ablated_gain`: compare visual gain with proprio removed to ensure improvements are not pure proprio shortcutting.

These controls are required for the primary dynamics-trend claim because Stage-2c proprio_skill is near-saturated. The main expected winner is not pre-claimed. A plausible default is `dyn_refine_prop`, but the plan must accept if `dyn_only_prop` or `refine_only_prop` is better.

### D1-D: Visual Refiner Capacity

Purpose: test whether the trend signal reduces pressure on the visual predictor/refiner.

Run for the selected dynamics/proprio setting.

Rows:

| Variant | Meaning | Full runs |
|---|---|---:|
| `refine_depth_baseline` | same refiner depth as main D1-A selected model | 15 |
| `refine_depth_half` | roughly half visual refiner depth/blocks | 15 |
| `refine_depth_minimal` | minimal residual refiner | 15 |
| `no_fdyn_param_match` | same refiner capacity but no learned trend bottleneck | 15 |

Claim boundary: only claim "trend reduces visual predictor pressure" if reduced-depth guided models match or beat no-`F_dyn` matched controls under the primary metric.

### D1-E: External Decomposition Control

Purpose: ensure the result is not simply "any decomposition helps".

At least one external control is required:

| Variant | Meaning | Full runs |
|---|---|---:|
| `sparse_update_control` | DDP-WM-style sparse token update/routing without compact trend bottleneck | 15 |
| `latent_routing_control` | changed-token routing/gated residual control without explicit `d_h` | optional 15 |

These are controls, not replacements for the main method. They should be FLOP/parameter matched as closely as possible to the selected D1 architecture.

---

## 4. Metrics, Diagnostics, and Gates

### Primary Metric

Primary D1 metric:

```text
moved_region_change_skill at H=4
```

Definition: skill of predicted latent changes on a pre-registered moved-patch mask. The moved-patch mask must be derived from observations or dataset state, never from the model prediction. The same model-independent masks must be used for Stage-2c baseline re-diagnostics and all D1 candidates.

Mask source priority:

1. If a task exposes ground-truth object/agent coordinates or masks, map them to DINO patch coordinates and use their H1-H4 changed regions.
2. Otherwise compute patch-level observation motion from `|x_{t+h} - x_t|`, average within the DINO patch grid, threshold by a frozen rule, and optionally dilate one patch.
3. The threshold rule must be fixed before any candidate full run and written into `D1_BASELINE_FREEZE_20260602.md`. Default rule unless baseline diagnostics falsify it: per-sample top 20% non-background patch motion, one-patch dilation, minimum absolute RGB-difference floor frozen per task, and a static-camera background guard.
4. Add a moved-patch persistence-MSE denominator floor in the freeze doc. If persistence MSE is below this floor, report the sample as low-motion diagnostic and exclude it from the primary moved-region skill denominator to avoid unstable `1 - model/persist` ratios.

Per-task moved-region and cross-position definitions:

| Task | Moved-region source | Same-relative-motion positive pairs | Negative pairs |
|---|---|---|---|
| PushT | object and pusher patch regions from state if available; otherwise RGB-motion patches around object/pusher | similar object displacement vector with different object start position | similar actions but different object displacement direction/magnitude |
| Wall | agent/body moved patches plus collision/contact region if visible | similar agent displacement and collision flag with different absolute wall position | similar actions with collision vs no-collision or different displacement outcome |
| PointMaze | agent moved patches from state/observation motion; goal-relative displacement tracked separately | similar agent-to-goal relative displacement through different absolute maze locations | similar actions that produce different maze transition due to wall/corridor context |
| MW-R | end-effector/hand moved patches and target-relative delta-state | similar end-effector delta with different absolute workspace start | similar actions with different end-effector delta or target-relative outcome |
| MW-RW | end-effector/hand moved patches plus wall/contact context | similar end-effector delta and contact/wall relation with different absolute start | similar actions with different wall/contact outcome |

If a task lacks a clean moved-region or same-relative-motion definition after baseline audit, it remains in prediction metrics but is excluded from the reusable-trend gate. The exclusion must be declared before candidate launch. A broad five-task reusable-trend claim requires all five tasks to remain eligible; if any task is excluded, the claim must be narrowed to the eligible task set and the result can be at most `PASS-WITH-FLAGS` for reusable-trend consistency.

Secondary metrics:

- global `skill`;
- global `change_skill`;
- `proprio_skill`;
- horizon decay H1-H6;
- cross-position consistency of `d_h`;
- coarse-guidance readout accuracy before refinement;
- train step time, parameter count, FLOPs estimate;
- latent std, deterministic encode diff, finite-metric checks.

### Baseline Re-Diagnostic and Numeric Floor Freeze

Before any D1 candidate full run:

1. Run the extended D1 diagnostic on the 15 frozen Stage-2c checkpoints. This is **0 new training** and writes outputs only under `experiments/results/mgvt_d/d1_20260602/`; it must not overwrite the frozen Stage-2c artifact directory.
2. Reproduce the original Stage-2c `skill`, `change_skill`, and `proprio_skill` values from the gate doc within a pinned tolerance to prove the `skill_score.py` edit did not perturb the old metrics.
3. Compute per-task moved-region baseline mean `mu_t` and sample std `sigma_t`.
4. Freeze numeric floors in `D1_BASELINE_FREEZE_20260602.md` before launching candidates:
   - `floor_t = mu_t + max(0.02, 0.5 * sigma_t)` for each task;
   - overall effect threshold = mean over tasks of `max(0.03, 0.5 * sigma_t)`;
   - no-task harm threshold = `max(0.02, 0.5 * sigma_t)` per task.
5. Claude must review `D1_BASELINE_FREEZE_20260602.md` unconditionally before candidate launch, because it is the load-bearing pre-registration artifact for numeric floors, task masks, thresholds, exclusions, and denominator rules.

The `sigma_t` estimates come from only three seeds and are therefore high-variance. The absolute backstops in the formulas above (`0.02` for per-task floor/harm and `0.03` for overall effect) must not be weakened after seeing candidate results.

### Statistical Gate

D1 is small-n by task, so the primary gate must respect task structure:

- paired units: `(task, seed)` matched to Stage-2c re-diagnostics and candidate runs;
- primary estimator: mean over the five task means, not the raw pooled mean over 15 units;
- primary test: task-stratified paired sign-permutation or bootstrap, permuting signs within each task and reporting the cross-task mean effect;
- correction: Holm or Benjamini-Hochberg across D1-A candidate-vs-baseline tests and any declared D1-B dimension-choice tests;
- effect-size gate: candidate must clear frozen `floor_t` on at least 4/5 tasks, must not violate the harm threshold on the remaining task, and must clear the overall cross-task effect threshold from `D1_BASELINE_FREEZE_20260602.md`;
- cross-task consistency guard: effect direction must be positive in at least 4/5 tasks for a broad D1 claim;
- no-task harm guard: no task may violate its frozen harm threshold unless the result is marked `PASS-WITH-FLAGS` or `FAIL`;
- robustness guard: H4 mean-minus-std must remain positive for `skill`, `change_skill`, `proprio_skill`, and moved-region `change_skill` on each task.

Per-task bootstrap confidence intervals must be reported for interpretation, but per-task p-values are secondary because `n=3` per task is underpowered.

### `d_h` Necessity Gate

For a positive D1 claim:

- intact guided model must beat Stage-2c and parameter/FLOP matched controls;
- `remove_dh`, `shuffle_dh`, and `random_dh` must each reduce moved-region H4 `change_skill` by at least `max(0.02, 0.5 * sigma_t)` on the mean-over-tasks estimator, with paired direction consistent on at least 12/15 task-seed units;
- coarse-guidance readout must predict relative change above raw-action guidance;
- `d_h_dim in {4,8,16,32}` must show a plausible bottleneck tradeoff; the selected dim cannot be chosen solely by the highest prediction metric if leakage rises sharply;
- `d_h` cross-position consistency must be higher for same-relative-motion pairs than for mismatched-relative-motion pairs built as similar-action / different-realized-outcome negatives.

If the intact model improves but `d_h` ablations do not hurt, the result is an architecture/capacity improvement, not a dynamics-trend claim.

### Leakage Gate

Probe families:

- `d_h -> future z_{t+h}` reconstruction probe;
- `d_h -> absolute patch position / region id` probe;
- `d_h -> task id / episode id` probe;
- `d_h -> delta proprio` and relative motion probe.

Allowed behavior:

- high predictability of relative motion or `delta_p`;
- low predictability of full future visual latent, absolute location, task id, or episode id beyond what compact state/action already explains.

Leakage thresholds are calibrated per task, probe family, and `d_h_dim` using a shuffled-`d_h` null distribution. Default frozen rule: fail if the real probe exceeds `mu_null + 3 * sigma_null` after correction. Report absolute `Delta R^2`, `Delta AUROC`, and `Delta accuracy` as diagnostics; any value above `0.10` is an automatic flag even if the shuffled-null test is inconclusive.

Leakage fail if any holds after correction:

- `d_h -> future z` reconstruction exceeds the calibrated threshold over the compact `(s_t, a_{t:t+H-1})` baseline for full-latent reconstruction;
- absolute region or absolute position probe exceeds the calibrated threshold over compact state/action;
- `d_h` alone predicts task/episode identity well enough to explain the main result;
- leakage probes explain more of the final prediction than the relative-motion or `delta_p` probes.

Claude must approve any relaxation before D1 result interpretation.

### Cross-Position Test

For each task, define pairs of samples with:

- similar action sequence or relative displacement target;
- different absolute start positions or scene locations;
- same horizon.

Measure:

```text
sim(d_h_i, d_h_j | same relative motion, different position)
  >
sim(d_h_i, d_h_k | mismatched relative motion)
```

Negative pairs must include similar action sequences with different realized relative-motion outcomes, so the test isolates reusable trend from mere action encoding. The cross-position denominator is the pre-registered eligible task set from the baseline freeze doc. If all five tasks are eligible, require positive consistency on at least 4/5 tasks; if fewer than four tasks are eligible, D1 cannot make a broad reusable-trend claim. This supports the user's intended claim: common action-induced trend is reused across different positions. If this test fails, the method may still be useful but cannot claim reusable trend decomposition.

---

## 5. Files Authorized for D1 Implementation After Claude Review

This plan proposes the following implementation file list. It is not active until Claude review passes and the user approves implementation.

Core model files:

1. `app/plan_common/models/mgvt_dynamics_guided.py` *(new)*  
   Implements `Phi_state`, temporal `F_dyn` backbones (MLP/GRU/Mamba over the action horizon), `G_guidance`, `F_refine`, param/FLOP reporting hooks, and control modes.

2. `app/plan_common/models/mgvt_mixers.py`  
   May be touched only for non-invasive backend reporting or shared utility extraction. Do **not** reuse the existing spatial raster-scan `MambaMixer` as `F_dyn`; D1 `F_dyn` must be a temporal action-horizon module. Do not alter Stage-1/2 behavior.

3. `app/vjepa_wm/utils.py`  
   Register `pred_type` entries such as `mgvt_d_raw_action`, `mgvt_d_mlp`, `mgvt_d_gru`, `mgvt_d_mamba`, `mgvt_d_sparse_control`.

4. `app/vjepa_wm/video_wm.py`  
   Wire guided predictor outputs, auxiliary `delta_p`, coarse guidance readout, and rollout behavior without changing existing `AdaLN`, `dino_wm`, `vjepa2_ac`, or Stage-2 MGVT paths.

5. `app/vjepa_wm/train.py`  
   Log D1 auxiliary losses and diagnostics; keep target encoder, stop-gradient, EMA, masking, and checkpoint policy unchanged.

Diagnostics and analysis:

6. `app/vjepa_wm/diagnostics/skill_score.py`  
   Add moved-region mask metric, post-hoc coarse-guidance readout metrics, D1 metric fields, complete predictor/submodule parameter counting, per-run FLOP estimate in `skill_score.json`, and backward-compatible regression for existing Stage-2c JSON schema fields.

7. `app/vjepa_wm/diagnostics/mgvt_d1_probes.py` *(new)*  
   Implements `d_h` leakage probes, cross-position consistency, and readout aggregation.

8. `experiments/scripts/summarize_mgvt_d1.py` *(new)*  
   Aggregates raw JSON/probe outputs into CSV/MD with task-stratified paired effects, corrected p-values, per-task bootstrap CIs, and numeric baseline floor freeze rows.

Config and launchers:

9. `experiments/scripts/gen_stage_d1_configs.py` *(new)*  
   Generates all D1 configs from frozen Stage-2c templates and candidate matrix definitions.

10. `experiments/scripts/mgvt_d1_scan.sh` *(new)*  
    Handles smoke/full D1 runs, Lance preflight, provenance snapshots, and summary generation.

11. `configs/vjepa_wm/mgvt_d1/*.yaml` *(generated)*  
    Generated configs only; no hand edits after generation.

Tests:

12. `tests/models/test_mgvt_dynamics_guided.py` *(new)*  
    CPU shape tests, gradient tests, `d_h` ablation behavior, and no-fallback Mamba contract.

13. `tests/configs/test_mgvt_d1_configs.py` *(new)*  
    Config schema tests, generated-matrix tests, no CEM/planning/H8 tests, and Stage-2c template equivalence checks.

14. `tests/diagnostics/test_mgvt_d1_diagnostics.py` *(new)*  
    Moved-mask, `change_skill`, coarse readout, leakage probe parser, paired statistics tests, and a Stage-2c backward-compat regression that pins the 15 frozen H4 values from the gate doc.

Result files after runs:

15. `experiments/results/mgvt_d/d1_20260602/` *(new artifacts after pull only)*  
    Raw pulled artifacts, `D1_BASELINE_FREEZE_20260602.md`, summaries, and a Codex draft gate doc. Claude ratifies final result.

Not authorized in D1 implementation:

- `evals/**`;
- `configs/online_plan_evals/**`;
- `notes/alignment_log.md`;
- Notion;
- existing Stage-1/2a/2b/2c configs or result artifacts, except read-only use as templates/baselines.

---

## 6. Implementation Rules

Architecture:

- `F_dyn` may read compact state extracted from visual/proprio history, but its output must be low-dimensional or bottlenecked relative trend `d_h`, not full future visual latent.
- Default `d_h_dim` is 16 for D1-A. D1-B must sweep `{4,8,16,32}` before any compact-bottleneck claim.
- Full visual latent history belongs to `F_refine`, not unrestricted `F_dyn`.
- `L_trend` is default off. It may appear as an optional diagnostic loss but cannot be used in the primary D1 claim unless a separate ablation proves it is needed.
- Confidence heads are out of scope except placeholder config validation that they are disabled.
- Mamba final GPU runs must use real `mamba_ssm` on CUDA tensors. A backend label is insufficient: logs/tests must assert that the CUDA selective-scan path actually executed. GRU fallback is allowed only for CPU unit tests. If `mamba_ssm` is unavailable or the CUDA path does not execute on lab01, Mamba rows are `BLOCKED`, not silently converted into GRU rows.
- `delta_p` must be decoded from the same `d_h` that drives guidance, or from an explicitly labeled `b_t` variant. It cannot be satisfied by a parallel side head that does not constrain the guidance signal.
- Forward-inverse alignment is out of D1 primary scope and disabled by default. If enabled by later amendment, it is train-only, never an inference/CEM input, and must trigger a fresh leakage re-check.
- Parameter and FLOP matching must include every new submodule: `Phi_state`, `F_dyn`, `G_guidance`, `F_refine`, auxiliary readouts used in training, action/proprio encoders, and any routing/control modules. Matching tolerance is `+/-10%` on both parameters and FLOPs unless Claude approves a tighter/fairer task-specific tolerance.
- FLOP estimates must be emitted per run into `skill_score.json` and aggregated by `summarize_mgvt_d1.py`; parameter/FLOP matching cannot be checked by prose only.
- Leakage null calibration must record shuffle count and RNG seed for each `d_h_dim` and probe family.

Training:

- Keep Stage-2c optimizer, epochs, rollout, batch/data settings unless a candidate explicitly varies refiner depth or parameter matching.
- Keep `rollout_stop_gradient=true`.
- Do not change frozen DINO encoder behavior.
- Do not select checkpoints by test metrics.
- Keep train/val/test split behavior identical to Stage-2c.

Diagnostics:

- Every smoke and full run must output `skill_score.json`, train CSV, frozen config, checkpoint provenance, coarse-guidance readout, `d_h` probe bundle, and moved-mask metric summary.
- JSON schema additions must be backward-compatible: existing Stage-2c fields keep the same names and types.
- The baseline re-diagnostic must prove old Stage-2c `skill`, `change_skill`, and `proprio_skill` values are unchanged within a pinned tolerance before any new moved-region claim is used.

---

## 7. Local Validation Before Any Remote GPU

Required local checks:

```bash
cd C:/Users/17695/.config/superpowers/worktrees/jepa-wms/exp-20260529-mgvt-v1
python experiments/scripts/gen_stage_d1_configs.py --dry-run
python experiments/scripts/gen_stage_d1_configs.py
python -m pytest tests/models/test_mgvt_dynamics_guided.py -q
python -m pytest tests/configs/test_mgvt_d1_configs.py -q
python -m pytest tests/diagnostics/test_mgvt_d1_diagnostics.py -q
python experiments/scripts/summarize_mgvt_d1.py --check-only --baseline-freeze-dry-run
bash -n experiments/scripts/mgvt_d1_scan.sh
```

Required scope checks:

```bash
git status --short --untracked-files=all
rg -n "cem|planning|H8|num_pred: 8|rollout_steps: 8|Stage-3|stage3" configs/vjepa_wm/mgvt_d1 experiments/scripts tests app src
```

The `rg` command may match negative assertions in tests/docs, but must not reveal active D1 configs or launcher code that runs CEM, planning, H=8, or Stage-3.

Remote non-GPU preflight:

- source `~/.jepawm_env`;
- activate conda env `jepa-wms`;
- `cd $JEPAWM_HOME/jepa-wms`;
- confirm Lance stores exist: `PushT.lance`, `Wall.lance`, `PointMaze.lance`, `Metaworld.lance`;
- confirm branch and commit match the reviewed D1 implementation;
- do not allocate GPU in preflight.

---

## 8. Smoke and Full Execution Plan

Smoke is two tasks, one seed, one candidate:

- recommended smoke tasks: Wall and PushT;
- recommended smoke seed: 234;
- recommended candidate: `mamba_trend` if `mamba_ssm` is available, otherwise `gru_trend` and mark Mamba blocked;
- smoke run ids: `20260602_mgvt_d1_smoke_wall_mamba_trend_seed234` and `20260602_mgvt_d1_smoke_pusht_mamba_trend_seed234`.

Wall exercises a high-skill/low-variance baseline path. PushT exercises clearer object motion and the moved-region/object-mask path.

Smoke pass criteria:

- process exits zero;
- checkpoint exists;
- train CSV exists and includes D1 auxiliary fields;
- `skill_score.json` exists and has finite H1-H6 values;
- moved-region metrics exist and are finite;
- coarse-guidance readout exists;
- `d_h` probe bundle exists;
- no raw fallback, no CEM/planning/H8/Stage-3;
- logs prove real `mamba_ssm` CUDA execution for Mamba rows, not just a backend string.

Full confirmation sequence:

0. Stage-2c baseline re-diagnostics on the 15 frozen checkpoints, then write and freeze `D1_BASELINE_FREEZE_20260602.md`.
1. D1-A full scan.
2. Analyze D1-A using the frozen numeric floors. If no trend variant clears the primary gate, stop and write a negative/neutral gate draft. Do not run D1-B/C/D/E unless Claude/user approve a diagnostic continuation.
3. D1-B on selected winner(s), including `d_h_dim` sweep before final bottleneck interpretation.
4. D1-C on selected architecture, including required proprio-fixed/visual-ablated controls.
5. D1-D on selected dynamics/proprio setting.
6. D1-E external decomposition control.

Full scan launch must be detached and provenance-captured:

- `git_commit.txt`;
- `git describe --dirty`;
- launcher snapshot;
- frozen generated config;
- environment summary;
- GPU state snapshot;
- scan start/end files.

---

## 9. Result Gate

D1 `PASS` requires all of:

- Stage-2c baseline remains the frozen comparison and is not overwritten;
- the extended diagnostic re-runs on frozen Stage-2c checkpoints reproduce original Stage-2c skill/change/proprio metrics within tolerance;
- broad-claim PASS requires the cross-task mean to clear the overall effect threshold and at least 4/5 tasks to clear their frozen `floor_t`; the remaining task, if any, must not violate its frozen harm threshold;
- corrected task-stratified paired test passes (`q < 0.10` minimum; use `q < 0.05` if power allows);
- mean-over-tasks effect direction is positive in at least 4/5 tasks;
- no-task harm guard passes;
- parameter/FLOP matched implicit control, matched within `+/-10%` on both parameters and FLOPs, does not explain the improvement;
- `remove_dh`, `shuffle_dh`, and `random_dh` degrade performance;
- `d_h_dim` sweep supports a compact bottleneck instead of only a large latent;
- `delta_p` is decoded from `d_h` or an explicitly labeled `b_t` variant, not an unconstrained side head;
- proprio-fixed/visual-ablated controls show the visual moved-region gain is not a proprio passthrough shortcut;
- leakage probes do not fail;
- cross-position same-relative-motion consistency is positive;
- all five tasks keep positive H<=4 skill/change/proprio metrics and finite diagnostics;
- no collapse, eval leakage, raw fallback, hidden checkpoint selection, CEM/planning/H8/Stage-3, or protocol drift.

D1 `PASS-WITH-FLAGS` if:

- the main trend effect is positive and robust but one diagnostic is weak, such as one task harm guard marginally failing or cross-position evidence being inconclusive;
- only 3/5 tasks clear `floor_t` but the cross-task mean and corrected test pass; the claim must then be narrowed to the passing tasks and cannot be a broad five-task D1 PASS;
- one task is excluded from reusable-trend consistency before candidate launch due to an invalid moved-region or cross-position definition;
- the flag must be explicit and must narrow the claim.

D1 `FAIL` if:

- no trend variant beats Stage-2c after correction;
- improvement is explained by parameter/FLOP matched controls;
- `d_h` ablations do not hurt;
- `d_h_dim` only works when enlarged enough to behave like a future-latent carrier;
- proprio-fixed/visual-ablated controls remove the claimed gain;
- leakage probe fails;
- any metric collapse or unrepaired integrity failure occurs.

Permitted conclusion after D1 PASS:

> D1 provides prediction-only evidence that a compact dynamics-trend bottleneck, relative to parameter/FLOP-matched implicit controls, improves scene-conditioned latent prediction on pre-registered moved regions and/or horizon decay under H<=4 diagnostics.

If D1-E is deferred or fails, the conclusion must be narrowed to "relative to the implicit AdaLN baseline only."

Not permitted after D1:

- "planning works";
- "confidence-aware CEM works";
- "Mamba discovers skills/options";
- "the method is proven as a general world model";
- "long-horizon H>4 behavior is solved".

---

## 10. Time Estimate

Reference: Stage-2c 15-run prediction-only full scan took about 12 h 52 m wall-clock on six GPUs, with average train step roughly 280-300 ms.

Expected D1 cost, assuming similar per-run cost:

| Block | New training runs | Rough wall time on 6 GPUs |
|---|---:|---:|
| local implementation/tests | 0 | 1-2 days engineering |
| smoke | 2 | 2-4 hours including diagnostics |
| baseline re-diagnostics | 0 training / 15 diagnostics | 3-8 hours, diagnostic-bound |
| D1-A | 75 | 2.5-3.5 days |
| D1-B | about 90 per selected model | 3-4.5 days |
| D1-C | 75 | 2.5-3.5 days |
| D1-D | 60 | 2-3 days |
| D1-E | 15-30 | 0.5-1.5 days |

Total if the full staged matrix proceeds for one selected D1-B model: roughly 10-16 days of GPU wall time, plus analysis and possible retries. A fully exhaustive ungated expansion can exceed this. Quality gates take priority over runtime compression.

---

## 11. Handoffs

Before implementation:

1. Claude reviews this revised plan in a new session using an English prompt.
2. If Claude returns `REVIEW PASS` or a non-blocking `PASS-WITH-FLAGS`, Codex may implement only section 5 files after user approval.
3. If Claude requests changes, Codex edits this plan only unless Claude explicitly says code changes are needed.

Before remote smoke:

1. Codex provides change summary, tests, scope check, and scientific risk summary.
2. Claude code review passes the implementation.
3. User approval is required for remote GPU launch under current project guardrails.

After runs:

1. Codex pulls artifacts and drafts result analysis/gate doc.
2. Claude ratifies result gate.
3. Claude owns `notes/alignment_log.md` and Notion sync.

---

## 12. Claude Review Questions

Claude should review:

1. Does this D1 plan correctly preserve the user's main idea: dynamics trend first, visual latent refinement second?
2. Is `F_dyn` allowed to read compact visual state without collapsing back into original JEPA-WMS implicit prediction?
3. Are D1-A/B/C/D/E sufficient to prove or falsify "dynamics-visual decoupling" rather than just capacity gain?
4. Are the moved-region metric, cross-position test, and leakage probes rigorous enough?
5. Are the effect-size gate, paired test, and multiple-comparison correction defensible with 3 seeds?
6. Is the authorized file list too broad, too narrow, or scientifically risky?
7. Are D2 confidence and D3 CEM correctly excluded from D1 claims?

Requested Claude output:

```text
VERDICT: REVIEW PASS / PASS-WITH-FLAGS / BLOCKED
BLOCKERS:
FLAGS:
REQUIRED PLAN EDITS:
OPTIONAL SUGGESTIONS:
CLAIM BOUNDARY:
```
