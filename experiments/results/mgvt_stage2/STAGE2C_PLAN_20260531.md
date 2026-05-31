# MGVT-JEPA Stage-2c Plan

Date: 2026-05-31

Author: Claude (plan owner). Executor: Codex (plan-scoped).

Branch: `exp/20260530-mgvt-stage2`

Worktree: `C:\Users\17695\.config\superpowers\worktrees\jepa-wms\exp-20260529-mgvt-v1`

Precedent: `experiments/results/mgvt_stage2/STAGE2B_PLAN_20260530.md` (plan),
`experiments/results/mgvt_stage2/stage2b_1seed_20260531/MGVT_STAGE2B_GATE_20260531.md`
(ratified Gate-4).

---

## 0. What this plan authorizes and what it does not

This plan **authorizes Codex to create and edit exactly the files enumerated in
section 3, and nothing else**. Everything outside that list stays review-only;
Codex must stop and ask before touching it (AGENTS.md binding rule).

This plan **does not authorize any remote GPU launch.** Implementation, local
non-GPU checks, and a Claude code-review pass make a run *eligible*, not launched.
Remote GPU execution requires an explicit human `go` (section 8 and section 13).

Out of scope for Stage-2c entirely (hard-gated on a separate approved plan):
CEM, Terver planning evaluation, H=8 or any horizon > 4 training, Stage-3,
Granular, the flagged PointMaze Mamba 3-seed recheck, and any ConvMixer/Mamba
Stage-2c arm. Stage-2c is **AdaLN(depth=1), prediction-only, five tasks,
three seeds**. Lance stays a validated data-IO backend only; it is never
described as an algorithmic contribution.

---

## 1. Goal and scientific question

**Goal.** Convert the Stage-2b Gate-4 *provisional one-seed* breadth verdict into
a *confirmed three-seed* prediction-only result for `AdaLN(depth=1)` on all five
Phase-1 tasks, with explicit cross-seed variance, before any planning, CEM, H=8,
or Stage-3 work is considered.

**Primary scientific question.** Does `AdaLN(depth=1)`'s H=1..4 prediction skill
stay strictly positive and non-collapsing across **three independent seeds per
task**, and is the moved-patch skill at the training horizon
(`change_skill[4]`) *robust* (3-seed `mean − std > 0`) rather than a single-seed
artifact?

**Secondary questions (report-only, not gate deciders).**
- Is PointMaze's known softening profile (raw skill peaks ~H=2, decays through
  H=6) stable across seeds, i.e. does any seed cross into `skill[4] ≤ 0`?
  PointMaze is AdaLN's soft spot from Stage-2b.
- Does AdaLN keep its large MetaWorld proprio separation (Stage-2b
  `proprio_skill[4]` ≈ 0.94–0.95 on MW-R/MW-RW) across seeds?
- Sanity reproduction: the Stage-2c seed that equals the Stage-2b seed
  (PushT/Wall/Maze `234`; MW-R/MW-RW `1`) should reproduce the Stage-2b H=4
  numbers within run-to-run noise, because the architecture and data IO are
  byte-identical (section 4). A large divergence is an integrity flag, not a
  science result.

**Explicitly NOT asked by Stage-2c.** Planning success rate; any ranking of
AdaLN vs Mamba/ConvMixer; behaviour beyond H=4; anything requiring CEM or the
Terver eval.

---

## 2. Exact run matrix and seeds

- Backbone: **`AdaLN(depth=1)` only.**
- Tasks: PushT, Wall, PointMaze (Maze), MW-R, MW-RW.
- Seeds per task family (the Stage-1 convention; the Stage-2b seed is included
  as the first of the three so the Gate-4 point is a clean subset of Gate-5):
  - PushT / Wall / Maze: `234 235 236`
  - MW-R / MW-RW: `1 2 3`
- **Matrix: 5 tasks × 3 seeds × 1 backbone = 15 training runs.**
- Per run (inherited verbatim from the Stage-2b AdaLN configs):
  - W = 2 context (`data.custom.num_hist = 2`).
  - H = 4 training (`data.custom.num_pred = 4`, `rollout_cfg.rollout_steps = 4`,
    `ctxt_window_train_rollout = 2`).
  - Prediction-only (`evals: null`, `eval_freq: 999999`,
    `plan_only_eval_mode: false`).
  - Frozen DINOv2-S/14 encoder (`freeze_encoder: true`), normalized latent
    targets (`normalize_reps: true`), `rollout_stop_gradient: true`.
  - Proprio enabled (`proprio_emb_dim: 16`, `proprio_encoder_inpred: false`).
  - 10 epochs × 1000 iterations, ImageNet normalization, Lance backend with
    `fall_back_to_raw_if_unsupported: false`.
  - DDP world size 6 (`tasks_per_node = 6`), one run at a time.
- Diagnostics: `skill_score --max-horizon 6` ⇒ H = 1..6 reported; **Gate-5
  reads H = 1..4** (identical horizon window to Stage-2b Gate-4).

W = 2 is deliberately the Stage-2a/2b prediction-only context, **not** the
Terver planning W = 3. This is an intentional continuation, not a drift; do not
"fix" it.

**Per-task dataset binding (unchanged from Stage-2b; restated for review).**

| Task | `datasets` | Lance URI | `filter_tasks` | val camera | seeds |
| --- | --- | --- | --- | --- | --- |
| PushT | `[PushT]` | `${JEPAWM_DSET_LANCE}/PushT.lance` | null | null | 234 235 236 |
| Wall | `[Wall]` | `${JEPAWM_DSET_LANCE}/Wall.lance` | null | null | 234 235 236 |
| Maze | `[PointMaze]` | `${JEPAWM_DSET_LANCE}/PointMaze.lance` | null | null | 234 235 236 |
| MW-R | `[METAWORLD_HF]` | `${JEPAWM_DSET_LANCE}/Metaworld.lance` | `[mw-reach]` | `exterior_image_2_left` | 1 2 3 |
| MW-RW | `[METAWORLD_HF]` | `${JEPAWM_DSET_LANCE}/Metaworld.lance` | `[mw-reach-wall]` | `exterior_image_2_left` | 1 2 3 |

**Runtime budget (estimate, for scheduling only).** AdaLN H=4 step time was
262–307 ms in Stage-2b ⇒ ~50 min training + ~5–15 min skill scoring per run ⇒
**~13–16 h wall for all 15 runs, sequential, on one 6-GPU node.** Run detached
(tmux/`nohup`). This is a bounded overnight batch.

---

## 3. Files Codex is allowed to edit

These are the **only** files this plan authorizes Codex to create or modify.

**A. New code/config Codex creates (plan→code handoff):**

1. `experiments/scripts/gen_stage2c_configs.py` — Stage-2c config generator
   (section 4).
2. `configs/vjepa_wm/mgvt_stage2c/pusht_stage2c_pred_adaln_depth1_lance_h4.yaml`
   *(generated)*
3. `configs/vjepa_wm/mgvt_stage2c/wall_stage2c_pred_adaln_depth1_lance_h4.yaml`
   *(generated)*
4. `configs/vjepa_wm/mgvt_stage2c/maze_stage2c_pred_adaln_depth1_lance_h4.yaml`
   *(generated)*
5. `configs/vjepa_wm/mgvt_stage2c/mw_r_stage2c_pred_adaln_depth1_lance_h4.yaml`
   *(generated)*
6. `configs/vjepa_wm/mgvt_stage2c/mw_rw_stage2c_pred_adaln_depth1_lance_h4.yaml`
   *(generated)*
7. `experiments/scripts/mgvt_stage2c_scan.sh` — AdaLN-only, 3-seed scan launcher
   (section 5).
8. `tests/configs/test_mgvt_stage2c_configs.py` — contract test (section 6).

The five YAMLs (2–6) must be produced **by running the generator**, never
hand-edited, so they are reproducible and the contract test is meaningful.

**B. Result/record files Codex drafts (results→decision handoff, post-run):**

9. `experiments/results/mgvt_stage2/stage2c_3seed_<date>/` — pulled-back
   artifacts + `MGVT_STAGE2C_GATE_<date>.md` (**draft** gate verdict; Codex
   cites artifacts and marks the verdict provisional-until-ratified).
10. `experiments/registry.md` — **append** 15 rows only (section 12). No
    rewriting of existing rows.

**C. Claude-only (Codex must NOT write these):**

- `notes/alignment_log.md` — final ratified Gate-5 entry (Claude).
- This plan and the final Gate-5 verdict — Claude ratifies.
- Notion sync — Claude.

**D. Explicitly review-only (Codex must stop and ask before any change):**

- `experiments/scripts/gen_stage2b_configs.py`, `mgvt_stage2b_scan.sh`,
  `mgvt_stage1_scan.sh`, `_common.sh`, `_naming.md`.
- Any `configs/vjepa_wm/mgvt_stage1/**`, `mgvt_stage2/**`, `mgvt_stage2b/**`
  (these are frozen prior-stage configs and the Stage-2c generator's source).
- Any file under `app/`, `src/`, `evals/`, or any other config tree.
- Any existing Stage-1/2a/2b artifact directory.

---

## 4. Config generation strategy

`gen_stage2c_configs.py` produces a **pure seed-replication** of the Stage-2b
primary arm. For each task it deep-copies the corresponding Stage-2b AdaLN
config and changes **only the stage tag**:

- Source (read-only): `configs/vjepa_wm/mgvt_stage2b/{task}_stage2_pred_adaln_depth1_lance_h4.yaml`
- Output: `configs/vjepa_wm/mgvt_stage2c/{task}_stage2c_pred_adaln_depth1_lance_h4.yaml`
- The only edits:
  - `folder`: `.../mgvt_stage2b/{stem2b}` → `.../mgvt_stage2c/{stem2c}`
  - `checkpoint_folder`: same `mgvt_stage2b` → `mgvt_stage2c` retag
  - (stem `_stage2_` → `_stage2c_` inside those two paths)
- **Nothing else changes**: no model, predictor, optimizer, rollout, data,
  backend, normalization, proprio, or seed field is touched.

Rationale: identical architecture and data IO guarantee Stage-2c is a clean
3-seed superset of the Stage-2b Gate-4 evidence, so the secondary "sanity
reproduction" check in section 1 is valid.

Seed handling: the checked-in YAML keeps the Stage-2b default seed (`234` for
PushT/Wall/Maze, `1` for MW) as a placeholder. **The three seeds per task are
injected at launch** by the scan's `make_config` step, which overwrites both
`meta.seed` and `data.seed`. Configs are checked in (no dynamic generation at
run time), matching Stage-2b.

The generator must:
- be idempotent and emit exactly 5 files;
- assert at generation time that each emitted config equals its Stage-2b source
  except for `folder` and `checkpoint_folder` (fail loudly otherwise);
- print each path it writes.

Reuse `src.utils.yaml_utils` / `ruamel.yaml` as `gen_stage2b_configs.py` does;
do not introduce a new YAML library.

---

## 5. Launcher / run commands for lab01

`mgvt_stage2c_scan.sh` is a minimal edit of `mgvt_stage2b_scan.sh` with these
differences and nothing more:

- `CONFIGS` lists **only the 5 Stage-2c AdaLN YAMLs**.
- Default seed lists: `PUSHT_SEEDS="234 235 236"`, `WALL_SEEDS="234 235 236"`,
  `MAZE_SEEDS="234 235 236"`, `MW_R_SEEDS="1 2 3"`, `MW_RW_SEEDS="1 2 3"`.
- Task token parse handles the new stage tag: `task="${stem%%_stage2c_*}"`.
- Default `RUN_ROOT="$JEPAWM_LOGS/mgvt_stage2c_3seed"`,
  `CKPT_ROOT="$JEPAWM_CKPT/mgvt_stage2c_3seed"`, `SKILL_MAX_HORIZON=6`,
  `REQUIRE_LANCE_STORES=1`.
- Keep the Stage-2b health check, per-run provenance snapshot
  (`launcher.sh`, `git_commit.txt`, generated `config.yaml`), `skill_score`
  call (`--max-horizon "$SKILL_MAX_HORIZON"`, `--num-workers 0`,
  `--train-log-csv .../log_r0.csv`), and Lance preflight (4 stores) **verbatim**.
- Aggregator: same per-`(task, model, horizon)` mean/`stdev` logic as Stage-2b,
  renamed outputs `summary/mgvt_stage2c_horizon_{raw,summary}.csv` and
  `..._summary.md`, **plus three derived columns** per `(task, horizon)`:
  `skill_mean_minus_std`, `change_skill_mean_minus_std`,
  `proprio_skill_mean_minus_std` (= mean − sample std, used directly by Gate-5
  C3). `mamba_backend` will be empty (AdaLN only); leave the column.

Do not change DDP, NCCL, or GPU-guard behaviour: the script must keep
`source _common.sh` (which sets `MUJOCO_GL=egl`, `NCCL_P2P_DISABLE=1`, `cd`
repo root, conda activate), `check_gpus_free`, and `WORLD_SIZE > 6` refusal.

**Commands.**

Local (no GPU) — generate + validate (section 6):
```bash
python experiments/scripts/gen_stage2c_configs.py
python -m pytest tests/configs/test_mgvt_stage2c_configs.py -q
bash -n experiments/scripts/mgvt_stage2c_scan.sh
```

Remote smoke (lab01, **only after** Claude review + user `go`) — 1 config,
2-epoch quick-debug, single seed, isolated roots:
```bash
QUICK_DEBUG=1 \
CONFIG_FILTER=wall_stage2c_pred_adaln_depth1_lance_h4 \
WALL_SEEDS=234 PUSHT_SEEDS="" MAZE_SEEDS="" MW_R_SEEDS="" MW_RW_SEEDS="" \
RUN_ROOT="$JEPAWM_LOGS/mgvt_stage2c_smoke" \
CKPT_ROOT="$JEPAWM_CKPT/mgvt_stage2c_smoke" \
bash experiments/scripts/mgvt_stage2c_scan.sh
```

Remote full scan (lab01, **only after** Claude review + user `go`) — detached:
```bash
RUN_ROOT="$JEPAWM_LOGS/mgvt_stage2c_3seed_<date>" \
CKPT_ROOT="$JEPAWM_CKPT/mgvt_stage2c_3seed_<date>" \
DEVICES="cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5" \
bash experiments/scripts/mgvt_stage2c_scan.sh
```
GPU policy: 6 cards, **avoid GPU 6**; if any of 0–5 is busy, pick from
`{0,1,2,3,4,5,7}` and pass them via `DEVICES`. `_common.sh` handles
`source ~/.jepawm_env`, conda activation, repo-root `cd`, and
`NCCL_P2P_DISABLE=1`.

---

## 6. Required local checks before remote launch

Codex runs these on the login node / locally (no GPU) and reports exact commands
and outcomes to Claude:

1. `python experiments/scripts/gen_stage2c_configs.py` writes exactly 5 YAMLs and
   prints the equivalence assertion as satisfied.
2. `git status` shows **only** the 8 enumerated section-3A files as new/modified
   — nothing else.
3. `python -m pytest tests/configs/test_mgvt_stage2c_configs.py -q` passes. The
   test asserts (mirroring `test_mgvt_stage2b_configs.py`, adapted to Stage-2c):
   - exactly 5 configs, one per task, all backbone token `adaln_depth1`,
     pred_type `AdaLN`;
   - per-task `datasets`, `seed`, `filter_tasks`, val camera, Lance URI suffix,
     ImageNet norm;
   - `backend.kind == swm_lance`, `fall_back_to_raw_if_unsupported is False`;
   - `num_hist == 2`, `num_pred == 4`, `num_frames_pred == 6`,
     `normalize_reps is True`, `proprio_emb_dim == 16`,
     `proprio_encoder_inpred is False`, `pred_depth == 1`,
     `init_scale_factor_adaln == 0`, `rollout_steps == 4`,
     `ctxt_window_train_rollout == 2`, `rollout_stop_gradient is True`,
     `do_sequential_rollout is True`, `do_parallel_rollout is False`,
     `evals is None`, `num_epochs == 10`;
   - **equivalence clause**: each Stage-2c config equals the matching Stage-2b
     AdaLN config except `folder`/`checkpoint_folder` (the core Stage-2c
     invariant).
4. `bash -n experiments/scripts/mgvt_stage2c_scan.sh` (and `shellcheck` if
   available) is clean; confirm `CONFIGS` has 5 entries that match the generated
   files and the task parse yields `{pusht, wall, maze, mw_r, mw_rw}`.
5. Capture `git rev-parse HEAD` and confirm a clean tree apart from the
   enumerated files.
6. Lance preflight (existence only, no build): confirm
   `${JEPAWM_DSET_LANCE}/{PushT,Wall,PointMaze,Metaworld}.lance` all exist
   (reuse the Stage-2b stores; `Wall.lance` should already be present). **Do not
   build or convert anything** under this plan; if a store is missing, stop and
   report — do not silently fall back to raw.

No GPU work occurs in this section.

---

## 7. Claude code-review checklist before GPU launch

Claude performs this review on the committed-but-unlaunched code. A pass makes
the run **eligible**, not launched.

- **Scope**: diff touches only the 8 section-3A files; no source/`_common.sh`/
  prior-stage config drift.
- **Config equivalence**: each Stage-2c config == Stage-2b AdaLN source except
  `folder`/`checkpoint_folder`. Spot-check the frozen knobs: `freeze_encoder`,
  `normalize_reps`, `fall_back_to_raw_if_unsupported:false`, `num_hist:2`,
  `num_pred:4`, `num_frames_pred:6`, `rollout_steps:4`,
  `ctxt_window_train_rollout:2`, `rollout_stop_gradient:true`,
  `do_sequential_rollout:true`/`do_parallel_rollout:false`, `evals:null`,
  `num_epochs:10`, `pred_type:AdaLN`, `pred_depth:1`,
  `init_scale_factor_adaln:0`, `proprio_emb_dim:16`,
  `proprio_encoder_inpred:false`, ImageNet norm, correct
  datasets/filter_tasks/val_camera/lance_uri per task.
- **Seed wiring**: scan injects 3 seeds/task; both `meta.seed` and `data.seed`
  set per seed; run-id carries the seed; the Stage-2b seed (234/1) is seed #1;
  no seed collisions; no cross-seed checkpoint reuse.
- **Stop-gradient / EMA / frozen-encoder integrity** unchanged from Stage-2b.
- **Diagnostic indexing**: `skill_score --max-horizon 6`; Gate-5 reads H=1..4;
  `--train-log-csv` wired to the run's `log_r0.csv`; deterministic-encode check
  present.
- **Aggregator**: per-`(task, horizon)` `n == 3`, mean + sample `stdev`; the
  three `*_mean_minus_std` columns computed correctly; raw CSV has one row per
  run×horizon (expect 15 × 6 = 90 rows); no cross-task pooling.
- **Lance**: no-fallback preserved; preflight present; `REQUIRE_LANCE_STORES`
  default 1.
- **No leakage / no eval**: `evals:null`, `eval_freq` huge,
  `plan_only_eval_mode:false`; no CEM/planning hook anywhere.
- **Provenance**: per-run `launcher.sh`, `git_commit.txt`, generated
  `config.yaml`; run-id uniqueness vs Stage-2b guaranteed by the `stage2c` tag.
- **GPU policy**: `WORLD_SIZE ≤ 6`, GPU-6 avoidance documented,
  `check_gpus_free` intact.
- Claude emits an explicit **REVIEW PASS** or **REVISIONS REQUIRED** verdict.

---

## 8. Human approval point for remote GPU launch

Sequencing (one active writer; never launch from a dirty tree):

1. Codex implements section 3A → runs section 6 local checks → reports.
2. Claude runs section 7 review → **REVIEW PASS** or **REVISIONS**.
3. On PASS, Codex commits the reviewed code to `exp/20260530-mgvt-stage2`
   (commit at handoff), pushes origin, and **STOPS**.
4. **Codex presents the eligible run to the user and waits for an explicit
   `go`.** Code-review eligibility is necessary but not sufficient.
5. Only after the user says `go`: Codex remote-pulls, runs the section-5 smoke,
   confirms health, then launches the full detached scan.

This plan, by itself, confers **no** launch authority.

---

## 9. Diagnostic aggregation requirements

For AdaLN only, computed from the 15 `skill_score.json` files:

- **Raw CSV** (`mgvt_stage2c_horizon_raw.csv`): one row per `(run_id, horizon)`
  for h=1..6, with `seed, skill, change_skill, proprio_skill, mse_model,
  mse_persist, proprio_mse_model, proprio_mse_persist, param_count,
  train_step_time_ms, deterministic_encode_max_abs_diff` (90 rows total;
  `mamba_backend` empty).
- **Summary CSV/MD** (`mgvt_stage2c_horizon_summary.{csv,md}`): per
  `(task, horizon)`, `n=3`, `mean` and sample `std` of skill / change_skill /
  proprio_skill, plus `skill_mean_minus_std`, `change_skill_mean_minus_std`,
  `proprio_skill_mean_minus_std`, plus `param_count_mean`,
  `train_step_time_ms_mean`, `seeds`.
- **Gate readouts** the Stage-2c gate doc must include:
  1. Per-task H=4 table: skill / change_skill / proprio_skill as `mean ± std`
     over 3 seeds.
  2. Per-task per-seed **positivity matrix** for h=1..4 (each of the 3 seeds ×
     {skill, change, proprio}: pass/fail), so Gate-5 C1 is auditable per seed.
  3. Per-task H=1..6 `mean ± std` curve (decay-profile; explicit PointMaze
     watch for any seed approaching `skill[4] ≤ 0`).
  4. The C3 margins at H=4: `mean − std` for skill, change_skill, proprio_skill.
  5. Sanity-reproduction line: Stage-2c seed-234/seed-1 H=4 vs Stage-2b H=4.
- **Integrity aggregates**: all 15 `deterministic_encode_max_abs_diff == 0.0`;
  all 15 `log_r0.csv` reach `epoch=10, itr=999` (10,001 rows); freeze /
  normalize / Lance-no-fallback / W2 / H4 confirmed from each generated config
  snapshot.
- **No pooling across tasks.** Stage-2c MW-RW (ImageNet norm) may be compared to
  Stage-2b MW-RW (also ImageNet) as same-protocol, but **must not** be pooled
  with the Stage-2a MW-RW `0.5/0.5`-norm run.

---

## 10. Frozen Gate-5 criteria

These thresholds are **frozen at plan time** and may not be relaxed after
results are seen. Any post-hoc change requires a new plan and explicit user
approval. `change_skill[4]` (moved-patch skill at the training horizon) is the
privileged metric, especially on Wall and PointMaze where static geometry can
inflate raw persistence skill.

**Integrity preconditions (per run; a violation makes the run invalid — fix and
rerun via `_retryN`, it is not a science verdict):**
- I1: reaches `epoch=10, itr=999`, 10,001 CSV rows, all-finite loss.
- I2: `deterministic_encode_max_abs_diff == 0.0`.
- I3: generated config snapshot preserves Lance no-fallback,
  `freeze_encoder:true`, `normalize_reps:true`, W=2 (`num_hist:2`), H=4
  (`num_pred:4`, `rollout_steps:4`), `evals:null`, `rollout_stop_gradient:true`,
  ImageNet norm, `pred_type:AdaLN`, `pred_depth:1`.
- I4: `skill_score.json` parses with finite H=1..6 skill/change/proprio.

**Per-task decision (AdaLN, n = 3 seeds):**

- **CONFIRM** requires all of:
  - **C1 — per-seed positivity**: for every seed `s` of the 3 and every
    `h ∈ {1,2,3,4}`: `skill[h,s] > 0` AND `change_skill[h,s] > 0` AND
    `proprio_skill[h,s] > 0`.
  - **C2 — no collapse**: for every seed, no NaN/inf and no `skill[h,s] ≤ 0` at
    any `h ≤ 4` (i.e. the negative-skill collapse signature is absent in every
    seed).
  - **C3 — robustness margin**: across the 3 seeds,
    `mean(skill[4]) − std(skill[4]) > 0` AND
    `mean(change_skill[4]) − std(change_skill[4]) > 0` AND
    `mean(proprio_skill[4]) − std(proprio_skill[4]) > 0`
    (sample std, n−1 dof; read directly from the `*_mean_minus_std` columns).
- **CONDITIONAL**: C1 and C2 hold (all three seeds individually positive, no
  collapse) but C3 fails on ≥1 metric. The task is *positively confirmed but
  high-variance*; the gate doc must name which metric(s) missed. CONDITIONAL
  does **not** by itself fail the stage, but such a task is **not** planning-ready
  without an explicit caveat in a future plan.
- **FAIL**: any seed violates C1 (some `skill/change/proprio[h] ≤ 0` at `h ≤ 4`)
  or C2 (collapse/NaN), or an integrity precondition I1–I4 cannot be remediated.

**Stage-level Gate-5 verdict:**
- **PASS** — all five tasks CONFIRM.
- **PASS-WITH-FLAGS** — all five tasks CONFIRM or CONDITIONAL, with ≥1
  CONDITIONAL (the doc lists the high-variance task(s) and the missed metric).
- **FAIL** — any task FAIL.

**Watch items (report-only, never a gate failure):** PointMaze per-seed decay
profile; MetaWorld proprio separation magnitude; the Stage-2b-seed sanity
reproduction. No ConvMixer/Mamba arm is run in Stage-2c, so **no backbone
ranking claim may be made** from these results.

---

## 11. Stop conditions

- **Stop after the Stage-2c Gate-5 verdict and report.** Do not proceed to any
  next step automatically.
- Do **not**, without a new approved plan and explicit user `go`: run CEM, the
  Terver planning eval, H=8 or any horizon > 4, Stage-3, Granular, a Stage-2c
  ConvMixer/Mamba arm, or the flagged PointMaze Mamba 3-seed recheck.
- Do **not** build/convert datasets, remove `NCCL_P2P_DISABLE=1`, or modify any
  file outside section 3.
- Do **not** touch or overwrite Stage-1/2a/2b artifacts, logs, or checkpoints.
- If any task FAILs Gate-5, **stop and report**; the only permitted reruns are
  documented infra-failure retries (`_retryN` / `_vN`, with a registry +
  alignment-log note), not silent re-rolls to chase a pass.
- One active writer at a time in the shared worktree; never launch from a dirty
  tree.

---

## 12. Provenance and registry / alignment-log requirements

**Per-run provenance (produced by the scan, verified before pull):**
`launcher.sh` snapshot, `git_commit.txt` (`git rev-parse HEAD` +
`git describe --dirty --always`), generated `config.yaml`, `log_r0.csv`,
`skill_scores/<run>/skill_score.{json,md}`. Capture the **remote run_root** and
**remote experiment commit** in the gate doc (as Stage-2b did).

**Artifact pull-back:** to workspace-root
`experiments/results/mgvt_stage2/stage2c_3seed_<date>/`, mirroring the Stage-2b
layout (`provenance/`, `skill_scores/`, `summary/`, `scan_*.txt`, `scan.log`,
and `MGVT_STAGE2C_GATE_<date>.md`). Validate all 15 JSON parse before drafting.

**Registry (`experiments/registry.md`): append 15 rows, one per run×seed.** Use
the established MGVT-track run-id form (per the Stage-2b alignment-log entry):
`<LAUNCHDATE>_mgvt_stage2c_<env>_adaln_seed<N>`, with
`env ∈ {pusht, wall, maze, mwr, mwrw}` and `<LAUNCHDATE>` = lab01 local launch
date. Columns:
- `date` = launch/completion date; `env` per the above; `method` = `mgvt`
  (MGVT-JEPA AdaLN predictor line — define this token here); `phase` = `main`;
  `seed` = the numeric seed; `status` = `done`/`failed`;
  `commit` = run snapshot commit; `config_path` = the generated config path;
  `raw_log_path` = the pulled scan-dir stem
  `experiments/results/mgvt_stage2/stage2c_3seed_<date>/.../{task}_stage2c_pred_adaln_depth1_lance_h4_seed<N>/`
  (this maps the canonical run-id to its artifacts); `wandb` = `n/a`
  (`use_wandb` is false); `success_rate_pct` = `n/a`; `vs_paper_target` = `n/a`;
  `protocol_compatible` = `yes`; `metric_source` = the `skill_score.json` path;
  `notes` = H=4 skill/change/proprio for that seed **and** the per-task Gate-5
  tier (CONFIRM/CONDITIONAL/FAIL). The `notes` must state **"prediction-only
  diagnostic; not a planning-success measurement."**

**Alignment log (`notes/alignment_log.md`): Claude-only, after ratification.**
Codex drafts the verdict inside the gate doc; Claude writes the canonical entry.
Skeleton:
```
<DATE> | result | MGVT-JEPA Stage-2c Gate-5 <PASS|PASS-WITH-FLAGS|FAIL> (3 seeds): AdaLN(depth=1) ...
        | Scan: 5 tasks x 1 backbone x 3 seeds, prediction-only, W=2/H=4 train, H=1..6 diag; commit <c>; run_root <remote>; seeds 234/235/236 (PushT/Wall/Maze), 1/2/3 (MW-R/MW-RW).
        | Per-task tier + change_skill[4] mean±std; PointMaze decay note; proprio separation; Stage-2b-seed reproduction check.
        | Verified by Claude: 15/15 epoch=10/itr=999; deterministic encode 0.0; config-equivalence to Stage-2b AdaLN; Lance no-fallback. Doc: .../MGVT_STAGE2C_GATE_<date>.md. Stop: no CEM/planning/H8/Stage-3 without explicit user approval.
```

**Notion:** Claude-only mirror; Codex does not write Notion run data.

---

## 13. Codex must not launch remote GPU jobs until the user says go

**Codex may NOT start any remote GPU job — smoke or full scan — until the user
gives an explicit `go`.** Implementation, local non-GPU checks (section 6), a
Claude code-review pass (section 7), a clean tree, and a committed branch are all
prerequisites, but **none of them — and not this plan — authorize launch.** The
only trigger for GPU execution is an explicit human `go` after the run is
presented as eligible (section 8). If in doubt, stop and ask.
