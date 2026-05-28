# Lance Backend Equivalence Validation - 2026-05-29

## Context

This pass validates whether the Lance data backend can be used as the trusted data source for the Phase-1 training-time estimates, without running 50-epoch training. It covers PointMaze, Wall, PushT, and Metaworld data/backend equivalence plus short loss-trajectory agreement for Maze and MW. It does not claim Terver success-rate equivalence or full-training checkpoint equivalence.

Timing table being validated, from `experiments/results/training_time_profile_20260528.{csv,md}`:

| env | raw step ms | swm_lance step ms | speedup | raw CV % | lance CV % | status |
|---|---:|---:|---:|---:|---:|---|
| maze | 848.5 | 231.2 | 3.67x | 1.30 | 0.17 | stable |
| mw | 841.1 | 236.8 | 3.55x | 1.93 | 0.09 | stable |
| pusht | 1384.5 | 230.2 | 6.01x | 7.22 | 0.20 | raw needs_review |
| wall | 307.1 | 235.9 | 1.30x | 1.12 | 0.44 | stable |

Canonical validation environment: lab01, worktree `$JEPAWM_HOME/worktrees/phase1-timing-profile`, commit `b5aa861`.

## Tier A - Static/Test Validation

Command:

```bash
source ~/.jepawm_env
source /home/ps/miniconda3/etc/profile.d/conda.sh
conda activate jepa-wms
cd "$JEPAWM_HOME/worktrees/phase1-timing-profile"

python -m pytest \
  tests/datasets/test_lance_equivalence_point_maze.py \
  tests/datasets/test_lance_equivalence_wall.py \
  tests/datasets/test_lance_equivalence_pusht.py \
  tests/datasets/test_lance_equivalence_metaworld.py \
  tests/datasets/test_lance_codec_authority.py \
  tests/datasets/test_lance_equivalence_point_maze_jpeg.py \
  -q
```

Result:

```text
HEAD=b5aa861
29 passed, 1 xfailed, 12 warnings in 25.10s
```

The only expected non-pass item is the JPEG variant xfail. There were no skips. The warnings are Lance fork-safety warnings from the Lance package.

Diff hygiene:

```text
forbidden paths changed: none
changed_files=40
```

Forbidden path filter:

```text
app/vjepa_wm/
src/models/
src/losses/
evals/
src/distributed.py
src/checkpoint.py
app/vjepa_wm/train.py
```

## Tier B - Short Smoke-Train Loss Comparison

Runs used 6 GPUs (`cuda:0` through `cuda:5`), `BATCH_SIZE=32`, `NUM_WORKERS=16`, `PROFILE_WARMUP=30`, `PROFILE_STEPS=300`, and `PROFILE_IPE=360`. The raw/Lance pair differs only in backend fields after ignoring the run `folder`.

Run IDs:

```text
20260529_lance_equiv_maze_raw
20260529_lance_equiv_maze_lance
20260529_lance_equiv_mw_raw
20260529_lance_equiv_mw_lance
```

Config hygiene:

| env | config equal ignoring folder/backend | raw backend | lance backend |
|---|---|---|---|
| maze | true | `kind=raw` | `kind=swm_lance`, `lance_uri=$JEPAWM_DSET/_lance_20260528/PointMaze.lance` |
| mw | true | `kind=raw` | `kind=swm_lance`, `lance_uri=$JEPAWM_DSET/_lance_20260528/Metaworld.lance` |

Loss comparison reads `log_r0.csv`, aligns by `(epoch, itr)`, discards the first 30 warmup rows, and compares the following 300 measured rows. Timing columns are intentionally not compared.

| env | rows | step mismatch | first5 max abs | mean abs diff | max abs diff | last abs diff | monotonic nondecreasing |
|---|---:|---|---:|---:|---:|---:|---|
| maze | 300 | none | 0.000010000 | 0.000088800 | 0.001750000 | 0.000030000 | false |
| mw | 300 | none | 0.000000000 | 0.000126467 | 0.001030000 | 0.000100000 | false |

First five measured losses:

| env | raw | swm_lance |
|---|---|---|
| maze | `[1.24794, 1.24677, 1.24481, 1.24453, 1.24116]` | `[1.24794, 1.24676, 1.24481, 1.24453, 1.24116]` |
| mw | `[1.07581, 1.08662, 1.08183, 1.08227, 1.10542]` | `[1.07581, 1.08662, 1.08183, 1.08227, 1.10542]` |

Pass criteria:

| criterion | maze | mw |
|---|---|---|
| first five measured losses within `atol=1e-4, rtol=1e-4` | pass | pass |
| mean absolute loss diff over 300 rows `< 1e-3` | pass | pass |
| no monotonic growth trend in diff trace | pass | pass |

## Tier C - Real-Dataset 256-Window Audit

Helper script added:

```text
src/scripts/lance_equivalence_window_audit.py
```

The script is read-only, imports only dataset loader paths plus `read_metadata`, and does not import Lance writers, model, loss, optimizer, eval, or training code.

Commands:

```bash
source ~/.jepawm_env
source /home/ps/miniconda3/etc/profile.d/conda.sh
conda activate jepa-wms
cd "$JEPAWM_HOME/worktrees/phase1-timing-profile"
export JEPAWM_DSET_LANCE="$JEPAWM_DSET/_lance_20260528"

python src/scripts/lance_equivalence_window_audit.py --env maze  --raw-path "$JEPAWM_DSET/point_maze"      --lance-uri "$JEPAWM_DSET_LANCE/PointMaze.lance"  --n-windows 256 --seed 0
python src/scripts/lance_equivalence_window_audit.py --env wall  --raw-path "$JEPAWM_DSET/wall_single"    --lance-uri "$JEPAWM_DSET_LANCE/Wall.lance"       --n-windows 256 --seed 0
python src/scripts/lance_equivalence_window_audit.py --env pusht --raw-path "$JEPAWM_DSET/pusht_noise"    --lance-uri "$JEPAWM_DSET_LANCE/PushT.lance"      --n-windows 256 --seed 0
python src/scripts/lance_equivalence_window_audit.py --env mw    --raw-path "$JEPAWM_DSET/Metaworld/data" --lance-uri "$JEPAWM_DSET_LANCE/Metaworld.lance"  --n-windows 256 --seed 0
```

Result from `phase1_lance_equiv_tierC_20260529_rerun.log`:

```text
OK 256/256 windows match (env=maze, codec=png)
OK 256/256 windows match (env=wall, codec=raw_float32)
OK 256/256 windows match (env=pusht, codec=png)
OK 256/256 windows match (env=mw, codec=png)
```

The first Tier C run exposed a reporting bug in the audit script: PushT stores metadata in `train.lance/` and `val.lance/`, not at the split root. The fix is commit `b5aa861`; the rerun above is the passing evidence.

## Pass/Fail Matrix

| tier | scope | result | evidence |
|---|---|---|---|
| A | tiny fixture equivalence, codec authority, JPEG xfail, diff hygiene | pass | `29 passed, 1 xfailed`, no forbidden path diff |
| B | Maze and MW short loss trajectory, 300 measured rows | pass | first5 loss max `<=1e-5`, mean abs diff `<1.3e-4`, no monotonic growth |
| C | real dataset 256 sampled train windows per env | pass | all four envs report `OK 256/256` |

## Remaining Risks

- This does not prove Terver success-rate equivalence.
- This does not prove 50-epoch checkpoint or final-policy equivalence.
- PushT raw timing remains noisy (`raw needs_review` in the timing table), although the Lance data windows matched exactly in this audit.
- Lab01 runs emitted DataLoader worker cleanup warnings after completed training runs; `log_r0.csv` and profiler outputs were complete, so this was not treated as a data-equivalence failure.

## Recommendation

The Lance timing numbers in `experiments/results/training_time_profile_20260528.{csv,md}` are safe to cite as train-time estimates for the measured configs. State the boundary explicitly: this validates data/backend equivalence and short-run loss alignment, not Terver success-rate equivalence or long-run checkpoint identity.
