# Lance Backend Equivalence Validation - 2026-05-29

## Context

This pass validates whether the Lance data backend can be used as the trusted data source for the Phase-1 training-time estimates, without running 50-epoch training. It covers PointMaze, Wall, PushT, and Metaworld data/backend equivalence. It does not claim Terver success-rate equivalence or full-training checkpoint equivalence.

Revision note after independent audit: this report no longer treats the raw-to-Lance mean-wall speedup ratios as clean algorithmic speedups, and no longer treats Tier B as a loss-trajectory proof. The strongest supported claims are: Lance data-window equivalence is robust, Lance absolute step times are stable, and raw mean-wall timings in this profile are heavily tail-latency affected.

Timing table being validated, from `experiments/results/training_time_profile_20260528.{csv,md}`. `mean step` is the wall-clock mean used by `fifty_epoch_estimate_h`; `median step` shows the typical-step behavior.

| env | raw mean step ms | raw median step ms | raw p95 ms | Lance mean step ms | Lance median step ms | mean-wall ratio | median-step ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| maze | 848.5 | 237.9 | 4943.6 | 231.2 | 231.1 | 3.67x | 1.03x |
| mw | 841.1 | 244.0 | 4159.6 | 236.8 | 236.8 | 3.55x | 1.03x |
| pusht | 1384.5 | 233.3 | 8745.1 | 230.2 | 230.1 | 6.01x | 1.01x |
| wall | 307.1 | 262.8 | 337.2 | 235.9 | 230.5 | 1.30x | 1.14x |

Interpretation: the Lance absolute step time is stable and repeatable. The raw mean-wall baseline is tail-heavy, with multi-second p95 events in Maze, MW, and PushT. Therefore the mean-wall speedup ratios are best described as observed under this profiling condition and as an upper-bound view of wall-clock savings under raw tail latency. They should not be presented as guaranteed per-step algorithmic speedups.

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

## Tier B - Short Smoke-Train Metric Comparison

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

Important audit correction: `log_r0.csv` has a header/value ordering issue in these historical runs. The values previously reported as "loss" match the logged `act_max` / `train_rollout/visual_l1_loss/1` position rather than the scalar training loss. The literal `loss` column is degenerate zeros, and fields like `act_max < act_mean < act_min` are mathematically inconsistent, which confirms the logger/header issue is in the file itself. A follow-up code fix aligns CSV headers for future runs, but these historical logs are not rewritten.

Because of that, Tier B is not used as a proof of loss-trajectory equivalence. It remains useful for checking that the raw/Lance runs were aligned by `(epoch, itr)` and that selected logged scalar positions do not show monotonic raw-vs-Lance divergence. Tier A and Tier C carry the data-equivalence claim.

The earlier report's compared scalar position:

| env | rows | step mismatch | first5 max abs | mean abs diff | max abs diff | last abs diff | monotonic nondecreasing |
|---|---:|---|---:|---:|---:|---:|---|
| maze | 300 | none | 0.000010000 | 0.000088800 | 0.001750000 | 0.000030000 | false |
| mw | 300 | none | 0.000000000 | 0.000126467 | 0.001030000 | 0.000100000 | false |

First five values from that scalar position:

| env | raw | swm_lance |
|---|---|---|
| maze | `[1.24794, 1.24677, 1.24481, 1.24453, 1.24116]` | `[1.24794, 1.24676, 1.24481, 1.24453, 1.24116]` |
| mw | `[1.07581, 1.08662, 1.08183, 1.08227, 1.10542]` | `[1.07581, 1.08662, 1.08183, 1.08227, 1.10542]` |

Recomputed named rollout-loss column, using `train_rollout/loss/1` from the same 300 measured rows:

| env | first5 max abs | mean abs diff | max abs diff | last abs diff | monotonic nondecreasing | status under original `<1e-3` criterion |
|---|---:|---:|---:|---:|---|---|
| maze | 0.000000000 | 0.000456267 | 0.008880000 | 0.000720000 | false | pass |
| mw | 0.000100000 | 0.002718100 | 0.021820000 | 0.003220000 | false | fail |

Interpretation: the MW named rollout-loss diff fails the originally planned mean-abs `<1e-3` threshold, but the non-monotonic trace and Tier C's exact data-window match point to train-side nondeterminism / logging noise rather than a data backend mismatch. Do not cite Tier B as "loss equivalence passed."

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
| B | Maze and MW short run log alignment / scalar comparison | limited | configs and steps align, but CSV header/value ordering makes this inconclusive as a loss proof |
| C | real dataset 256 sampled train windows per env | pass | all four envs report `OK 256/256` |

## Remaining Risks

- This does not prove Terver success-rate equivalence.
- This does not prove 50-epoch checkpoint or final-policy equivalence.
- Raw mean-wall timing is tail-latency affected in Maze, MW, and PushT; speedup ratios from mean wall-clock should be reported as observed profile results, not guaranteed algorithmic speedups.
- PushT raw timing remains especially noisy (`raw needs_review` in the timing table), although the Lance data windows matched exactly in this audit.
- `log_r0.csv` has a header/value ordering issue in these historical Tier B runs; Tier B should not be used as a literal loss-equivalence proof. Future runs use the fixed CSV schema, but this report intentionally does not rewrite old logs.
- Lab01 runs emitted DataLoader worker cleanup warnings after completed training runs; `log_r0.csv` and profiler outputs were complete, so this was not treated as a data-equivalence failure.

## Recommendation

The Lance absolute timing numbers in `experiments/results/training_time_profile_20260528.{csv,md}` are safe to cite as Lance train-time estimates for the measured configs. The mean-wall raw-to-Lance ratios can be cited only with the tail-latency caveat: they describe the observed profile condition, where raw runs had severe long-tail stalls, and should not be presented as guaranteed algorithmic speedups.

For one serial four-env 50-epoch pass, the mean-wall estimates are:

| backend | estimate h | estimate days | interpretation |
|---|---:|---:|---|
| raw mean-wall | 263.8 | 11.0 | includes severe raw tail latency |
| Lance mean-wall | 51.9 | 2.16 | stable absolute Lance estimate |
| raw median-typical-step | 53.0 | 2.21 | typical-step view, removes most raw tail stalls |
| Lance median-typical-step | 51.9 | 2.16 | near identical to Lance mean-wall |

State the boundary explicitly: this validates data/backend equivalence and stable Lance absolute timing, not Terver success-rate equivalence, long-run checkpoint identity, or clean raw-to-Lance algorithmic speedup.
