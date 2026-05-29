# Alignment Log

## 2026-05-29 - Lance backend equivalence validation

Validated the Phase-1 Lance data backend on lab01 without a 50-epoch run. Tier A fixture tests passed (`29 passed, 1 xfailed`) and Tier C real-dataset 256-window audits passed for Maze, Wall, PushT, and MW. After independent audit, Tier B is treated as limited because the historical `log_r0.csv` files have a header/value ordering issue and should not be cited as literal loss-equivalence proof; a follow-up code fix aligns CSV headers for future runs without rewriting old logs. Full report: `experiments/results/lance_equivalence_validation_20260529.md`. Boundary: this supports data/backend equivalence and stable Lance absolute train-time estimates; raw-to-Lance mean-wall speedups require a tail-latency caveat and this does not prove Terver success-rate or long-run checkpoint equivalence.
