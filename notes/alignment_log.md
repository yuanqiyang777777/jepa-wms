# Alignment Log

## 2026-05-29 - Lance backend equivalence validation

Validated the Phase-1 Lance data backend on lab01 without a 50-epoch run. Tier A fixture tests passed (`29 passed, 1 xfailed`), Tier B Maze/MW short loss trajectories matched within the planned tolerances, and Tier C real-dataset 256-window audits passed for Maze, Wall, PushT, and MW. Full report: `experiments/results/lance_equivalence_validation_20260529.md`. Boundary: this supports using `training_time_profile_20260528.{csv,md}` as train-time estimates, but it does not prove Terver success-rate or long-run checkpoint equivalence.
