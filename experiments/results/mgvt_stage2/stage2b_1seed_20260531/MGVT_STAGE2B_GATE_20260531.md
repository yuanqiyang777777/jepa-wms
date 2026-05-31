# MGVT-JEPA Stage-2b Gate-4 Result

Status: Claude-ratified Gate-4 result, provisional one-seed evidence.

Date: 2026-05-31

Ratification: Claude reviewed the draft, frozen plan, raw per-horizon CSV, source
JSONs, provenance snapshots, config snapshots, and run bracketing. Verdict:
**RATIFY WITH REVISIONS**. Revisions from Claude are incorporated below.

## Scope

- Stage: prediction-only H=4 breadth scan.
- Tasks: PushT, Wall, PointMaze, MW-R, MW-RW.
- Backbones: AdaLN(depth=1), MGVT ConvMixer, MGVT Mamba.
- Seeds: one seed per task family (`234` for PushT/Wall/Maze, `1` for MW-R/MW-RW).
- Diagnostics: H=1..6; Gate 4 uses H=1..4.
- Remote run root:
  `/home/ps/Code/yqy/DINO-WM/jepawm_logs/mgvt_stage2b_1seed_20260530`
- Remote experiment commit:
  `dde363e8d8a4cf8a262dc98f08eb66eb7f617e55`

This is a one-seed breadth gate. It is not a final scientific comparison and does
not authorize CEM, planning evaluation, H=8, Stage-3, or 3-seed confirmation.

## Validation

- All 15 `skill_score.json` files parse successfully.
- Every JSON contains finite H=1..6 visual skill, moved-patch skill, and proprio skill.
- Every deterministic DINO encode check is exactly `0.0`.
- Every Mamba result records the real backend: `mamba_ssm`.
- All 15 provenance snapshots use the same clean remote commit.
- Every `log_r0.csv` reaches `epoch=10, itr=999` with the expected 10,001 CSV rows.
- Every generated config preserves:
  - Lance backend with `fall_back_to_raw_if_unsupported: false`
  - W=2 context and H=4 rollout training
  - `num_pred=4`, `num_frames_pred=6`, `proprio_emb_dim=16`
  - frozen DINO latents with `normalize_reps: true`
  - rollout stop-gradient enabled
  - evaluation disabled during training

Completion is verified from CSV logs, checkpoint/score artifacts, run bracketing,
and finite diagnostics. Worker-abort/NCCL/TCPStore teardown warnings were observed
in remote runtime monitoring, but they are not present in the pulled artifact
bundle (`scan.log` is a launcher index and has no such warning lines). Therefore
the archived evidence supports "not truncated"; the teardown-warning ordering is
only a remote-observation inference unless per-run stderr is later archived.

## H=4 Readout

| Task | Backbone | Visual skill | Moved-patch skill | Proprio skill | Params | Step ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| PushT | AdaLN(depth=1) | 0.5504 | 0.7027 | 0.9963 | 343,480 | 262.58 |
| PushT | ConvMixer | 0.5286 | 0.6717 | 0.9900 | 336,279 | 93.90 |
| PushT | Mamba | 0.5448 | 0.6965 | 0.9961 | 362,136 | 111.59 |
| Wall | AdaLN(depth=1) | 0.8119 | 0.8804 | 0.9539 | 343,448 | 305.32 |
| Wall | ConvMixer | 0.7594 | 0.8520 | 0.9435 | 336,247 | 96.62 |
| Wall | Mamba | 0.7701 | 0.8553 | 0.9334 | 362,104 | 115.45 |
| PointMaze | AdaLN(depth=1) | 0.3631 | 0.6141 | 0.8959 | 343,480 | 306.67 |
| PointMaze | ConvMixer | 0.4193 | 0.6065 | 0.8777 | 336,279 | 94.17 |
| PointMaze | Mamba | 0.4760 | 0.6538 | 0.8884 | 362,136 | 111.67 |
| MW-R | AdaLN(depth=1) | 0.6734 | 0.8086 | 0.9386 | 344,680 | 297.46 |
| MW-R | ConvMixer | 0.5440 | 0.7234 | 0.5463 | 337,469 | 94.14 |
| MW-R | Mamba | 0.5027 | 0.6840 | 0.4801 | 363,176 | 114.37 |
| MW-RW | AdaLN(depth=1) | 0.7007 | 0.8415 | 0.9520 | 344,680 | 306.41 |
| MW-RW | ConvMixer | 0.5830 | 0.7548 | 0.6659 | 337,469 | 96.15 |
| MW-RW | Mamba | 0.5733 | 0.7517 | 0.7227 | 363,176 | 113.34 |

## Gate-4 Verdict

Primary AdaLN(depth=1) passes Gate 4 on all five tasks under one-seed breadth
evidence:

- `skill[h] > 0` for H=1..4.
- `change_skill[h] > 0` for H=1..4.
- `proprio_skill[h] > 0` for H=1..4.
- No H=4 collapse is observed.

Wall and PointMaze moved-patch checks are positive:

- Wall: AdaLN moved-patch skill at H=4 is `0.8804`.
- PointMaze: AdaLN moved-patch skill at H=4 is `0.6141`.

PointMaze is AdaLN's soft spot: raw visual skill peaks at H=2 and decays by H=6,
but H=4 remains positive and moved-patch skill is robustly positive, so this is a
softening profile rather than a collapse.

Mamba remains an ablation, but it should be flagged for an approval-gated,
Maze-specific 3-seed recheck. The reason is stronger than a single H=4 point:
on PointMaze, Mamba beats AdaLN at every diagnostic horizon H=1..6 on both raw
visual skill and moved-patch skill, while running at substantially lower step
time. At H=4:

- Mamba: `0.6538`
- AdaLN(depth=1): `0.6141`
- Absolute difference: `+0.0397`

This does not establish a Mamba win. The breadth scan has only one seed, and no
3-seed confirmation is authorized by this result.

## Interpretation

- AdaLN(depth=1) is the strongest breadth candidate and remains the Stage-2
  mainline.
- AdaLN's MetaWorld proprio separation is large: H=4 proprio skill is `0.9386`
  on MW-R and `0.9520` on MW-RW, compared with ConvMixer/Mamba ranges of roughly
  `0.48..0.72`. This is the most robust backbone separation in the scan.
- Mamba is competitive on PushT and leads the one-seed PointMaze readout while
  retaining much lower step time than AdaLN.
- ConvMixer is the cheapest control and is useful as a latency floor, but it
  does not displace AdaLN as the mainline candidate.
- Close one-seed margins are not ranking claims. In particular, PushT
  AdaLN-vs-Mamba H=4 moved-patch skill differs by only `0.0062`, and MW-RW
  ConvMixer-vs-Mamba moved-patch skill differs by only `0.0032`.
- Lance remains a validated data-IO backend only. It is not an algorithmic
  contribution and does not prove planning success equivalence.

## Stop Condition

Gate-4 is closed at this verdict. Do not launch CEM, Terver planning eval, H=8,
Stage-3, 3-seed confirmation, or the flagged PointMaze Mamba recheck without an
approved next-stage plan and explicit user `go`.
