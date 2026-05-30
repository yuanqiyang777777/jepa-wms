# Experiment Registry

| Date | Experiment | Protocol | Result | Provenance |
| --- | --- | --- | --- | --- |
| 2026-05-30 | MGVT-JEPA Stage-1 Backbone Gate | H=1 frozen DINOv2-S/14 latent prediction; W=2; Lance IO; 4 backbones x 2 tasks x 3 seeds | Gate 1 PASS on PushT and MW-RW. Gate 2 FAIL for Mamba. Prefer AdaLN(depth=1); keep ConvMixer as lightweight control; demote Mamba to ablation. | `experiments/results/mgvt_stage1/MGVT_STAGE1_GATE_20260530.md`; raw artifacts under `experiments/results/mgvt_stage1/real_mamba_20260530/`; remote logs under `$JEPAWM_LOGS/mgvt_stage1_3seed_mamba_ssm/` |
