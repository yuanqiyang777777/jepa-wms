import pytest
import torch

from app.plan_common.models.mgvt_mixers import (
    MGVTBlock,
    vit_predictor_mgvt_convmixer,
    vit_predictor_mgvt_mamba,
    vit_predictor_mgvt_mlp,
)
from app.plan_common.models.AdaLN_vit import FWAdaLNBlock, vit_predictor_AdaLN
from app.vjepa_wm.utils import init_video_model


PREDICTOR_KWARGS = {
    "img_size": (224, 224),
    "patch_size": 14,
    "num_frames": 3,
    "tubelet_size": 1,
    "embed_dim": 384,
    "predictor_embed_dim": 96,
    "depth": 1,
    "num_heads": 4,
    "action_dim": 96,
    "proprio_dim": None,
    "use_proprio": False,
    "proprio_encoding": "feature",
    "proprio_emb_dim": 0,
    "proprio_tokens": 0,
    "proprio_encoder_inpred": False,
    "action_encoder_inpred": False,
    "init_scale_factor_adaln": 0,
    "use_rope": False,
    "local_window": (-1, -1, -1),
}


@pytest.mark.parametrize(
    "factory",
    [
        vit_predictor_mgvt_mlp,
        vit_predictor_mgvt_convmixer,
        vit_predictor_mgvt_mamba,
        vit_predictor_AdaLN,
    ],
)
def test_stage1_predictors_return_h1_latents(factory):
    predictor = factory(**PREDICTOR_KWARGS)
    x = torch.randn(2, 3, 1, 16, 16, 384)
    actions = torch.randn(2, 3, 1, predictor.predictor_total_embed_dim)

    pred, action_features, proprio_features = predictor(x, actions, None)

    assert pred.shape == (2, 3, 16 * 16, 384)
    assert action_features is None
    assert proprio_features is None
    assert torch.isfinite(pred).all()


@pytest.mark.parametrize(
    "block_cls, kwargs",
    [
        (
            FWAdaLNBlock,
            {
                "num_heads": 4,
                "use_rope": False,
            },
        ),
        (
            MGVTBlock,
            {
                "mixer_type": "mlp",
            },
        ),
        (
            MGVTBlock,
            {
                "mixer_type": "convmixer",
            },
        ),
        (
            MGVTBlock,
            {
                "mixer_type": "mamba",
            },
        ),
    ],
)
def test_adaln_zero_blocks_are_identity_at_init(block_cls, kwargs):
    block = block_cls(dim=64, grid_size=16, mlp_ratio=2.0, **kwargs)
    torch.nn.init.zeros_(block.adaLN_modulation[1].weight)
    torch.nn.init.zeros_(block.adaLN_modulation[1].bias)
    x = torch.randn(2, 3 * 16 * 16, 64)
    z = torch.randn(2, 3, 64)

    out = block(x, z, T=3, H_patches=16, W_patches=16)

    torch.testing.assert_close(out, x, rtol=1e-5, atol=1e-5)


def test_stage1_backbone_param_counts_are_matched():
    # Widths are matched for the W=2 ContextStacker
    # (predictor_embed input = 2 * embed_dim) on the local GRU-fallback Mamba path.
    matched_specs = [
        (vit_predictor_mgvt_mlp, 106),
        (vit_predictor_mgvt_convmixer, 103),
        (vit_predictor_mgvt_mamba, 88),
        (vit_predictor_AdaLN, 104),
    ]
    counts = []
    for factory, width in matched_specs:
        kwargs = dict(PREDICTOR_KWARGS)
        kwargs["predictor_embed_dim"] = width
        kwargs["action_dim"] = width
        counts.append(sum(p.numel() for p in factory(**kwargs).parameters()))
    mean_count = sum(counts) / len(counts)

    assert all(abs(count - mean_count) / mean_count <= 0.10 for count in counts), counts


@pytest.mark.parametrize("pred_type", ["mgvt_mlp", "mgvt_convmixer", "mgvt_mamba"])
def test_init_video_model_builds_mgvt_pred_types(pred_type):
    predictor, encoder, action_encoder, proprio_encoder = init_video_model(
        device=torch.device("cpu"),
        enc_type="dino",
        enc_version="dinov2_vits14",
        img_size=224,
        embed_dim=384,
        pred_embed_dim=96,
        pred_depth=1,
        pred_num_heads=4,
        pred_type=pred_type,
        num_frames_pred=3,
        tubelet_size=1,
        action_dim=7,
        action_conditioning="token",
        action_tokens=1,
        action_encoder_inpred=False,
        proprio_dim=None,
        use_proprio=False,
        proprio_tokens=0,
        proprio_emb_dim=0,
        proprio_encoder_inpred=False,
        init_scale_factor_adaln=0,
        use_rope=False,
        cfgs_attn_pattern={"local_window_time": -1, "local_window_h": -1, "local_window_w": -1},
    )

    assert predictor is not None
    assert action_encoder is not None
    assert proprio_encoder is None


@pytest.mark.parametrize(
    "factory",
    [vit_predictor_mgvt_mlp, vit_predictor_mgvt_convmixer, vit_predictor_mgvt_mamba],
)
def test_stage1_predictor_can_overfit_tiny_latent_batch(factory):
    predictor = factory(**PREDICTOR_KWARGS)
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=1e-2)
    x = torch.randn(2, 3, 1, 16, 16, 384)
    actions = torch.randn(2, 3, 1, predictor.predictor_total_embed_dim)
    target = torch.randn(2, 3, 16 * 16, 384)

    with torch.no_grad():
        initial = torch.nn.functional.mse_loss(predictor(x, actions, None)[0], target).item()

    for _ in range(50):
        optimizer.zero_grad(set_to_none=True)
        pred = predictor(x, actions, None)[0]
        loss = torch.nn.functional.mse_loss(pred, target)
        loss.backward()
        optimizer.step()

    final = torch.nn.functional.mse_loss(predictor(x, actions, None)[0], target).item()
    assert final < initial


def test_context_window_conditions_on_previous_frame():
    x = torch.randn(2, 3, 1, 16, 16, 384)
    x_perturbed = x.clone()
    x_perturbed[:, 0] += 5.0

    pred_w1 = vit_predictor_mgvt_mlp(**{**PREDICTOR_KWARGS, "context_window": 1})
    actions_w1 = torch.randn(2, 3, 1, pred_w1.predictor_total_embed_dim)
    out_w1 = pred_w1(x, actions_w1, None)[0][:, 1]
    out_w1_perturbed = pred_w1(x_perturbed, actions_w1, None)[0][:, 1]
    torch.testing.assert_close(out_w1, out_w1_perturbed, rtol=0.0, atol=1e-6)

    pred_w2 = vit_predictor_mgvt_mlp(**{**PREDICTOR_KWARGS, "context_window": 2})
    actions_w2 = torch.randn(2, 3, 1, pred_w2.predictor_total_embed_dim)
    out_w2 = pred_w2(x, actions_w2, None)[0][:, 1]
    out_w2_perturbed = pred_w2(x_perturbed, actions_w2, None)[0][:, 1]
    assert not torch.allclose(out_w2, out_w2_perturbed, atol=1e-6)
