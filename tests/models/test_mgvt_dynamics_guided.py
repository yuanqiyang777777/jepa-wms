import pytest
import torch

from app.plan_common.models.AdaLN_vit import vit_predictor_AdaLN
from app.plan_common.models.mgvt_dynamics_guided import (
    D1RTrendPredictor,
    DynamicsGuidedPredictor,
    TemporalMambaOrGRU,
    vit_predictor_mgvt_d1r,
    vit_predictor_mgvt_d_gru,
    vit_predictor_mgvt_d_mamba,
    vit_predictor_mgvt_d_mlp,
    vit_predictor_mgvt_d_raw_action,
    vit_predictor_mgvt_d_sparse_control,
)
from app.vjepa_wm.utils import init_video_model


PREDICTOR_KWARGS = {
    "img_size": (224, 224),
    "patch_size": 14,
    "num_frames": 3,
    "tubelet_size": 1,
    "embed_dim": 384,
    "predictor_embed_dim": 64,
    "depth": 1,
    "num_heads": 4,
    "action_dim": 80,
    "proprio_dim": 7,
    "use_proprio": True,
    "proprio_encoding": "feature",
    "proprio_emb_dim": 16,
    "proprio_tokens": 0,
    "proprio_encoder_inpred": False,
    "action_encoder_inpred": False,
    "d_h_dim": 16,
    "state_dim": 32,
    "context_window": 2,
    "require_cuda_mamba": False,
}


@pytest.mark.parametrize(
    "factory",
    [
        vit_predictor_mgvt_d_raw_action,
        vit_predictor_mgvt_d_mlp,
        vit_predictor_mgvt_d_gru,
        vit_predictor_mgvt_d_mamba,
        vit_predictor_mgvt_d_sparse_control,
    ],
)
def test_d1_predictors_return_stage2_contract(factory):
    predictor = factory(**PREDICTOR_KWARGS)
    x = torch.randn(2, 3, 1, 16, 16, 384)
    actions = torch.randn(2, 3, 1, predictor.predictor_total_embed_dim)
    proprio = torch.randn(2, 3, 16 * 16, 16)

    pred, action_features, pred_proprio = predictor(x, actions, proprio)

    assert pred.shape == (2, 3, 16 * 16, 384)
    assert action_features is None
    assert pred_proprio.shape == (2, 3, 16 * 16, 16)
    assert torch.isfinite(pred).all()
    assert torch.isfinite(pred_proprio).all()
    assert "mgvt_d/d_h_std" in predictor.last_aux_stats


def test_d1_predictor_backpropagates_through_trend_path():
    predictor = vit_predictor_mgvt_d_mlp(**PREDICTOR_KWARGS)
    x = torch.randn(2, 3, 1, 16, 16, 384)
    actions = torch.randn(2, 3, 1, predictor.predictor_total_embed_dim, requires_grad=True)
    proprio = torch.randn(2, 3, 16 * 16, 16)

    pred = predictor(x, actions, proprio)[0]
    loss = pred.pow(2).mean()
    loss.backward()

    grad_norm = sum(
        p.grad.detach().abs().sum().item()
        for name, p in predictor.named_parameters()
        if "f_dyn" in name and p.grad is not None
    )
    assert grad_norm > 0.0
    assert actions.grad is not None


@pytest.mark.parametrize(
    "factory",
    [
        vit_predictor_mgvt_d_raw_action,
        vit_predictor_mgvt_d_mlp,
        vit_predictor_mgvt_d_gru,
        vit_predictor_mgvt_d_mamba,
    ],
)
def test_d1_default_training_loss_touches_all_trainable_params(factory):
    predictor = factory(**PREDICTOR_KWARGS)
    x = torch.randn(2, 3, 1, 16, 16, 384)
    actions = torch.randn(2, 3, 1, predictor.predictor_total_embed_dim)
    proprio = torch.randn(2, 3, 16 * 16, 16)

    pred, _action_features, pred_proprio = predictor(x, actions, proprio)
    loss = pred.square().mean() + pred_proprio.square().mean()
    loss.backward()

    missing = [name for name, param in predictor.named_parameters() if param.requires_grad and param.grad is None]
    assert missing == []


def test_raw_action_guidance_does_not_create_unused_compact_state_modules():
    predictor = vit_predictor_mgvt_d_raw_action(**PREDICTOR_KWARGS)

    assert predictor.uses_compact_state is False
    assert predictor.visual_state is None
    assert predictor.proprio_state is None
    assert predictor.state_fuse is None


def test_d1_delta_p_head_is_explicit_not_default():
    predictor = vit_predictor_mgvt_d_mlp(**PREDICTOR_KWARGS)
    assert predictor.delta_p_head is None

    with_delta = vit_predictor_mgvt_d_mlp(**{**PREDICTOR_KWARGS, "delta_p_dim": 7})
    assert with_delta.delta_p_head is not None


@pytest.mark.parametrize("dh_ablation", ["remove", "random"])
def test_dh_ablation_keeps_fdyn_graph_for_ddp(dh_ablation):
    predictor = vit_predictor_mgvt_d_mlp(**{**PREDICTOR_KWARGS, "dh_ablation": dh_ablation})
    x = torch.randn(2, 3, 1, 16, 16, 384)
    actions = torch.randn(2, 3, 1, predictor.predictor_total_embed_dim)
    proprio = torch.randn(2, 3, 16 * 16, 16)

    pred = predictor(x, actions, proprio)[0]
    pred.square().mean().backward()

    missing = [
        name
        for name, param in predictor.named_parameters()
        if name.startswith("f_dyn") and param.requires_grad and param.grad is None
    ]
    assert missing == []


def test_remove_dh_ablation_changes_prediction():
    base = vit_predictor_mgvt_d_mlp(**PREDICTOR_KWARGS)
    remove = vit_predictor_mgvt_d_mlp(**{**PREDICTOR_KWARGS, "dh_ablation": "remove"})
    remove.load_state_dict(base.state_dict(), strict=True)
    x = torch.randn(2, 3, 1, 16, 16, 384)
    actions = torch.randn(2, 3, 1, base.predictor_total_embed_dim)
    proprio = torch.randn(2, 3, 16 * 16, 16)

    out_base = base(x, actions, proprio)[0]
    out_remove = remove(x, actions, proprio)[0]

    assert not torch.allclose(out_base, out_remove)


def test_random_dh_ablation_is_deterministic_for_rescoring():
    predictor = vit_predictor_mgvt_d_mlp(**{**PREDICTOR_KWARGS, "dh_ablation": "random"})
    x = torch.randn(2, 3, 1, 16, 16, 384)
    actions = torch.randn(2, 3, 1, predictor.predictor_total_embed_dim)
    proprio = torch.randn(2, 3, 16 * 16, 16)

    out_a = predictor(x, actions, proprio)[0]
    out_b = predictor(x, actions, proprio)[0]

    assert torch.allclose(out_a, out_b)


def test_mamba_cuda_requirement_blocks_fallback():
    mixer = TemporalMambaOrGRU(dim=16, require_cuda_mamba=True)
    if mixer.backend != "mamba_ssm":
        with pytest.raises(RuntimeError):
            mixer(torch.randn(2, 3, 16))
    else:
        with pytest.raises(RuntimeError):
            mixer(torch.randn(2, 3, 16))


@pytest.mark.parametrize(
    "pred_type",
    ["mgvt_d_raw_action", "mgvt_d_mlp", "mgvt_d_gru", "mgvt_d_mamba", "mgvt_d_sparse_control"],
)
def test_init_video_model_builds_d1_pred_types(pred_type):
    predictor, _encoder, action_encoder, proprio_encoder = init_video_model(
        device=torch.device("cpu"),
        enc_type="dino",
        enc_version="dinov2_vits14",
        img_size=224,
        embed_dim=384,
        pred_embed_dim=64,
        pred_depth=1,
        pred_num_heads=4,
        pred_type=pred_type,
        num_frames_pred=3,
        tubelet_size=1,
        action_dim=7,
        action_conditioning="token",
        action_tokens=1,
        action_encoder_inpred=False,
        proprio_dim=7,
        use_proprio=True,
        proprio_tokens=0,
        proprio_emb_dim=16,
        proprio_encoder_inpred=False,
        init_scale_factor_adaln=0,
        use_rope=False,
        cfgs_attn_pattern={"local_window_time": -1, "local_window_h": -1, "local_window_w": -1},
        require_cuda_mamba=False,
        d_h_dim=16,
        state_dim=32,
    )

    assert isinstance(predictor, DynamicsGuidedPredictor)
    assert action_encoder is not None
    assert proprio_encoder is not None
    assert predictor.predictor_total_embed_dim == 80


def test_init_video_model_forwards_d1_specific_kwargs():
    predictor, _encoder, _action_encoder, _proprio_encoder = init_video_model(
        device=torch.device("cpu"),
        enc_type="dino",
        enc_version="dinov2_vits14",
        img_size=224,
        embed_dim=384,
        pred_embed_dim=64,
        pred_depth=1,
        pred_num_heads=4,
        pred_type="mgvt_d_mamba",
        num_frames_pred=3,
        tubelet_size=1,
        action_dim=7,
        action_conditioning="token",
        action_tokens=1,
        action_encoder_inpred=False,
        proprio_dim=7,
        use_proprio=True,
        proprio_tokens=0,
        proprio_emb_dim=16,
        proprio_encoder_inpred=False,
        init_scale_factor_adaln=0,
        use_rope=False,
        cfgs_attn_pattern={"local_window_time": -1, "local_window_h": -1, "local_window_w": -1},
        d_h_dim=8,
        state_dim=48,
        context_window=1,
        proprio_flow="refine_only",
        dh_ablation="remove",
        refiner_depth=0,
        require_cuda_mamba=True,
        fdyn_type="mlp",
        guidance_mode="trend",
    )

    assert predictor.d_h_dim == 8
    assert predictor.proprio_flow == "refine_only"
    assert predictor.dh_ablation == "remove"
    assert predictor.context_window == 1
    assert predictor.visual_state[1].out_features == 48
    assert len(predictor.refiner.blocks) == 0
    assert predictor.fdyn_type == "mamba"
    assert predictor.guidance_mode == "trend"
    assert predictor.temporal_mixer.require_cuda_mamba is True


D1R_KWARGS = {
    **PREDICTOR_KWARGS,
    "use_proprio": False,
    "proprio_emb_dim": 0,
    "proprio_tokens": 0,
    "d_h_dim": 64,
    "r_h_dim": 64,
    "state_dim": 64,
    "p_dyn_dim": 64,
    "proprio_flow": "no_proprio",
}


def _d1r_batch():
    x = torch.randn(2, 3, 1, 16, 16, 384)
    future = x + 0.1 * torch.randn_like(x)
    actions = torch.randn(2, 3, 1, 64)
    return x, future, actions


def test_d1r_r1_teacher_stage_replaces_base_loss_and_freezes_deployable_path():
    predictor = vit_predictor_mgvt_d1r(**{**D1R_KWARGS, "d1r_stage": "r1_teacher"})
    x, future, actions = _d1r_batch()

    pred, _action_features, pred_proprio = predictor(x, actions, None, future_video_features=future)
    aux, replace = predictor.get_d1r_aux_losses()
    aux["d1r/loss_total"].backward()

    assert isinstance(predictor, D1RTrendPredictor)
    assert pred.shape == (2, 3, 16 * 16, 384)
    assert pred_proprio is None
    assert replace is True
    assert {"d1r/loss_delta", "d1r/loss_inv_r", "d1r/loss_sigreg", "d1r/loss_total"}.issubset(aux)
    assert any(p.requires_grad for p in predictor.p_dyn.parameters())
    assert not any(p.requires_grad for p in predictor.f_dyn.parameters())


def test_d1r_future_blind_prediction_sentinel():
    predictor = vit_predictor_mgvt_d1r(**{**D1R_KWARGS, "d1r_stage": "g_refine"})
    x, future, actions = _d1r_batch()

    pred_a = predictor(x, actions, None, future_video_features=future)[0]
    pred_b = predictor(x, actions, None, future_video_features=future + 1000.0)[0]

    assert torch.allclose(pred_a, pred_b)


def test_d1r_oracle_predict_is_explicitly_target_conditioned():
    predictor = vit_predictor_mgvt_d1r(**{**D1R_KWARGS, "d1r_stage": "oracle"})
    x, future, _actions = _d1r_batch()

    oracle_a = predictor.oracle_predict(x[:, :2], future[:, 2:3])
    oracle_b = predictor.oracle_predict(x[:, :2], future[:, 2:3] + 10.0)

    assert oracle_a.shape == (2, 1, 16 * 16, 384)
    assert not torch.allclose(oracle_a, oracle_b)


def test_d1r_r2_student_stage_uses_trend_loss_and_optional_inverse_loss():
    predictor = vit_predictor_mgvt_d1r(**{**D1R_KWARGS, "d1r_stage": "r2_student", "use_inv_d": True})
    x, future, actions = _d1r_batch()

    predictor(x, actions, None, future_video_features=future)
    aux, replace = predictor.get_d1r_aux_losses()
    aux["d1r/loss_total"].backward()

    assert replace is True
    assert aux["d1r/loss_trend"].item() >= 0.0
    assert aux["d1r/loss_inv_d"].item() >= 0.0
    assert not any(p.requires_grad for p in predictor.p_dyn.parameters())
    assert any(p.requires_grad for p in predictor.f_dyn.parameters())


@pytest.mark.parametrize(
    ("stage", "use_inv_d", "uses_aux_replace_loss"),
    [
        ("implicit", False, False),
        ("r1_teacher", False, True),
        ("r2_student", False, True),
        ("r2_student", True, True),
        ("g_refine", False, False),
        ("oracle", False, True),
    ],
)
def test_d1r_stage_trainable_params_all_receive_grad(stage, use_inv_d, uses_aux_replace_loss):
    predictor = vit_predictor_mgvt_d1r(
        **{**D1R_KWARGS, "d1r_stage": stage, "use_inv_d": use_inv_d}
    )
    x, future, actions = _d1r_batch()

    pred = predictor(x, actions, None, future_video_features=future)[0]
    aux, replace = predictor.get_d1r_aux_losses()
    loss = aux["d1r/loss_total"] if uses_aux_replace_loss else pred.square().mean()
    loss.backward()

    assert replace is uses_aux_replace_loss
    missing = [
        name
        for name, param in predictor.named_parameters()
        if param.requires_grad and param.grad is None
    ]
    assert missing == []


def test_d1r_inference_path_param_count_excludes_training_only_heads():
    predictor = vit_predictor_mgvt_d1r(**{**D1R_KWARGS, "d1r_stage": "g_refine"})
    inference_params = predictor.estimate_inference_path_params()
    total_params = sum(p.numel() for p in predictor.parameters())

    assert 0 < inference_params < total_params


def test_d1r_c1_adaln_inference_params_are_matched_within_five_percent():
    target = vit_predictor_mgvt_d1r(
        **{**D1R_KWARGS, "predictor_embed_dim": 128, "d1r_stage": "g_refine"}
    ).estimate_inference_path_params()
    c1_adaln = vit_predictor_AdaLN(
        img_size=PREDICTOR_KWARGS["img_size"],
        patch_size=PREDICTOR_KWARGS["patch_size"],
        num_frames=PREDICTOR_KWARGS["num_frames"],
        tubelet_size=PREDICTOR_KWARGS["tubelet_size"],
        embed_dim=PREDICTOR_KWARGS["embed_dim"],
        predictor_embed_dim=112,
        depth=2,
        num_heads=PREDICTOR_KWARGS["num_heads"],
        action_dim=PREDICTOR_KWARGS["action_dim"],
        proprio_dim=PREDICTOR_KWARGS["proprio_dim"],
        use_proprio=False,
        proprio_emb_dim=0,
        proprio_tokens=0,
        proprio_encoder_inpred=False,
        action_encoder_inpred=False,
        init_scale_factor_adaln=0,
    )

    assert abs(c1_adaln.estimate_inference_path_params() - target) / target <= 0.05


def test_init_video_model_builds_d1r_pred_type():
    predictor, _encoder, action_encoder, proprio_encoder = init_video_model(
        device=torch.device("cpu"),
        enc_type="dino",
        enc_version="dinov2_vits14",
        img_size=224,
        embed_dim=384,
        pred_embed_dim=64,
        pred_depth=1,
        pred_num_heads=4,
        pred_type="mgvt_d1r",
        num_frames_pred=3,
        tubelet_size=1,
        action_dim=7,
        action_conditioning="token",
        action_tokens=1,
        action_encoder_inpred=False,
        proprio_dim=7,
        use_proprio=False,
        proprio_tokens=0,
        proprio_emb_dim=0,
        proprio_encoder_inpred=False,
        init_scale_factor_adaln=0,
        use_rope=False,
        cfgs_attn_pattern={"local_window_time": -1, "local_window_h": -1, "local_window_w": -1},
        d_h_dim=64,
        r_h_dim=64,
        state_dim=64,
        p_dyn_dim=64,
        d1r_stage="oracle",
    )

    assert isinstance(predictor, D1RTrendPredictor)
    assert predictor.d1r_stage == "oracle"
    assert action_encoder is not None
    assert proprio_encoder is None
