from types import SimpleNamespace

import pytest
import torch

from app.vjepa_wm.video_wm import VideoWM
from src.losses.mfl import motion_focal_weight


def _loss_cfg(**overrides):
    cfg = {
        "cos_loss_weight": 0.0,
        "l1_loss_weight": 0.0,
        "l2_loss_weight": 1.0,
        "smooth_l1_loss_weight": 0.0,
    }
    cfg.update(overrides)
    return cfg


def _fake_wm(cfgs_loss, grid_size=4, use_proprio=False):
    return SimpleNamespace(
        cfgs_loss=cfgs_loss,
        grid_size=grid_size,
        proprio_loss=use_proprio,
        use_proprio=use_proprio,
    )


def _video(batch=2, time=3, grid_size=4, dim=8, requires_grad=False):
    return torch.randn(batch, time, 1, grid_size, grid_size, dim, requires_grad=requires_grad)


def test_motion_focal_weight_is_frame_normalized_non_negative_and_detached():
    target = torch.randn(2, 4, 16, 8, requires_grad=True)
    prev_target = torch.randn(2, 4, 16, 8, requires_grad=True)

    weight = motion_focal_weight(target, prev_target, eta=0.5, gamma=1.0)

    assert torch.allclose(weight.mean(dim=-1), torch.ones(2, 4), atol=1e-6)
    assert bool((weight >= 0).all())
    assert not weight.requires_grad
    assert weight.grad_fn is None


def test_eta_zero_matches_missing_mfl_key_for_teacher_forcing_and_rollout():
    torch.manual_seed(0)
    pred_tf = _video(time=4)
    target_tf = _video(time=4)
    pred_rollout = _video(time=1)
    target_rollout = _video(time=1)

    base_tf = VideoWM.compute_loss(
        _fake_wm(_loss_cfg()), pred_tf, None, target_tf, None, shift=1
    )["loss"]
    eta0_tf = VideoWM.compute_loss(
        _fake_wm(_loss_cfg(mfl_eta=0.0)), pred_tf, None, target_tf, None, shift=1
    )["loss"]
    base_rollout = VideoWM.compute_loss(
        _fake_wm(_loss_cfg()), pred_rollout, None, target_rollout, None, shift=0
    )["loss"]
    eta0_rollout = VideoWM.compute_loss(
        _fake_wm(_loss_cfg(mfl_eta=0.0)), pred_rollout, None, target_rollout, None, shift=0
    )["loss"]

    assert torch.equal(base_tf, eta0_tf)
    assert torch.equal(base_rollout, eta0_rollout)


def test_mfl_loss_backpropagates_to_prediction_not_target():
    torch.manual_seed(0)
    pred = _video(requires_grad=True)
    target = _video(requires_grad=True)

    loss = VideoWM.compute_loss(
        _fake_wm(_loss_cfg(mfl_eta=0.5, mfl_gamma=1.0)),
        pred,
        None,
        target,
        None,
        shift=1,
    )["loss"]
    loss.backward()

    assert pred.grad is not None
    assert bool((pred.grad != 0).any())
    assert target.grad is None


def test_use_mfl_false_disables_mfl_assert_and_matches_baseline_loss():
    torch.manual_seed(0)
    pred = _video(time=1)
    target = _video(time=1)

    baseline = VideoWM.compute_loss(
        _fake_wm(_loss_cfg()), pred, None, target, None, shift=0
    )["loss"]
    eval_loss = VideoWM.compute_loss(
        _fake_wm(_loss_cfg(mfl_eta=0.5, mfl_gamma=1.0)),
        pred,
        None,
        target,
        None,
        shift=0,
        use_mfl=False,
    )["loss"]

    assert torch.equal(baseline, eval_loss)


def test_invalid_mfl_hyperparameters_fail_when_mfl_is_enabled():
    pred = _video()
    target = _video()

    with pytest.raises(ValueError, match="mfl_eta"):
        VideoWM.compute_loss(
            _fake_wm(_loss_cfg(mfl_eta=1.5, mfl_gamma=1.0)),
            pred,
            None,
            target,
            None,
            shift=1,
        )
    with pytest.raises(ValueError, match="mfl_gamma"):
        VideoWM.compute_loss(
            _fake_wm(_loss_cfg(mfl_eta=0.5, mfl_gamma=0.0)),
            pred,
            None,
            target,
            None,
            shift=1,
        )


def test_no_pred_rollout_uses_timestep_before_target_as_prev_video_features():
    torch.manual_seed(0)
    video_features = _video(batch=1, time=4, grid_size=2, dim=3)
    action_features = torch.randn(1, 4, 1, 2)
    captured = {}

    def forward_pred(vid_feats, act_feats, prop_feats, debug=False):
        del act_feats, prop_feats, debug
        return vid_feats, None, None

    def compute_loss(pred_video_features, pred_proprio_features, video_targets, prop_targets, **kwargs):
        del pred_video_features, pred_proprio_features, video_targets, prop_targets
        captured["prev_video_features"] = kwargs["prev_video_features"]
        return {"loss": torch.tensor(0.0)}

    wm = SimpleNamespace(
        use_proprio=False,
        forward_pred=forward_pred,
        compute_loss=compute_loss,
    )

    VideoWM.rollout(
        wm,
        video_features=video_features,
        action_features=action_features,
        pred_video_features=None,
        t=1,
        rollout_steps=1,
    )

    assert torch.equal(captured["prev_video_features"], video_features[:, 1:2])
