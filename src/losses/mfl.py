# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""Motion-Focal Loss (MFL).

Patch-wise motion-weighted reweighting of the latent prediction loss for JEPA
world models. See notes/memory_seeds/mfl_method_design.md and
method/MFL/mfl_project_full_proposal_v3.pdf for the design.
"""

import torch.nn.functional as F


def motion_focal_weight(target, prev_target, eta, gamma, eps=1e-6):
    """Per-patch motion-focal weight for the latent prediction loss.

    Args:
        target:      [B, T, N, C] target-side latent z_t.
        prev_target: [B, T, N, C] target-side latent z_{t-1}, frame-aligned with target.
        eta:         mix in [0, 1]; eta=0 yields uniform weights (caller should skip).
        gamma:       focal sharpness > 0.
        eps:         numerical floor for the two normalizations.

    Returns:
        w: [B, T, N] non-negative weight; per-frame mean over patches is ~1.
    """
    # Weights depend only on target-side latents — no feedback from prediction error.
    target = target.detach().float()
    prev_target = prev_target.detach().float()

    # Per-patch motion score delta = 1 - cos(z_t, z_{t-1}), in [0, 2].
    delta = 1.0 - F.cosine_similarity(target, prev_target, dim=-1, eps=1e-8)

    # Per-frame normalization over patches (dim=-1), not per-batch.
    r = delta / (delta.mean(dim=-1, keepdim=True) + eps)

    # Focal weight: eta=0 -> uniform 1.0, eta=1 -> pure r**gamma.
    u = (1.0 - eta) + eta * r.clamp_min(0.0).pow(gamma)

    # Sum-normalization: per-frame mean(w) ~= 1 so loss scale stays baseline-comparable.
    n_patches = u.shape[-1]
    w = n_patches * u / (u.sum(dim=-1, keepdim=True) + eps)
    return w
