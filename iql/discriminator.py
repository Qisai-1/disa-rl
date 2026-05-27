"""
Density-ratio discriminator d_psi(s, a) -> P(real | s, a).

Trained on the fly alongside IQL on (real, synthetic) batches with BCE loss.
Used by SA-IQL to importance-weight Q-loss on syn transitions:

    w(s,a) = clip( d / (1 - d), [w_min, w_max] )

This is a principled correction for distribution shift under the standard
importance-sampling identity:

    E_real[L(s,a)] = E_syn[(p_real / p_syn) L(s,a)]

We learn p_real / p_syn = d / (1 - d) via a probabilistic classifier
(Sugiyama et al. 2012, Goodfellow et al. 2014).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from iql.networks import build_mlp, orthogonal_init


class DensityRatioDiscriminator(nn.Module):
    """
    Small MLP classifier on (obs, action) returning a logit.
    The implied density ratio is `sigmoid(logit) / (1 - sigmoid(logit))`,
    which we equivalently compute as `exp(logit)` (more numerically stable).
    """

    def __init__(self, obs_dim: int, action_dim: int,
                 hidden_dims=(256, 256)):
        super().__init__()
        self.net = build_mlp(obs_dim + action_dim, 1, hidden_dims, use_ln=True)
        orthogonal_init(self.net)

    def forward(self, obs: Tensor, action: Tensor) -> Tensor:
        """Returns logit (B,) — apply sigmoid for probability of real."""
        x = torch.cat([obs, action], dim=-1)
        return self.net(x).squeeze(-1)

    @torch.no_grad()
    def density_ratio(
        self,
        obs:    Tensor,
        action: Tensor,
        clip:   tuple[float, float] = (0.5, 2.0),
    ) -> Tensor:
        """
        Compute clipped density ratio w(s,a) = p_real / p_syn.

        Stable via logit form: w = exp(logit) (mathematically equivalent to
        d / (1-d) for d = sigmoid(logit)).
        """
        logit = self.forward(obs, action)
        # clamp the logit before exponentiating to avoid overflow
        log_w = logit.clamp(min=torch.log(torch.tensor(clip[0], device=obs.device)),
                            max=torch.log(torch.tensor(clip[1], device=obs.device)))
        return log_w.exp()

    def bce_loss(
        self,
        real_obs:    Tensor,
        real_action: Tensor,
        syn_obs:     Tensor,
        syn_action:  Tensor,
    ) -> Tensor:
        """
        Binary cross-entropy on real (label=1) vs syn (label=0).
        Returns scalar loss to back-prop.
        """
        l_real = self.forward(real_obs, real_action)
        l_syn  = self.forward(syn_obs,  syn_action)
        targets_real = torch.ones_like(l_real)
        targets_syn  = torch.zeros_like(l_syn)
        loss = (
            F.binary_cross_entropy_with_logits(l_real, targets_real)
            + F.binary_cross_entropy_with_logits(l_syn,  targets_syn)
        )
        return loss
