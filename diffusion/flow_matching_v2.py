"""
v2 Conditional Flow Matching — includes REWARD as a generated channel.

Differences vs flow_matching.py (v1):
  - Loss adds an L_reward term (per-channel MSE on the reward velocity).
  - The model output is (obs, action, reward); we slice the last channel
    for the reward loss.
  - Sampling is unchanged (uses the same Heun / Euler schemes).

Loss:
    L = λ_obs · ||v_obs − target_obs||²
      + λ_action · ||v_action − target_action||²
      + λ_reward · ||v_reward − target_reward||²
      + λ_temporal · ||Δobs_pred − Δobs_real||²

The reward dim is z-score normalized by data_v2.DataNormalizer BEFORE this
loss sees it, so equal weighting (λ_reward = 1.0, same as λ_obs/λ_action)
is appropriate — all three live in roughly N(0, 1) space.
"""

from __future__ import annotations
import sys, os as _os
_d = _os.path.dirname(_os.path.abspath(__file__))
if _d not in sys.path: sys.path.insert(0, _d)
if _os.path.dirname(_d) not in sys.path: sys.path.insert(0, _os.path.dirname(_d))

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Dict, Tuple

from model_v2 import TrajectoryDiTV2
from flow_matching import sample_logit_normal


class ConditionalFlowMatchingV2:
    """
    v2 — diffuses (obs, action, reward) jointly.

    Args:
        model        : TrajectoryDiTV2 (output dim D = obs_dim + action_dim + 1)
        lambda_obs   : weight on obs velocity loss
        lambda_action: weight on action velocity loss
        lambda_reward: weight on reward velocity loss (default 1.0, equal weight)
        lambda_temporal : weight on obs-delta consistency loss
        cfg_scale    : guidance scale at sampling time
    """
    def __init__(
        self,
        model:           TrajectoryDiTV2,
        device:          torch.device,
        lambda_obs:      float = 1.0,
        lambda_action:   float = 1.0,
        lambda_reward:   float = 1.0,
        lambda_temporal: float = 0.1,
        cfg_scale:       float = 1.5,
    ):
        self.model           = model
        self.device          = device
        self.lambda_obs      = lambda_obs
        self.lambda_action   = lambda_action
        self.lambda_reward   = lambda_reward
        self.lambda_temporal = lambda_temporal
        self.cfg_scale       = cfg_scale

    def get_train_tuple(self, x1: Tensor):
        x0    = torch.randn_like(x1)
        tau   = sample_logit_normal(x1.shape[0], x1.device)
        x_tau = tau[:, None, None] * x1 + (1.0 - tau[:, None, None]) * x0
        return x_tau, tau, x1 - x0, x0

    def loss(self, x1: Tensor, cond: Tensor) -> Tuple[Tensor, Dict[str, float]]:
        """
        x1   : (B, T, D)  clean (obs, action, reward) trajectory; D = obs+action+1
        cond : (B, cond_dim)
        Returns: scalar loss, metrics dict
        """
        x1   = x1.to(self.device)
        cond = cond.to(self.device)

        x_tau, tau, target, _ = self.get_train_tuple(x1)
        drop_mask = torch.rand(x1.shape[0], device=self.device) < self.model.cfg_dropout_prob
        velocity  = self.model(x_tau, tau, cond, drop_mask)

        obs_dim    = self.model.obs_dim
        action_dim = self.model.action_dim
        oa_end     = obs_dim + action_dim

        # Per-channel velocity losses
        L_obs    = F.mse_loss(velocity[..., :obs_dim],         target[..., :obs_dim])
        L_action = F.mse_loss(velocity[..., obs_dim:oa_end],   target[..., obs_dim:oa_end])
        L_reward = F.mse_loss(velocity[..., oa_end:oa_end+1],  target[..., oa_end:oa_end+1])

        # Temporal consistency on obs deltas (matches v1)
        x1_pred    = x_tau + (1.0 - tau[:, None, None]) * velocity
        delta_pred = x1_pred[:, 1:, :obs_dim] - x1_pred[:, :-1, :obs_dim]
        delta_real = x1[:, 1:, :obs_dim]      - x1[:, :-1, :obs_dim]
        L_temporal = F.mse_loss(delta_pred, delta_real)

        total = (self.lambda_obs      * L_obs
               + self.lambda_action   * L_action
               + self.lambda_reward   * L_reward
               + self.lambda_temporal * L_temporal)

        metrics = {
            "loss/total":    total.item(),
            "loss/obs":      L_obs.item(),
            "loss/action":   L_action.item(),
            "loss/reward":   L_reward.item(),
            "loss/temporal": L_temporal.item(),
        }
        return total, metrics

    # ------------------------------------------------------------------
    # Sampling — same as v1 but produces a wider tensor (D includes reward)
    # ------------------------------------------------------------------

    def _velocity(self, x, tau, cond, cfg_scale):
        if cfg_scale == 1.0:
            return self.model(x, tau, cond)
        B     = x.shape[0]
        x2    = torch.cat([x,    x],    dim=0)
        tau2  = torch.cat([tau,  tau],  dim=0)
        cond2 = torch.cat([cond, cond], dim=0)
        drop  = torch.cat([
            torch.zeros(B, dtype=torch.bool, device=x.device),
            torch.ones( B, dtype=torch.bool, device=x.device),
        ])
        v2 = self.model(x2, tau2, cond2, drop)
        v_cond, v_un = v2[:B], v2[B:]
        return v_un + cfg_scale * (v_cond - v_un)

    @torch.no_grad()
    def heun_sample(self, batch_size, cond, nfe=50, cfg_scale=None):
        """Heun 2nd-order ODE sampler. Default NFE 50 (v2; v1 used 20)."""
        w    = cfg_scale if cfg_scale is not None else self.cfg_scale
        cond = cond.to(self.device)
        x    = torch.randn((batch_size, self.model.T, self.model.D), device=self.device)
        dt   = 1.0 / nfe
        for i in range(nfe):
            t_curr = torch.full((batch_size,), i * dt,           device=self.device)
            t_next = torch.full((batch_size,), min((i+1)*dt,1.0), device=self.device)
            k1     = self._velocity(x,      t_curr, cond, w)
            x_pred = x + k1 * dt
            k2     = self._velocity(x_pred, t_next, cond, w)
            x      = x + (k1 + k2) * 0.5 * dt
        return x

    @torch.no_grad()
    def euler_sample(self, batch_size, cond, nfe=50, cfg_scale=None):
        w    = cfg_scale if cfg_scale is not None else self.cfg_scale
        cond = cond.to(self.device)
        x    = torch.randn((batch_size, self.model.T, self.model.D), device=self.device)
        dt   = 1.0 / nfe
        for i in range(nfe):
            tau = torch.full((batch_size,), i * dt, device=self.device)
            x   = x + self._velocity(x, tau, cond, w) * dt
        return x


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TrajectoryDiTV2(obs_dim=17, action_dim=6, trajectory_length=100,
                            hidden_size=128, depth=2, num_heads=2,   # small for quick test
                            mlp_dropout=0.0).to(device)
    cfm = ConditionalFlowMatchingV2(model, device)
    B, T, D = 4, 100, 24
    x1   = torch.randn(B, T, D, device=device)
    cond = torch.randn(B, 18, device=device)
    loss, metrics = cfm.loss(x1, cond)
    assert torch.isfinite(loss), "loss is NaN/Inf"
    print(f"v2 loss OK: {loss.item():.4f}  | metrics keys: {list(metrics.keys())}")

    # Sampling smoke test
    sample = cfm.heun_sample(2, cond[:2], nfe=4)
    assert sample.shape == (2, T, D), f"Got {sample.shape}"
    print(f"v2 Heun sampling OK: shape={sample.shape}")
