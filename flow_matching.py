"""
Conditional Flow Matching with:
  - Logit-normal time sampling (Esser et al., SD3)
  - Per-modality weighted loss (obs / action / reward)
  - Temporal consistency auxiliary loss
  - Heun ODE sampler with CFG double-batch trick

Loss structure
--------------
  L_total = λ_obs    · ||v_obs  − target_obs   ||²
          + λ_action · ||v_act  − target_action ||²
          + λ_reward · ||v_rew  − target_reward ||²
          + λ_temp   · ||Δobs_pred_{t+1,t} − Δobs_real_{t+1,t}||²

The temporal consistency term is computed on the *predicted clean trajectory*
x̂₁ = x_τ + (1 − τ) · v_pred  (Euler step from τ to 1).
This supervises the first-order difference structure of observations,
directly discouraging physically implausible discontinuities.
"""

from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Dict, Tuple

from model import TrajectoryDiT
from config import LossConfig


# ──────────────────────────────────────────────────────────────────────────────
# Time sampling
# ──────────────────────────────────────────────────────────────────────────────

def sample_logit_normal(
    batch_size: int,
    device:     torch.device,
    mean:       float = 0.0,
    std:        float = 1.0,
) -> Tensor:
    """
    Sample τ from a logit-normal distribution.

        u ~ N(mean, std)
        τ = sigmoid(u)  ∈ (0, 1)

    Concentrates density around τ = 0.5 (hardest denoising region).
    Empirically improves sample quality vs. uniform sampling (SD3, Flux).
    """
    return torch.sigmoid(torch.randn(batch_size, device=device) * std + mean)


# ──────────────────────────────────────────────────────────────────────────────
# Main class
# ──────────────────────────────────────────────────────────────────────────────

class ConditionalFlowMatching:
    """
    Wraps TrajectoryDiT with CFM training and Heun-based generation.

    Parameters
    ----------
    model      : TrajectoryDiT  (multi-modal output heads)
    device     : torch.device
    loss_cfg   : LossConfig     (per-modality weights + temporal weight)
    cfg_scale  : default guidance scale for inference
    """

    def __init__(
        self,
        model:     TrajectoryDiT,
        device:    torch.device,
        loss_cfg:  LossConfig    = None,
        cfg_scale: float         = 1.5,
    ):
        self.model     = model
        self.device    = device
        self.loss_cfg  = loss_cfg or LossConfig()
        self.cfg_scale = cfg_scale

    # ──────────────────────────────────────────────────────────────────────
    # Training tuple
    # ──────────────────────────────────────────────────────────────────────

    def get_train_tuple(
        self,
        x1: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Sample a flow matching training tuple.

        x1     : (B, T, D) clean trajectory (normalised)
        Returns: x_tau, tau, target_vel, x0
        """
        x0    = torch.randn_like(x1)
        tau   = sample_logit_normal(x1.shape[0], x1.device)    # (B,)
        x_tau = tau[:, None, None] * x1 + (1.0 - tau[:, None, None]) * x0
        return x_tau, tau, x1 - x0, x0   # target velocity = x1 − x0

    # ──────────────────────────────────────────────────────────────────────
    # Loss
    # ──────────────────────────────────────────────────────────────────────

    def loss(
        self,
        x1:   Tensor,    # (B, T, D) clean trajectory
        cond: Tensor,    # (B, cond_dim) condition
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        Compute the total weighted flow matching loss.

        Returns
        -------
        total_loss : scalar tensor  (for .backward())
        metrics    : dict of per-component loss values (for WandB logging)
        """
        x1   = x1.to(self.device)
        cond = cond.to(self.device)

        x_tau, tau, target, x0 = self.get_train_tuple(x1)

        # CFG dropout: replace condition with null embedding for some samples
        drop_mask = torch.rand(x1.shape[0], device=self.device) < self.model.cfg_dropout_prob

        velocity, v_obs, v_act, v_rew = self.model(x_tau, tau, cond, drop_mask)

        lc = self.loss_cfg
        obs_dim    = self.model.obs_dim
        action_dim = self.model.action_dim

        # ── Per-modality flow matching losses ──────────────────────────────
        tgt_obs = target[..., :obs_dim]
        tgt_act = target[..., obs_dim:obs_dim + action_dim]
        tgt_rew = target[..., -1:]

        L_obs    = F.mse_loss(v_obs, tgt_obs)
        L_action = F.mse_loss(v_act, tgt_act)
        L_reward = F.mse_loss(v_rew, tgt_rew)

        # ── Temporal consistency auxiliary loss ────────────────────────────
        # Reconstruct predicted clean trajectory from velocity prediction:
        #   x̂₁ = x_τ + (1 − τ) · v_pred
        x1_pred = x_tau + (1.0 - tau[:, None, None]) * velocity   # (B, T, D)

        # First-order differences of observations
        delta_pred = x1_pred[:, 1:, :obs_dim] - x1_pred[:, :-1, :obs_dim]   # (B, T-1, obs_dim)
        delta_real = x1[:, 1:, :obs_dim]       - x1[:, :-1, :obs_dim]

        L_temporal = F.mse_loss(delta_pred, delta_real)

        # ── Total loss ─────────────────────────────────────────────────────
        total = (
            lc.lambda_obs    * L_obs
          + lc.lambda_action * L_action
          + lc.lambda_reward * L_reward
          + lc.lambda_temporal * L_temporal
        )

        metrics = {
            "loss/total":    total.item(),
            "loss/obs":      L_obs.item(),
            "loss/action":   L_action.item(),
            "loss/reward":   L_reward.item(),
            "loss/temporal": L_temporal.item(),
        }
        return total, metrics

    # ──────────────────────────────────────────────────────────────────────
    # CFG velocity helper
    # ──────────────────────────────────────────────────────────────────────

    def _velocity(
        self,
        x:         Tensor,
        tau:       Tensor,
        cond:      Tensor,
        cfg_scale: float,
    ) -> Tensor:
        """
        Compute CFG-weighted velocity in a single forward pass.

        When cfg_scale == 1.0, only the conditional branch runs.
        Otherwise, both branches run in one double-batch forward pass:
            v_cfg = v_uncond + cfg_scale · (v_cond − v_uncond)
        """
        if cfg_scale == 1.0:
            v, _, _, _ = self.model(x, tau, cond)
            return v

        B = x.shape[0]
        x2    = torch.cat([x,    x],    dim=0)
        tau2  = torch.cat([tau,  tau],  dim=0)
        cond2 = torch.cat([cond, cond], dim=0)
        drop  = torch.cat([
            torch.zeros(B, dtype=torch.bool, device=x.device),
            torch.ones( B, dtype=torch.bool, device=x.device),
        ])
        v2, _, _, _ = self.model(x2, tau2, cond2, drop)
        v_cond, v_uncond = v2[:B], v2[B:]
        return v_uncond + cfg_scale * (v_cond - v_uncond)

    # ──────────────────────────────────────────────────────────────────────
    # Heun sampler
    # ──────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def heun_sample(
        self,
        batch_size: int,
        cond:       Tensor,
        nfe:        int            = 20,
        cfg_scale:  Optional[float] = None,
    ) -> Tensor:
        """
        2nd-order Heun ODE solver.

        Each step uses a predictor-corrector pair:
            k₁ = v(x_i,       t_i    )
            k₂ = v(x_i + k₁dt, t_{i+1})
            x_{i+1} = x_i + (k₁ + k₂)/2 · dt

        Achieves substantially better quality than Euler at the same NFE
        because it uses a corrected slope estimate.  Actual network
        evaluations = 2 · nfe.

        Returns: (B, T, D) normalised generated trajectories.
        """
        w    = cfg_scale if cfg_scale is not None else self.cfg_scale
        cond = cond.to(self.device)
        x    = torch.randn((batch_size, self.model.T, self.model.D), device=self.device)
        dt   = 1.0 / nfe

        for i in range(nfe):
            t_curr = torch.full((batch_size,), i * dt,          device=self.device)
            t_next = torch.full((batch_size,), min((i+1)*dt, 1.0), device=self.device)

            k1     = self._velocity(x,        t_curr, cond, w)
            x_pred = x + k1 * dt
            k2     = self._velocity(x_pred,   t_next, cond, w)

            x = x + (k1 + k2) * 0.5 * dt

        return x   # still in normalised space

    # ──────────────────────────────────────────────────────────────────────
    # Euler sampler (fast eval / debug)
    # ──────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def euler_sample(
        self,
        batch_size: int,
        cond:       Tensor,
        nfe:        int            = 50,
        cfg_scale:  Optional[float] = None,
    ) -> Tensor:
        """First-order Euler sampler.  Faster per step, lower quality."""
        w    = cfg_scale if cfg_scale is not None else self.cfg_scale
        cond = cond.to(self.device)
        x    = torch.randn((batch_size, self.model.T, self.model.D), device=self.device)
        dt   = 1.0 / nfe
        for i in range(nfe):
            tau = torch.full((batch_size,), i * dt, device=self.device)
            x   = x + self._velocity(x, tau, cond, w) * dt
        return x