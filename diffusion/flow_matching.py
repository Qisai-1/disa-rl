"""
Conditional Flow Matching for (obs, action) trajectory generation.

No reward dimension — diffusion model learns dynamics only.
Rewards computed separately by reward_computer.py.

Loss:
    L = λ_obs · ||v_obs − target_obs||²
      + λ_action · ||v_action − target_action||²
      + λ_temp · ||Δobs_pred_{t+1,t} − Δobs_real_{t+1,t}||²

Pure float32 — no autocast. Avoids NaN with torch.compile on PyTorch 2.2.
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

from model import TrajectoryDiT
from config import LossConfig


def sample_logit_normal(batch_size, device, mean=0.0, std=1.0):
    return torch.sigmoid(torch.randn(batch_size, device=device) * std + mean)


class ConditionalFlowMatching:
    def __init__(self, model: TrajectoryDiT, device: torch.device,
                 loss_cfg: LossConfig = None, cfg_scale: float = 1.5):
        self.model     = model
        self.device    = device
        self.loss_cfg  = loss_cfg or LossConfig()
        self.cfg_scale = cfg_scale
        # Kinematic-consistency loss state (set via set_kinematics; off by default)
        self.kin = None          # dict(n_pos, vel_offset, dt) or None
        self._obs_mean = None     # (obs_dim,) tensor, raw-space denorm
        self._obs_std  = None
        self._kin_scale = None    # (n_pos,) tensor, per-dim real |Δpos| scale

    def set_kinematics(self, obs_mean, obs_std, n_pos: int, vel_offset: int,
                       dt: float, scale, device=None):
        """Enable the kinematic-consistency loss.

        obs_mean/obs_std : (obs_dim,) normalizer stats (denormalize x1_pred).
        n_pos            : number of leading obs dims that are positions.
        vel_offset       : index where the matching velocities start; pos[i]
                           pairs with obs[vel_offset + i].
        dt               : env physics timestep (Δpos ≈ dt·vel).
        scale            : (n_pos,) per-dim real mean|Δpos| — makes the residual
                           dimensionless so lambda_dyn is env-agnostic.
        """
        dev = device or self.device
        t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=dev)
        self._obs_mean = t(obs_mean)
        self._obs_std  = t(obs_std)
        self._kin_scale = t(scale)
        self.kin = dict(n_pos=int(n_pos), vel_offset=int(vel_offset), dt=float(dt))

    def get_train_tuple(self, x1: Tensor):
        x0    = torch.randn_like(x1)
        tau   = sample_logit_normal(x1.shape[0], x1.device)
        x_tau = tau[:, None, None] * x1 + (1.0 - tau[:, None, None]) * x0
        return x_tau, tau, x1 - x0, x0

    def loss(self, x1: Tensor, cond: Tensor) -> Tuple[Tensor, Dict[str, float]]:
        """
        x1   : (B, T, D)  clean (obs, action) trajectory, D = obs_dim + action_dim
        cond : (B, cond_dim)
        Returns: scalar loss, metrics dict
        """
        x1   = x1.to(self.device)
        cond = cond.to(self.device)

        x_tau, tau, target, _ = self.get_train_tuple(x1)

        drop_mask = torch.rand(x1.shape[0], device=self.device) < self.model.cfg_dropout_prob

        # Pure float32 — no autocast
        velocity = self.model(x_tau, tau, cond, drop_mask)

        lc         = self.loss_cfg
        obs_dim    = self.model.obs_dim
        action_dim = self.model.action_dim

        # Per-modality losses
        L_obs    = F.mse_loss(velocity[..., :obs_dim],               target[..., :obs_dim])
        L_action = F.mse_loss(velocity[..., obs_dim:obs_dim+action_dim], target[..., obs_dim:obs_dim+action_dim])

        # Temporal consistency — first-order obs differences
        x1_pred    = x_tau + (1.0 - tau[:, None, None]) * velocity
        delta_pred = x1_pred[:, 1:, :obs_dim] - x1_pred[:, :-1, :obs_dim]
        delta_real = x1[:, 1:, :obs_dim]       - x1[:, :-1, :obs_dim]
        L_temporal = F.mse_loss(delta_pred, delta_real)

        total = (lc.lambda_obs    * L_obs
               + lc.lambda_action * L_action
               + lc.lambda_temporal * L_temporal)

        metrics = {
            "loss/obs":      L_obs.item(),
            "loss/action":   L_action.item(),
            "loss/temporal": L_temporal.item(),
        }

        # Kinematic-consistency — Δpos ≈ dt·velocity WITHIN the predicted clean
        # trajectory, in raw obs space (denormalize so dt·vel has the right
        # scale incl. velocity mean; divide by per-dim real |Δpos| → dimensionless).
        # Weighted by tau²: at high noise (tau→0) x1_pred is unreliable, so only
        # enforce physics where the clean-trajectory estimate is trustworthy.
        if getattr(lc, "lambda_dyn", 0.0) > 0.0 and self.kin is not None:
            npos = self.kin["n_pos"]; voff = self.kin["vel_offset"]; dtp = self.kin["dt"]
            obs_pred = x1_pred[..., :obs_dim] * self._obs_std + self._obs_mean   # (B,T,obs_dim) raw
            dpos = obs_pred[:, 1:, :npos] - obs_pred[:, :-1, :npos]              # (B,T-1,npos)
            vel  = obs_pred[:, :-1, voff:voff + npos]
            resid = (dpos - dtp * vel) / self._kin_scale                         # dimensionless
            w = (tau ** 2)[:, None, None]                                        # (B,1,1)
            L_dyn = (w * resid.pow(2)).mean()
            total = total + lc.lambda_dyn * L_dyn
            metrics["loss/dyn"] = L_dyn.item()

        metrics["loss/total"] = total.item()
        return total, metrics

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
        v2           = self.model(x2, tau2, cond2, drop)
        v_cond, v_un = v2[:B], v2[B:]
        return v_un + cfg_scale * (v_cond - v_un)

    @torch.no_grad()
    def heun_sample(self, batch_size, cond, nfe=20, cfg_scale=None):
        w    = cfg_scale if cfg_scale is not None else self.cfg_scale
        cond = cond.to(self.device)
        x    = torch.randn((batch_size, self.model.T, self.model.D), device=self.device)
        dt   = 1.0 / nfe
        for i in range(nfe):
            t_curr = torch.full((batch_size,), i * dt,           device=self.device)
            t_next = torch.full((batch_size,), min((i+1)*dt,1.0), device=self.device)
            k1     = self._velocity(x,        t_curr, cond, w)
            x_pred = x + k1 * dt
            k2     = self._velocity(x_pred,   t_next, cond, w)
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