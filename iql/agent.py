"""
IQL agent — loss functions and parameter update steps.

DiSA-RL improvements over standard IQL:

1. BC anchor on real data only:
   Actor loss = AWR(mixed batch) + λ_bc * BC(real batch only)
   Synthetic data used for Q/V learning, real data anchors the policy.
   Prevents policy from chasing unreachable synthetic states.

2. Separate real/synthetic batch tracking:
   The update() method accepts an optional real_batch argument.
   When provided, BC term is computed on real transitions only.
   When absent (offline_only mode), standard AWR is used.

Standard IQL losses (unchanged):
1. V update (expectile regression):
   L_V(φ) = E_{(s,a)~D} [ Lτ(Q̄(s,a) − V_φ(s)) ]
2. Q update (TD with V target):
   L_Q(θ) = E_{(s,a,r,s')~D} [ (r + γ · V_φ(s') − Q_θ(s,a))² ]
3. Actor update (AWR + BC anchor):
   L_π(ω) = E_{mixed} [ −exp(β·A(s,a)) · log π_ω(a|s) ]
           + λ_bc * E_{real} [ −log π_ω(a_real|s_real) ]
"""

from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.cuda.amp import GradScaler, autocast
from typing import Dict, Optional, Tuple

from iql.networks import GaussianActor, TwinQNetwork, ValueNetwork, EMATarget


class IQLAgent:
    """
    IQL agent with DiSA-RL improvements.

    Parameters
    ----------
    obs_dim, action_dim : environment dimensions
    hidden_dims         : MLP hidden layer sizes for all networks
    expectile           : τ for the expectile loss (0.7 = standard for locomotion)
    temperature         : β for AWR actor update (3.0 = standard)
    discount            : γ
    tau                 : EMA rate for target Q-network (0.005 = standard)
    lr_q, lr_v, lr_pi   : per-network learning rates
    bc_weight           : λ_bc — weight for BC anchor on real data (0.0 = disabled)
    """

    def __init__(
        self,
        obs_dim:     int,
        action_dim:  int,
        hidden_dims: Tuple[int, ...] = (256, 256),
        expectile:   float = 0.7,
        temperature: float = 3.0,
        discount:    float = 0.99,
        tau:         float = 0.005,
        lr_q:        float = 3e-4,
        lr_v:        float = 3e-4,
        lr_pi:       float = 3e-4,
        bc_weight:   float = 0.1,      # ← NEW: BC anchor weight
        device:      torch.device = torch.device("cpu"),
    ):
        self.expectile   = expectile
        self.temperature = temperature
        self.discount    = discount
        self.bc_weight   = bc_weight
        self.device      = device

        # Networks
        self.q      = TwinQNetwork(obs_dim, action_dim, hidden_dims).to(device)
        self.v      = ValueNetwork(obs_dim, hidden_dims).to(device)
        self.actor  = GaussianActor(obs_dim, action_dim, hidden_dims).to(device)
        self.q_tgt  = EMATarget(self.q, tau=tau).to(device)

        # Optimisers
        self.opt_q  = torch.optim.Adam(self.q.parameters(),     lr=lr_q)
        self.opt_v  = torch.optim.Adam(self.v.parameters(),     lr=lr_v)
        self.opt_pi = torch.optim.Adam(self.actor.parameters(), lr=lr_pi)

        # Mixed precision scalers
        self.scaler_q  = GradScaler()
        self.scaler_v  = GradScaler()
        self.scaler_pi = GradScaler()

        self.total_steps = 0

    # ──────────────────────────────────────────────────────────────────────────
    # Expectile loss
    # ──────────────────────────────────────────────────────────────────────────

    def _expectile_loss(self, diff: Tensor) -> Tensor:
        """Lτ(u) = |τ − 1(u < 0)| · u²"""
        weight = torch.where(diff >= 0,
                             torch.full_like(diff, self.expectile),
                             torch.full_like(diff, 1.0 - self.expectile))
        return (weight * diff.pow(2)).mean()

    # ──────────────────────────────────────────────────────────────────────────
    # Single update step
    # ──────────────────────────────────────────────────────────────────────────

    def update(
        self,
        batch:      Dict[str, Tensor],
        real_batch: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, float]:
        """
        One gradient step on all three networks.

        Parameters
        ----------
        batch      : mixed batch (real + synthetic) for Q/V/actor updates
        real_batch : real data only for BC anchor (optional)
                     If None, BC anchor is skipped (offline_only mode)

        Returns
        -------
        dict of scalar metrics for logging
        """
        obs      = batch["obs"]
        action   = batch["action"]
        reward   = batch["reward"]
        next_obs = batch["next_obs"]
        done     = batch["done"]

        metrics = {}

        # ── 1. Value update ────────────────────────────────────────────────
        self.opt_v.zero_grad(set_to_none=True)
        with autocast(enabled=(self.device.type == "cuda")):
            with torch.no_grad():
                q_tgt = self.q_tgt.target.min(obs, action)
            v      = self.v(obs)
            v_loss = self._expectile_loss(q_tgt - v)

        self.scaler_v.scale(v_loss).backward()
        self.scaler_v.unscale_(self.opt_v)
        torch.nn.utils.clip_grad_norm_(self.v.parameters(), 1.0)
        self.scaler_v.step(self.opt_v)
        self.scaler_v.update()

        metrics["loss/v"]       = v_loss.item()
        metrics["train/v_mean"] = v.mean().item()

        # ── 2. Q update ────────────────────────────────────────────────────
        self.opt_q.zero_grad(set_to_none=True)
        with autocast(enabled=(self.device.type == "cuda")):
            with torch.no_grad():
                v_next = self.v(next_obs)
                target = reward + self.discount * (1.0 - done) * v_next
            q1, q2 = self.q(obs, action)
            q_loss  = F.mse_loss(q1, target) + F.mse_loss(q2, target)

        self.scaler_q.scale(q_loss).backward()
        self.scaler_q.unscale_(self.opt_q)
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), 1.0)
        self.scaler_q.step(self.opt_q)
        self.scaler_q.update()
        self.q_tgt.update(self.q)

        metrics["loss/q"]       = q_loss.item()
        metrics["train/q_mean"] = ((q1 + q2) / 2).mean().item()

        # ── 3. Actor update (AWR + BC anchor) ─────────────────────────────
        self.opt_pi.zero_grad(set_to_none=True)
        with autocast(enabled=(self.device.type == "cuda")):

            # ── 3a. AWR loss on mixed batch ──────────────────────────────
            with torch.no_grad():
                q_adv  = self.q_tgt.target.min(obs, action)
                v_adv  = self.v(obs)
                adv    = (q_adv - v_adv).detach()
                weight = torch.exp(self.temperature * adv).clamp(max=100.0)

            # log π(a_dataset | s) — policy evaluated at dataset actions
            log_prob_awr = self.actor.log_prob(obs, action)
            awr_loss     = -(weight * log_prob_awr).mean()

            # ── 3b. BC anchor on REAL data only ──────────────────────────
            # Keeps policy close to real dataset distribution.
            # Prevents chasing unreachable synthetic states.
            if real_batch is not None and self.bc_weight > 0.0:
                real_obs    = real_batch["obs"]
                real_action = real_batch["action"]
                log_prob_bc = self.actor.log_prob(real_obs, real_action)
                bc_loss     = -log_prob_bc.mean()
                actor_loss  = awr_loss + self.bc_weight * bc_loss
                metrics["loss/bc"] = bc_loss.item()
            else:
                actor_loss = awr_loss
                metrics["loss/bc"] = 0.0

        self.scaler_pi.scale(actor_loss).backward()
        self.scaler_pi.unscale_(self.opt_pi)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.scaler_pi.step(self.opt_pi)
        self.scaler_pi.update()

        metrics["loss/actor"]    = actor_loss.item()
        metrics["loss/awr"]      = awr_loss.item()
        metrics["train/adv_mean"] = adv.mean().item()

        self.total_steps += 1
        return metrics

    # ──────────────────────────────────────────────────────────────────────────
    # Checkpoint
    # ──────────────────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "q":           self.q.state_dict(),
            "v":           self.v.state_dict(),
            "actor":       self.actor.state_dict(),
            "q_tgt":       self.q_tgt.target.state_dict(),
            "opt_q":       self.opt_q.state_dict(),
            "opt_v":       self.opt_v.state_dict(),
            "opt_pi":      self.opt_pi.state_dict(),
            "total_steps": self.total_steps,
            "bc_weight":   self.bc_weight,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.q.load_state_dict(ckpt["q"])
        self.v.load_state_dict(ckpt["v"])
        self.actor.load_state_dict(ckpt["actor"])
        self.q_tgt.target.load_state_dict(ckpt["q_tgt"])
        self.opt_q.load_state_dict(ckpt["opt_q"])
        self.opt_v.load_state_dict(ckpt["opt_v"])
        self.opt_pi.load_state_dict(ckpt["opt_pi"])
        self.total_steps = ckpt["total_steps"]
        self.bc_weight   = ckpt.get("bc_weight", self.bc_weight)
        print(f"IQL agent loaded from {path} (step {self.total_steps:,})")