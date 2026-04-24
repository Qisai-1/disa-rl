"""
IQL agent — loss functions and parameter update steps.

IQL avoids querying the policy during Bellman backup by replacing the
max_a Q(s', a) with an implicit expectile regression on V(s').
This makes it purely offline — no OOD action evaluation.

Three update steps per gradient step:

1. V update (expectile regression):
   L_V(φ) = E_{(s,a)~D} [ Lτ(Q̄(s,a) − V_φ(s)) ]
   where Lτ(u) = |τ − 1(u < 0)| · u²
   τ = 0.7 for locomotion (asymmetric loss upweights positive advantages)

2. Q update (TD with V target):
   L_Q(θ) = E_{(s,a,r,s')~D} [ (r + γ · V_φ(s') − Q_θ(s,a))² ]
   Uses target Q̄ for the advantage in V update, online Q for the TD loss.

3. Actor update (advantage-weighted regression / AWR):
   L_π(ω) = E_{(s,a)~D} [ −exp(β · (Q̄(s,a) − V_φ(s))) · log π_ω(a|s) ]
   β = 3.0 controls how sharply to upweight high-advantage actions.
   The exp weight is clamped to [0, 100] to prevent instability.
"""

from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.cuda.amp import GradScaler, autocast
from typing import Dict, Tuple

from iql.networks import GaussianActor, TwinQNetwork, ValueNetwork, EMATarget


# ──────────────────────────────────────────────────────────────────────────────
# IQL Agent
# ──────────────────────────────────────────────────────────────────────────────

class IQLAgent:
    """
    IQL agent with augmentation-ready interface.

    Parameters
    ----------
    obs_dim, action_dim : environment dimensions
    hidden_dims         : MLP hidden layer sizes for all networks
    expectile           : τ for the expectile loss (0.7 = standard for locomotion)
    temperature         : β for AWR actor update (3.0 = standard)
    discount            : γ
    tau                 : EMA rate for target Q-network (0.005 = standard)
    lr_q, lr_v, lr_pi   : per-network learning rates
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
        device:      torch.device = torch.device("cpu"),
    ):
        self.expectile   = expectile
        self.temperature = temperature
        self.discount    = discount
        self.device      = device

        # Networks
        self.q      = TwinQNetwork(obs_dim, action_dim, hidden_dims).to(device)
        self.v      = ValueNetwork(obs_dim, hidden_dims).to(device)
        self.actor  = GaussianActor(obs_dim, action_dim, hidden_dims).to(device)
        self.q_tgt  = EMATarget(self.q, tau=tau).to(device)

        # Optimisers — separate LRs per network lets you tune them independently
        self.opt_q  = torch.optim.Adam(self.q.parameters(),     lr=lr_q)
        self.opt_v  = torch.optim.Adam(self.v.parameters(),     lr=lr_v)
        self.opt_pi = torch.optim.Adam(self.actor.parameters(), lr=lr_pi)

        # Mixed precision scalers (one per optimizer)
        self.scaler_q  = GradScaler()
        self.scaler_v  = GradScaler()
        self.scaler_pi = GradScaler()

        self.total_steps = 0

    # ──────────────────────────────────────────────────────────────────────
    # Expectile loss
    # ──────────────────────────────────────────────────────────────────────

    def _expectile_loss(self, diff: Tensor) -> Tensor:
        """
        Lτ(u) = |τ − 1(u < 0)| · u²

        For τ > 0.5:  positive residuals (Q > V) are upweighted more than
        negative ones, which makes V estimate the τ-th quantile of Q.
        τ = 0.7 → V tracks roughly the 70th percentile → conservative but
        not overly pessimistic.
        """
        weight = torch.where(diff >= 0,
                             torch.full_like(diff, self.expectile),
                             torch.full_like(diff, 1.0 - self.expectile))
        return (weight * diff.pow(2)).mean()

    # ──────────────────────────────────────────────────────────────────────
    # Single update step
    # ──────────────────────────────────────────────────────────────────────

    def update(self, batch: Dict[str, Tensor]) -> Dict[str, float]:
        """
        One gradient step on all three networks.

        Parameters
        ----------
        batch : dict with keys obs, action, reward, next_obs, done
                all tensors on self.device, shape (B, *)

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

        # ── 1. Value update ──────────────────────────────────────────────
        self.opt_v.zero_grad(set_to_none=True)
        with autocast(enabled=(self.device.type == "cuda")):
            with torch.no_grad():
                # Use target Q for the value regression target
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

        # ── 2. Q update ───────────────────────────────────────────────────
        self.opt_q.zero_grad(set_to_none=True)
        with autocast(enabled=(self.device.type == "cuda")):
            with torch.no_grad():
                # Bellman target: r + γ · V(s')
                v_next  = self.v(next_obs)
                target  = reward + self.discount * (1.0 - done) * v_next

            q1, q2 = self.q(obs, action)
            q_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)

        self.scaler_q.scale(q_loss).backward()
        self.scaler_q.unscale_(self.opt_q)
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), 1.0)
        self.scaler_q.step(self.opt_q)
        self.scaler_q.update()

        # Update target Q via EMA
        self.q_tgt.update(self.q)

        metrics["loss/q"]       = q_loss.item()
        metrics["train/q_mean"] = ((q1 + q2) / 2).mean().item()

        # ── 3. Actor update (AWR) ─────────────────────────────────────────
        self.opt_pi.zero_grad(set_to_none=True)
        with autocast(enabled=(self.device.type == "cuda")):
            with torch.no_grad():
                q_adv = self.q_tgt.target.min(obs, action)
                v_adv = self.v(obs)
                # Advantage: A(s,a) = Q(s,a) - V(s)
                # exp-weight clamped to [0, 100] for numerical stability
                adv    = (q_adv - v_adv).detach()
                weight = torch.exp(self.temperature * adv).clamp(max=100.0)

            # AWR: maximise E[w(s,a) · log π(a_offline|s)]
            # Evaluate log probability of DATASET actions under current policy
            # This regresses the policy toward high-advantage offline actions
            log_prob_offline = self.actor.log_prob(obs, action)
            actor_loss = -(weight * log_prob_offline).mean()

        self.scaler_pi.scale(actor_loss).backward()
        self.scaler_pi.unscale_(self.opt_pi)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.scaler_pi.step(self.opt_pi)
        self.scaler_pi.update()

        metrics["loss/actor"]    = actor_loss.item()
        metrics["train/adv_mean"] = adv.mean().item()

        self.total_steps += 1
        return metrics

    # ──────────────────────────────────────────────────────────────────────
    # Checkpoint
    # ──────────────────────────────────────────────────────────────────────

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
        print(f"IQL agent loaded from {path} (step {self.total_steps:,})")