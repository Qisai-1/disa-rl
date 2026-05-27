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

from iql.networks import (
    GaussianActor, TwinQNetwork, ValueNetwork, EMATarget, QEnsemble,
)
from iql.discriminator import DensityRatioDiscriminator


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
        q_hidden_dims: Optional[Tuple[int, ...]] = None,
        v_hidden_dims: Optional[Tuple[int, ...]] = None,
        expectile:   float = 0.7,
        expectile_real: Optional[float] = None,     # SA-IQL mixture (real path)
        expectile_syn:  Optional[float] = None,     # SA-IQL mixture (syn path)
        temperature: float = 3.0,
        discount:    float = 0.99,
        tau:         float = 0.005,
        lr_q:        float = 3e-4,
        lr_v:        float = 3e-4,
        lr_pi:       float = 3e-4,
        lr_d:        float = 3e-4,                  # discriminator lr
        bc_weight:   float = 0.1,
        adv_normalize: bool = True,
        num_critics:  int  = 2,                     # 2 = TwinQ; ≥3 = QEnsemble
        critic_subset_size: int = 2,                # REDQ subset for min target
        sa_iql:       bool = False,                 # enable SA-IQL
        sa_clip:      Tuple[float, float] = (0.5, 2.0),
        action_noise_std: float = 0.0,              # action-noise augmentation
        pa_weight:    float = 0.0,                  # PARS-style PA loss weight
        pa_min_q:     float = 0.0,                  # PA loss Q lower bound
        device:      torch.device = torch.device("cpu"),
    ):
        self.expectile   = expectile
        # SA-IQL: per-source expectile. If unset, fall back to single expectile.
        self.expectile_real = expectile_real if expectile_real is not None else expectile
        self.expectile_syn  = expectile_syn  if expectile_syn  is not None else expectile
        self.temperature = temperature
        self.discount    = discount
        self.bc_weight   = bc_weight
        self.adv_normalize = adv_normalize
        self.sa_iql       = sa_iql
        self.sa_clip      = sa_clip
        self.action_noise_std = action_noise_std
        self.pa_weight    = pa_weight
        self.pa_min_q     = pa_min_q
        self.action_dim   = action_dim
        self.device      = device

        # Networks. Q/V can be wider than the actor — common D4RL trick.
        q_h = q_hidden_dims or hidden_dims
        v_h = v_hidden_dims or hidden_dims
        if num_critics >= 3:
            self.q = QEnsemble(obs_dim, action_dim, q_h,
                               num_critics=num_critics,
                               subset_size=critic_subset_size).to(device)
        else:
            self.q = TwinQNetwork(obs_dim, action_dim, q_h).to(device)
        self.num_critics = num_critics
        self.v      = ValueNetwork(obs_dim, v_h).to(device)
        self.actor  = GaussianActor(obs_dim, action_dim, hidden_dims).to(device)
        self.q_tgt  = EMATarget(self.q, tau=tau).to(device)

        # SA-IQL: density-ratio discriminator
        if self.sa_iql:
            self.disc = DensityRatioDiscriminator(obs_dim, action_dim,
                                                  hidden_dims=q_h).to(device)
            self.opt_d = torch.optim.Adam(self.disc.parameters(), lr=lr_d)
        else:
            self.disc = None
            self.opt_d = None

        # Running stats for AWR advantage normalization
        self._adv_running_mean = 0.0
        self._adv_running_var  = 1.0
        self._adv_running_n    = 0

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

    def _mixture_expectile_loss(self, diff: Tensor, source: Tensor) -> Tensor:
        """
        SA-IQL: per-sample expectile τ(source).
          source=1 (real) → τ = expectile_real (optimistic)
          source=0 (syn)  → τ = expectile_syn  (conservative)
        """
        tau = source * self.expectile_real + (1.0 - source) * self.expectile_syn
        weight = torch.where(diff >= 0, tau, 1.0 - tau)
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
        source   = batch.get("source", torch.ones_like(reward))   # default = all real

        metrics = {}

        # ── 0. SA-IQL: train discriminator on real vs syn within the batch
        if self.sa_iql and self.disc is not None:
            real_mask = source > 0.5
            syn_mask  = ~real_mask
            n_real, n_syn = int(real_mask.sum()), int(syn_mask.sum())
            if n_real > 1 and n_syn > 1:
                d_loss = self.disc.bce_loss(
                    obs[real_mask], action[real_mask],
                    obs[syn_mask],  action[syn_mask],
                )
                self.opt_d.zero_grad(set_to_none=True)
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.disc.parameters(), 1.0)
                self.opt_d.step()
                metrics["loss/disc"] = d_loss.item()

        # ── 1. Value update ────────────────────────────────────────────────
        self.opt_v.zero_grad(set_to_none=True)
        with autocast(enabled=(self.device.type == "cuda")):
            with torch.no_grad():
                q_tgt = self.q_tgt.target.min(obs, action)
            v      = self.v(obs)
            v_diff = q_tgt - v
            if self.sa_iql:
                v_loss = self._mixture_expectile_loss(v_diff, source)
            else:
                v_loss = self._expectile_loss(v_diff)

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

                # SA-IQL: per-transition density-ratio weight
                if self.sa_iql and self.disc is not None:
                    w = self.disc.density_ratio(obs, action, clip=self.sa_clip)
                    # real transitions get w=1 (no correction needed),
                    # syn transitions get the importance ratio
                    w = source + (1.0 - source) * w
                else:
                    w = None

            all_q = self.q.all(obs, action)                       # (M, B)
            target_exp = target.unsqueeze(0).expand_as(all_q)
            sq_err = (all_q - target_exp).pow(2)                  # (M, B)
            if w is not None:
                sq_err = sq_err * w.unsqueeze(0)
            q_loss = sq_err.mean()

            # PARS-style PA loss: sample actions OUTSIDE the valid tanh range
            # and regress their Q values toward pa_min_q. Acts as a soft
            # support constraint — keeps Q from extrapolating to absurd
            # values on infeasible actions. Disabled if pa_weight == 0.
            if self.pa_weight > 0.0:
                # uniform in [-1, 1] → shift OUT of [-1, 1]: [-2, -1] ∪ [1, 2]
                u = torch.rand_like(action) * 2.0 - 1.0
                ood_action = torch.where(u < 0, u - 1.0, u + 1.0)
                ood_q = self.q.all(obs, ood_action)              # (M, B)
                pa_target = torch.full_like(ood_q, self.pa_min_q)
                pa_loss = (ood_q - pa_target).pow(2).mean()
                q_loss = q_loss + self.pa_weight * pa_loss
                metrics["loss/pa"] = pa_loss.item()
                metrics["train/q_ood_mean"] = ood_q.mean().item()

        self.scaler_q.scale(q_loss).backward()
        self.scaler_q.unscale_(self.opt_q)
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), 1.0)
        self.scaler_q.step(self.opt_q)
        self.scaler_q.update()
        self.q_tgt.update(self.q)

        metrics["loss/q"]       = q_loss.item()
        metrics["train/q_mean"] = all_q.mean().item()
        if w is not None:
            metrics["train/w_mean"] = w.mean().item()
            metrics["train/w_max"]  = w.max().item()

        # ── 3. Actor update (AWR + BC anchor) ─────────────────────────────
        self.opt_pi.zero_grad(set_to_none=True)
        with autocast(enabled=(self.device.type == "cuda")):

            # ── 3a. AWR loss on mixed batch ──────────────────────────────
            with torch.no_grad():
                q_adv  = self.q_tgt.target.min(obs, action)
                v_adv  = self.v(obs)
                adv    = (q_adv - v_adv).detach()
                if self.adv_normalize:
                    # Running mean/var of advantages (Welford-style)
                    n_new = self._adv_running_n + adv.numel()
                    mean_b = adv.mean().item()
                    delta = mean_b - self._adv_running_mean
                    self._adv_running_mean += delta * adv.numel() / n_new
                    var_b = adv.var(unbiased=False).item()
                    self._adv_running_var = (
                        (self._adv_running_n * self._adv_running_var
                         + adv.numel() * var_b
                         + delta**2 * self._adv_running_n * adv.numel() / n_new)
                        / n_new
                    )
                    self._adv_running_n = n_new
                    std = max(self._adv_running_var, 1e-6) ** 0.5
                    adv_for_weight = (adv - self._adv_running_mean) / std
                else:
                    adv_for_weight = adv
                weight = torch.exp(self.temperature * adv_for_weight).clamp(max=100.0)

            # Action-noise augmentation — adds small Gaussian noise to the
            # target action before computing log-prob. Acts as a Jacobian
            # regularizer on the policy and reduces overfitting to specific
            # noisy synthetic actions. Set --action_noise_std 0 to disable.
            action_for_logp = action
            if self.action_noise_std > 0:
                noise = torch.randn_like(action) * self.action_noise_std
                action_for_logp = (action + noise).clamp(-1 + 1e-5, 1 - 1e-5)

            log_prob_awr = self.actor.log_prob(obs, action_for_logp)
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
        payload = {
            "q":           self.q.state_dict(),
            "v":           self.v.state_dict(),
            "actor":       self.actor.state_dict(),
            "q_tgt":       self.q_tgt.target.state_dict(),
            "opt_q":       self.opt_q.state_dict(),
            "opt_v":       self.opt_v.state_dict(),
            "opt_pi":      self.opt_pi.state_dict(),
            "total_steps": self.total_steps,
            "bc_weight":   self.bc_weight,
            "arch": {
                "num_critics":    self.num_critics,
                "sa_iql":         self.sa_iql,
                "expectile_real": self.expectile_real,
                "expectile_syn":  self.expectile_syn,
            },
        }
        if self.disc is not None:
            payload["disc"] = self.disc.state_dict()
        torch.save(payload, path)

    def load(self, path: str, actor_only: bool = False) -> None:
        """
        Load a checkpoint. If `actor_only=True`, skip Q/V/disc (useful for
        eval where only the actor is needed and arch may differ).
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        # Actor is always loadable (its architecture is fixed across runs)
        self.actor.load_state_dict(ckpt["actor"])
        self.total_steps = ckpt.get("total_steps", 0)
        self.bc_weight   = ckpt.get("bc_weight", self.bc_weight)

        if actor_only:
            print(f"IQL actor loaded from {path} (step {self.total_steps:,})  [actor_only]")
            return

        # Try strict load on Q/V; fall back to actor-only if arch differs.
        try:
            self.q.load_state_dict(ckpt["q"])
            self.v.load_state_dict(ckpt["v"])
            self.q_tgt.target.load_state_dict(ckpt["q_tgt"])
            for k in ("opt_q", "opt_v", "opt_pi"):
                if k in ckpt:
                    try: getattr(self, k).load_state_dict(ckpt[k])
                    except Exception: pass
            if "disc" in ckpt and self.disc is not None:
                try: self.disc.load_state_dict(ckpt["disc"])
                except Exception: pass
        except RuntimeError as e:
            print(f"[load] arch mismatch on Q/V — actor_only fallback ({e!s:.120s})")

        print(f"IQL agent loaded from {path} (step {self.total_steps:,})")