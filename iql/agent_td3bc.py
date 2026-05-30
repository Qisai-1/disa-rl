"""
TD3+BC agent (Fujimoto & Gu, NeurIPS 2021) with DiSA-RL augmentation hooks.

Why a second backbone for the AAAI table:
  GTA (NeurIPS'24) reports backbone-agnostic gains on TD3+BC. TD3+BC is
  often the strongest published baseline on D4RL medium-replay (hopper
  ~66, walker ~84). Adding it lets us:
    - Report backbone-agnostic results like GTA (TD3+BC + IQL columns).
    - Test whether our augmentation generalizes beyond IQL.
    - Compete on a backbone where the AWR-temperature mismatch we
      diagnosed for IQL does not apply (TD3+BC uses Q directly, not
      exp(beta·adv)).

Core TD3+BC:
  - Deterministic actor π(s) → action ∈ [-1, 1] (tanh).
  - Twin Q (or ensemble), target-policy smoothing on s', delayed actor.
  - Actor loss = -λ·Q(s, π(s)) + MSE(π(s), a_BC)
      where λ = α / E[|Q(s,π(s))|]  (α default 2.5; auto-normalizes Q
      so the BC term has comparable scale).

DiSA augmentation hooks (off by default — set to mirror CAPA semantics):
  - real_only_critic=True: Q is trained on REAL transitions only (critic
    immune to syn-reward bias, exactly like CAPA). Default True.
  - mixed_actor=True: actor's -Q(s, π(s)) uses the MIXED batch; syn rows
    are extra policy-improvement proposals judged by the trusted real-only
    critic. BC term remains on REAL data only.
  - unc_beta>0: optional CAPA-style uncertainty gate on syn rows of the
    actor loss: weight_syn = exp(-unc_beta · q_ensemble_std). Requires
    num_critics ≥ 3.

Constraints:
  - Sampling: requires the same AugmentedReplayBuffer + sample_real()
    contract as CAPA. train_iql.py already passes `real_batch=` when
    `args.bc_weight > 0 or args.capa`. TD3+BC needs the same — gated on
    `args.backbone == "td3bc"` in train_iql.py.
  - Not compatible with --sa_iql (TD3+BC has no expectile mechanism).

This module reuses iql.networks.QNetwork / QEnsemble / build_mlp.
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, Optional, Tuple

import sys, os as _os
_h = _os.path.dirname(_os.path.abspath(__file__))
if _os.path.dirname(_h) not in sys.path:
    sys.path.insert(0, _os.path.dirname(_h))

from iql.networks import (
    build_mlp, orthogonal_init,
    QNetwork, TwinQNetwork, QEnsemble, EMATarget,
)


class DeterministicActor(nn.Module):
    """
    Deterministic tanh-squashed policy used by TD3 / TD3+BC.

    π(s) = tanh(MLP(s)). LayerNorm in the MLP — keeps offline RL stable
    on D4RL (same convention as our V/Q networks).
    """

    def __init__(
        self,
        obs_dim:     int,
        action_dim:  int,
        hidden_dims: Tuple[int, ...] = (256, 256),
    ):
        super().__init__()
        self.action_dim = action_dim
        self.net = build_mlp(obs_dim, action_dim, hidden_dims, use_ln=True)
        orthogonal_init(self.net)
        # Small output init → policy starts near 0 (tanh ≈ 0), avoiding
        # saturated outputs at random init.
        last = list(self.net.modules())[-1]
        if isinstance(last, nn.Linear):
            nn.init.orthogonal_(last.weight, gain=0.01)
            nn.init.zeros_(last.bias)

    def forward(self, obs: Tensor) -> Tensor:
        return torch.tanh(self.net(obs))

    # API parity with GaussianActor — Evaluator calls actor.act(obs, deterministic=True).
    @torch.no_grad()
    def act(self, obs: Tensor, deterministic: bool = True) -> Tensor:
        return self.forward(obs)


# ───────────────────────────────────────────────────────────────────────────────
# Target-policy smoothing helper
# ───────────────────────────────────────────────────────────────────────────────

def _smoothed_target_action(
    actor_tgt: nn.Module,
    next_obs:  Tensor,
    noise_std: float,
    noise_clip: float,
) -> Tensor:
    """TD3 trick: noisy action + clip → less brittle critic targets."""
    a = actor_tgt(next_obs)
    noise = (torch.randn_like(a) * noise_std).clamp(-noise_clip, noise_clip)
    return (a + noise).clamp(-1.0, 1.0)


# ───────────────────────────────────────────────────────────────────────────────
# Agent
# ───────────────────────────────────────────────────────────────────────────────

class TD3BCAgent:
    """
    TD3+BC agent (Fujimoto & Gu 2021) with DiSA augmentation hooks.

    Parameters
    ----------
    obs_dim, action_dim : env dimensions.
    hidden_dims         : MLP widths for actor and critic.
    bc_alpha            : α in λ = α / E[|Q|]. Controls Q vs BC balance.
                         2.5 = TD3+BC default for D4RL locomotion.
    bc_weight           : Extra coefficient on the BC term (1.0 = pure TD3+BC,
                         0.5 = soften BC; mainly for ablations).
    discount, tau       : RL hyperparameters (γ, target EMA rate).
    policy_noise        : σ for target-policy smoothing noise.
    noise_clip          : clip range for the noise.
    policy_freq         : actor + targets updated every `policy_freq` critic
                         updates (TD3's delayed actor).
    num_critics         : 2 = TwinQ; ≥3 = QEnsemble (CAPA-style ensemble gate).
    critic_subset_size  : REDQ-style subset for the min(Q_target) in TD targets.
    real_only_critic    : if True, Q trained on REAL transitions only.
    mixed_actor         : if True, actor's -Q(s,π(s)) uses MIXED batch.
    unc_beta            : CAPA-style uncertainty gate strength on syn rows
                         of the actor loss. 0.0 disables.
    """

    def __init__(
        self,
        obs_dim:     int,
        action_dim:  int,
        hidden_dims: Tuple[int, ...] = (256, 256),
        q_hidden_dims: Optional[Tuple[int, ...]] = None,
        bc_alpha:    float = 2.5,
        bc_weight:   float = 1.0,
        discount:    float = 0.99,
        tau:         float = 0.005,
        lr_q:        float = 3e-4,
        lr_pi:       float = 3e-4,
        policy_noise: float = 0.2,
        noise_clip:  float = 0.5,
        policy_freq: int   = 2,
        num_critics: int   = 2,
        critic_subset_size: int = 2,
        real_only_critic: bool = True,
        mixed_actor:      bool = True,
        unc_beta:    float = 0.0,
        device:      torch.device = torch.device("cpu"),
    ):
        self.bc_alpha    = float(bc_alpha)
        self.bc_weight   = float(bc_weight)
        self.discount    = float(discount)
        self.policy_noise = float(policy_noise)
        self.noise_clip  = float(noise_clip)
        self.policy_freq = int(policy_freq)
        self.real_only_critic = bool(real_only_critic)
        self.mixed_actor = bool(mixed_actor)
        self.unc_beta    = float(unc_beta)
        self.device      = device

        q_h = q_hidden_dims or hidden_dims
        if num_critics >= 3:
            self.q = QEnsemble(obs_dim, action_dim, q_h,
                                num_critics=num_critics,
                                subset_size=critic_subset_size).to(device)
        else:
            self.q = TwinQNetwork(obs_dim, action_dim, q_h).to(device)
        self.num_critics = num_critics

        self.actor = DeterministicActor(obs_dim, action_dim, hidden_dims).to(device)

        self.q_tgt     = EMATarget(self.q,     tau=tau).to(device)
        self.actor_tgt = EMATarget(self.actor, tau=tau).to(device)

        self.opt_q  = torch.optim.Adam(self.q.parameters(),     lr=lr_q)
        self.opt_pi = torch.optim.Adam(self.actor.parameters(), lr=lr_pi)
        self.scaler_q  = GradScaler()
        self.scaler_pi = GradScaler()

        if unc_beta > 0.0 and num_critics < 3:
            print(f"  WARN: unc_beta={unc_beta} but num_critics={num_critics}; "
                  "ensemble std degenerate. Set --num_critics 10 for CAPA gating.")
        print(f"TD3BCAgent: critic={'real-only' if real_only_critic else 'mixed'}, "
              f"actor={'mixed-batch' if mixed_actor else 'real-only'}, "
              f"bc_alpha={bc_alpha}, unc_beta={unc_beta}, num_critics={num_critics}")

        self.total_steps = 0
        self._actor_update_counter = 0

    # ─────────────────────────────────────────────────────────────────────────
    # Update step
    # ─────────────────────────────────────────────────────────────────────────

    def update(
        self,
        batch:      Dict[str, Tensor],
        real_batch: Optional[Dict[str, Tensor]] = None,
        critic_only: bool = False,
    ) -> Dict[str, float]:
        """
        One TD3+BC update. Compatible with the AugmentedReplayBuffer protocol:
          batch      = mixed real+syn (drives actor's -Q term)
          real_batch = real-only      (drives critic + BC term)
        Both must be provided when real_only_critic or bc_weight > 0.
        """
        if (self.real_only_critic or self.bc_weight > 0.0) and real_batch is None:
            raise ValueError(
                "TD3BCAgent.update requires real_batch when real_only_critic "
                "or bc_weight>0. Call aug_buffer.sample_real(batch_size)."
            )

        metrics: Dict[str, float] = {}

        # ── 1. Critic update ────────────────────────────────────────────────
        crit_batch = real_batch if self.real_only_critic else batch
        c_obs      = crit_batch["obs"]
        c_action   = crit_batch["action"]
        c_reward   = crit_batch["reward"]
        c_next_obs = crit_batch["next_obs"]
        c_done     = crit_batch["done"]

        self.opt_q.zero_grad(set_to_none=True)
        with autocast(enabled=(self.device.type == "cuda")):
            with torch.no_grad():
                a_next = _smoothed_target_action(
                    self.actor_tgt.target, c_next_obs,
                    self.policy_noise, self.noise_clip,
                )
                q_next = self.q_tgt.target.min(c_next_obs, a_next)
                target = c_reward + self.discount * (1.0 - c_done) * q_next

            all_q  = self.q.all(c_obs, c_action)             # (M, B)
            sq_err = (all_q - target.unsqueeze(0)).pow(2)
            q_loss = sq_err.mean()

        self.scaler_q.scale(q_loss).backward()
        self.scaler_q.unscale_(self.opt_q)
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), 1.0)
        self.scaler_q.step(self.opt_q)
        self.scaler_q.update()
        self.q_tgt.update(self.q)

        metrics["loss/q"]       = q_loss.item()
        metrics["train/q_mean"] = all_q.mean().item()

        # ── 2. Actor update (delayed every policy_freq steps) ───────────────
        # critic_only path: support UTD-like extra critic updates.
        if critic_only:
            return metrics

        self._actor_update_counter += 1
        if self._actor_update_counter % self.policy_freq != 0:
            self.total_steps += 1
            return metrics

        # Actor batch: mixed (for -Q term) or real only.
        a_batch = batch if self.mixed_actor else real_batch
        a_obs    = a_batch["obs"]
        a_action = a_batch["action"]
        source   = a_batch.get("source", torch.ones(a_obs.shape[0], device=self.device))

        self.opt_pi.zero_grad(set_to_none=True)
        with autocast(enabled=(self.device.type == "cuda")):
            pi = self.actor(a_obs)

            # Use the FIRST critic for the actor gradient (TD3 convention).
            # For QEnsemble we use the ensemble mean as a smoother signal;
            # the trust signal (std) is reserved for the syn gate below.
            if isinstance(self.q, QEnsemble):
                q_pi_all  = self.q.all(a_obs, pi)            # (M, B)
                q_pi      = q_pi_all.mean(dim=0)              # (B,)
                q_pi_std  = q_pi_all.std(dim=0).detach()       # (B,) for gating
            elif isinstance(self.q, TwinQNetwork):
                q_pi, _   = self.q(a_obs, pi)
                q_pi_std  = torch.zeros_like(q_pi)
            else:                                              # single QNetwork fallback
                q_pi      = self.q(a_obs, pi)
                q_pi_std  = torch.zeros_like(q_pi)

            # TD3+BC's λ: normalizes the Q magnitude so the BC term is
            # comparably scaled. Stop gradient through |Q| (used as scalar).
            lam = self.bc_alpha / (q_pi.detach().abs().mean() + 1e-6)

            # CAPA-style uncertainty gate on syn rows (optional).
            # weight_syn = exp(-unc_beta · ensemble_std). real rows: weight=1.
            if self.unc_beta > 0.0:
                gate_syn = torch.exp(-self.unc_beta * q_pi_std)
                gate     = source + (1.0 - source) * gate_syn
            else:
                gate = torch.ones_like(q_pi)

            q_term = -(lam * gate * q_pi).mean()

            # BC term — on REAL data only. Standard TD3+BC uses MSE between
            # π(s) and the dataset action, evaluated on the same batch as
            # the Q term. With mixed_actor=True the Q term sees syn rows;
            # the BC term anchors against real (s, a) pairs only, so the
            # policy isn't pulled toward unreachable syn actions.
            if self.bc_weight > 0.0:
                r_obs    = real_batch["obs"]
                r_action = real_batch["action"]
                pi_real  = self.actor(r_obs)
                bc_loss  = F.mse_loss(pi_real, r_action)
                actor_loss = q_term + self.bc_weight * bc_loss
                metrics["loss/bc"] = bc_loss.item()
            else:
                actor_loss = q_term
                metrics["loss/bc"] = 0.0

        self.scaler_pi.scale(actor_loss).backward()
        self.scaler_pi.unscale_(self.opt_pi)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.scaler_pi.step(self.opt_pi)
        self.scaler_pi.update()
        self.actor_tgt.update(self.actor)

        metrics["loss/actor"]  = actor_loss.item()
        metrics["loss/q_term"] = q_term.item()
        metrics["train/lambda"] = float(lam.item())
        if self.unc_beta > 0.0:
            sm = source < 0.5
            metrics["train/n_syn_in_batch"] = float(int(sm.sum().item()))
            if sm.any():
                metrics["train/uncert_gate_syn_mean"] = float(gate[sm].mean().item())

        self.total_steps += 1
        return metrics

    # ─────────────────────────────────────────────────────────────────────────
    # Checkpoint
    # ─────────────────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "q":            self.q.state_dict(),
            "actor":        self.actor.state_dict(),
            "q_tgt":        self.q_tgt.target.state_dict(),
            "actor_tgt":    self.actor_tgt.target.state_dict(),
            "opt_q":        self.opt_q.state_dict(),
            "opt_pi":       self.opt_pi.state_dict(),
            "total_steps":  self.total_steps,
            "arch": {"backbone": "td3bc", "num_critics": self.num_critics,
                     "real_only_critic": self.real_only_critic,
                     "mixed_actor": self.mixed_actor,
                     "unc_beta": self.unc_beta},
        }, path)

    def load(self, path: str, actor_only: bool = False) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        self.total_steps = ckpt.get("total_steps", 0)
        if actor_only:
            print(f"TD3BC actor loaded from {path} (step {self.total_steps:,})  [actor_only]")
            return
        try:
            self.q.load_state_dict(ckpt["q"])
            self.q_tgt.target.load_state_dict(ckpt["q_tgt"])
            self.actor_tgt.target.load_state_dict(ckpt["actor_tgt"])
            for k in ("opt_q", "opt_pi"):
                if k in ckpt:
                    try: getattr(self, k).load_state_dict(ckpt[k])
                    except Exception: pass
        except RuntimeError as e:
            print(f"[TD3BC load] arch mismatch on Q — actor_only fallback ({e!s:.120s})")
        print(f"TD3BC agent loaded from {path} (step {self.total_steps:,})")


# ───────────────────────────────────────────────────────────────────────────────
# Smoke
# ───────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a = TD3BCAgent(obs_dim=11, action_dim=3, device=device,
                   num_critics=10, critic_subset_size=2,
                   real_only_critic=True, mixed_actor=True, unc_beta=1.0)
    B = 16
    mixed = dict(
        obs=torch.randn(B, 11, device=device),
        action=torch.randn(B, 3, device=device).clamp(-1, 1),
        reward=torch.randn(B, device=device),
        next_obs=torch.randn(B, 11, device=device),
        done=torch.zeros(B, device=device),
        source=torch.cat([torch.ones(B//2), torch.zeros(B//2)]).to(device),
    )
    real = {k: v[:B//2] for k, v in mixed.items() if k != "source"}
    real["source"] = torch.ones(B//2, device=device)
    # Run a few updates
    for i in range(4):
        m = a.update(mixed, real_batch=real)
        print(f"step {i}: loss/q={m['loss/q']:.3f}  loss/actor={m.get('loss/actor', 0):.3f}  "
              f"λ={m.get('train/lambda', 0):.3f}  gate_syn={m.get('train/uncert_gate_syn_mean', 1):.3f}")
    print("PASS: TD3BCAgent smoke")
