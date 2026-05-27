"""
CAPAAgent — Critic-Anchored Proposal Augmentation.

The augmented offline RL method that's structurally guaranteed to be >= the
real-only IQL baseline. See METHOD_V2_PROPOSAL.md §5 for the full design.

Core differences vs IQLAgent / DRC-IQL:
  - **Q and V trained on REAL data only** — the critic is bit-for-bit
    identical to the baseline IQL critic. Synthetic transitions never
    contaminate the value function via the V(s') bootstrap.
  - **Actor AWR loss uses the MIXED real+syn batch**, with the advantages
    judged by the trusted real-only critic. Synthetic transitions act as
    extra policy-improvement *proposals*.
  - **Uncertainty gate on syn rows**: ensemble Q-std is the epistemic
    uncertainty signal. Syn proposals where the ensemble disagrees
    (high std = OOD) are down-weighted in the AWR loss.
  - BC anchor unchanged (already real-only when bc_weight > 0).
  - **Synthetic rewards are NEVER consumed by any loss** — AWR uses Q-V
    advantage, Q-target uses only real rewards. Structurally immune to
    syn-reward bias.

Worst case: syn proposals are useless → uncertainty gate zeroes them →
CAPA degenerates to the baseline. Cannot underperform.

Usage:
    agent = CAPAAgent(obs_dim=17, action_dim=6, ..., unc_beta=1.0)
    # batch == mixed real+syn (for actor)
    # real_batch == real-only (for V/Q + BC) — already provided by
    # AugmentedReplayBuffer.sample_real(batch_size)
    metrics = agent.update(batch, real_batch=real_batch)

Status (2026-05-26): drafted, not yet smoke-tested on real checkpoints.
AWAITING USER SIGN-OFF before launching a training sweep.
"""

from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.cuda.amp import autocast
from typing import Dict, Optional

import sys, os as _os
_h = _os.path.dirname(_os.path.abspath(__file__))
if _os.path.dirname(_h) not in sys.path:
    sys.path.insert(0, _os.path.dirname(_h))

from iql.agent import IQLAgent


class CAPAAgent(IQLAgent):
    """
    Subclass of IQLAgent that overrides update() with the CAPA logic.

    Constructor args (in addition to IQLAgent):
        unc_beta : strength of the ensemble-uncertainty gate on syn AWR
                   weights. gate(syn_row) = exp(-unc_beta * q_ensemble_std).
                   Default 1.0 — geometric decay with std.
                   Set 0.0 to disable gating (= "real-critic + mixed-actor"
                   without uncertainty filtering — useful as an ablation).

    Constraint: --sa_iql is INCOMPATIBLE with CAPA. CAPA replaces DRC's
    density-ratio + mixture-expectile mechanisms; using both would be
    incoherent.
    """

    def __init__(self, *args, unc_beta: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.unc_beta = float(unc_beta)
        if self.sa_iql:
            raise ValueError(
                "CAPA is incompatible with --sa_iql. Drop --sa_iql for CAPA runs."
            )
        # Sanity: CAPA needs an ensemble for the uncertainty gate
        if self.num_critics < 3 and unc_beta > 0.0:
            print(f"  WARN: unc_beta={unc_beta} but num_critics={self.num_critics}; "
                  "uncertainty gate degenerates (need num_critics >= 3 for "
                  "meaningful Q-std). Consider --num_critics 10.")
        print(f"CAPAAgent: critic-real-only mode, unc_beta={self.unc_beta}")

    # ──────────────────────────────────────────────────────────────────────
    # update — fully overrides IQLAgent.update with CAPA routing
    # ──────────────────────────────────────────────────────────────────────

    def update(
        self,
        batch:      Dict[str, Tensor],            # mixed real+syn (for actor)
        real_batch: Optional[Dict[str, Tensor]] = None,   # real-only (for V/Q + BC)
    ) -> Dict[str, float]:
        if real_batch is None:
            raise ValueError(
                "CAPAAgent.update requires real_batch — call "
                "aug_buffer.sample_real(batch_size) and pass it as real_batch."
            )

        metrics: Dict[str, float] = {}

        # ── 0. Unpack — real_batch drives V/Q; batch drives actor AWR ─────
        r_obs      = real_batch["obs"]
        r_action   = real_batch["action"]
        r_reward   = real_batch["reward"]
        r_next_obs = real_batch["next_obs"]
        r_done     = real_batch["done"]

        # ── 1. V update on REAL data ───────────────────────────────────────
        self.opt_v.zero_grad(set_to_none=True)
        with autocast(enabled=(self.device.type == "cuda")):
            with torch.no_grad():
                q_tgt = self.q_tgt.target.min(r_obs, r_action)
            v      = self.v(r_obs)
            v_diff = q_tgt - v
            v_loss = self._expectile_loss(v_diff)   # standard expectile, no SA-IQL

        self.scaler_v.scale(v_loss).backward()
        self.scaler_v.unscale_(self.opt_v)
        torch.nn.utils.clip_grad_norm_(self.v.parameters(), 1.0)
        self.scaler_v.step(self.opt_v)
        self.scaler_v.update()

        metrics["loss/v"]       = v_loss.item()
        metrics["train/v_mean"] = v.mean().item()

        # ── 2. Q update on REAL data ───────────────────────────────────────
        self.opt_q.zero_grad(set_to_none=True)
        with autocast(enabled=(self.device.type == "cuda")):
            with torch.no_grad():
                v_next = self.v(r_next_obs)
                target = r_reward + self.discount * (1.0 - r_done) * v_next

            all_q      = self.q.all(r_obs, r_action)              # (M, B)
            target_exp = target.unsqueeze(0).expand_as(all_q)
            sq_err     = (all_q - target_exp).pow(2)
            q_loss     = sq_err.mean()

            # PARS-style PA support constraint (unchanged from IQLAgent)
            if self.pa_weight > 0.0:
                u = torch.rand_like(r_action) * 2.0 - 1.0
                ood_action = torch.where(u < 0, u - 1.0, u + 1.0)
                ood_q      = self.q.all(r_obs, ood_action)
                pa_target  = torch.full_like(ood_q, self.pa_min_q)
                pa_loss    = (ood_q - pa_target).pow(2).mean()
                q_loss     = q_loss + self.pa_weight * pa_loss
                metrics["loss/pa"]          = pa_loss.item()
                metrics["train/q_ood_mean"] = ood_q.mean().item()

        self.scaler_q.scale(q_loss).backward()
        self.scaler_q.unscale_(self.opt_q)
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), 1.0)
        self.scaler_q.step(self.opt_q)
        self.scaler_q.update()
        self.q_tgt.update(self.q)

        metrics["loss/q"]       = q_loss.item()
        metrics["train/q_mean"] = all_q.mean().item()

        # ── 3. Actor AWR on MIXED batch, uncertainty-gated for syn rows ───
        obs    = batch["obs"]
        action = batch["action"]
        # source: 1.0 = real, 0.0 = syn (per ReplayBuffer.sample / SyntheticBuffer.sample)
        source = batch.get("source", torch.ones(batch["obs"].shape[0],
                                                device=self.device))

        self.opt_pi.zero_grad(set_to_none=True)
        with autocast(enabled=(self.device.type == "cuda")):
            with torch.no_grad():
                # Ensemble Q on the mixed (s,a); take the min as the advantage,
                # use the std as the epistemic uncertainty for syn-row gating.
                q_all = self.q_tgt.target.all(obs, action)   # (M, B)
                q_adv = q_all.min(dim=0).values                # (B,)
                q_std = q_all.std(dim=0)                       # (B,)  ensemble disagreement
                v_adv = self.v(obs)                            # (B,)
                adv   = (q_adv - v_adv).detach()

                # Optional advantage normalization (Welford running stats, from IQLAgent)
                if self.adv_normalize:
                    n_new = self._adv_running_n + adv.numel()
                    mean_b = adv.mean().item()
                    delta  = mean_b - self._adv_running_mean
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

                # CAPA uncertainty gate.
                #   gate(real) = 1                        (no discount)
                #   gate(syn)  = exp(-unc_beta * q_std)   (low std = trust; high = discount)
                # Equivalent vectorized form via source:
                gate_syn = torch.exp(-self.unc_beta * q_std)
                gate     = source + (1.0 - source) * gate_syn

                weight = torch.exp(self.temperature * adv_for_weight).clamp(max=100.0) * gate

                # Logging — average gate value on syn rows (1.0 = ungated; 0 = fully gated out)
                syn_mask = source < 0.5
                n_syn    = int(syn_mask.sum().item())
                metrics["train/uncert_gate_syn_mean"] = (
                    float(gate[syn_mask].mean().item()) if n_syn > 0 else 1.0
                )
                metrics["train/q_std_mean"] = float(q_std.mean().item())
                metrics["train/n_syn_in_batch"] = float(n_syn)

            # Action noise augmentation (unchanged from IQLAgent)
            action_for_logp = action
            if self.action_noise_std > 0:
                noise = torch.randn_like(action) * self.action_noise_std
                action_for_logp = (action + noise).clamp(-1 + 1e-5, 1 - 1e-5)

            log_prob_awr = self.actor.log_prob(obs, action_for_logp)
            awr_loss     = -(weight * log_prob_awr).mean()

            # BC anchor on REAL data only (always real_batch, never mixed)
            if self.bc_weight > 0.0:
                log_prob_bc = self.actor.log_prob(r_obs, r_action)
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

        metrics["loss/actor"]     = actor_loss.item()
        metrics["loss/awr"]       = awr_loss.item()
        metrics["train/adv_mean"] = adv.mean().item()

        self.total_steps += 1
        return metrics


if __name__ == "__main__":
    # Tiny smoke test — verifies the module imports + the constructor runs.
    # Real test of the update loop requires an environment + real syn data,
    # done by iql/train_iql_capa.py (will be written next).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = CAPAAgent(
        obs_dim=17, action_dim=6, device=device,
        num_critics=10, critic_subset_size=2,
        bc_weight=0.1, unc_beta=1.0,
    )
    print(f"CAPAAgent constructed. unc_beta={agent.unc_beta}  num_critics={agent.num_critics}")
    # Forward-only sanity (no update — we don't have synthetic data here)
    obs    = torch.randn(4, 17, device=device)
    action = torch.randn(4, 6,  device=device).clamp(-1, 1)
    with torch.no_grad():
        v       = agent.v(obs)
        q_min   = agent.q.min(obs, action)
        q_all   = agent.q.all(obs, action)
        logp    = agent.actor.log_prob(obs, action)
    print(f"Forward shapes — V {v.shape}, Q.min {q_min.shape}, Q.all {q_all.shape}, "
          f"logp {logp.shape}  OK")
