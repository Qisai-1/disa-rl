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

    def __init__(self, *args, unc_beta: float = 1.0,
                 critic_syn_gate: bool = False, critic_syn_coef: float = 1.0,
                 # Calibrated-stitching novelty knobs (default = legacy CAPA+):
                 awr_gate_mode: str = "scale",           # "scale" | "temper"
                 gbc_weight: float = 0.0,                # generative-BC anchor weight
                 gbc_gate_min: float = 0.5,              # gate threshold for GBC inclusion
                 asym_expectile_syn: bool = False,       # τ_syn = 0.5 + 0.2·gate
                 critic_syn_coef_warmup: int = 0,        # linear ramp 0→coef over N steps
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.unc_beta = float(unc_beta)
        # CAPA+: when True, low-uncertainty synthetic transitions ALSO enter the
        # V/Q updates (gated by exp(-unc_beta·Q_std)), instead of the critic
        # being strictly real-only. Trades a bit of CAPA's reward-immunity for
        # extra critic coverage — justified now that syn data is physics-
        # consistent and the reward is the exact analytic reward. gate→0 on
        # untrusted syn ⇒ degenerates back to vanilla CAPA (real-only critic).
        self.critic_syn_gate = bool(critic_syn_gate)
        self.critic_syn_coef = float(critic_syn_coef)

        # Novelty knobs — see METHOD docstring at top of file
        assert awr_gate_mode in ("scale", "temper"), awr_gate_mode
        self.awr_gate_mode           = awr_gate_mode
        self.gbc_weight              = float(gbc_weight)
        self.gbc_gate_min            = float(gbc_gate_min)
        self.asym_expectile_syn      = bool(asym_expectile_syn)
        self.critic_syn_coef_warmup  = int(critic_syn_coef_warmup)

        if self.sa_iql:
            raise ValueError(
                "CAPA is incompatible with --sa_iql. Drop --sa_iql for CAPA runs."
            )
        # Sanity: CAPA needs an ensemble for the uncertainty gate
        if self.num_critics < 3 and unc_beta > 0.0:
            print(f"  WARN: unc_beta={unc_beta} but num_critics={self.num_critics}; "
                  "uncertainty gate degenerates (need num_critics >= 3 for "
                  "meaningful Q-std). Consider --num_critics 10.")
        mode = "CAPA+ (gated-syn critic)" if self.critic_syn_gate else "critic-real-only"
        extras = []
        if self.awr_gate_mode != "scale":         extras.append(f"awr={self.awr_gate_mode}")
        if self.gbc_weight > 0:                    extras.append(f"gbc={self.gbc_weight}@{self.gbc_gate_min}")
        if self.asym_expectile_syn:                extras.append("asym_expectile")
        if self.critic_syn_coef_warmup > 0:        extras.append(f"coef_warmup={self.critic_syn_coef_warmup}")
        print(f"CAPAAgent: {mode} mode, unc_beta={self.unc_beta}"
              + (f", critic_syn_coef={self.critic_syn_coef}" if self.critic_syn_gate else "")
              + (f"  [{','.join(extras)}]" if extras else ""))

    def _current_critic_syn_coef(self) -> float:
        """Linear warmup of critic_syn_coef from 0 → self.critic_syn_coef over
        self.critic_syn_coef_warmup total_steps. Returns the live coefficient."""
        if self.critic_syn_coef_warmup <= 0:
            return self.critic_syn_coef
        frac = min(1.0, self.total_steps / float(self.critic_syn_coef_warmup))
        return self.critic_syn_coef * frac

    # ──────────────────────────────────────────────────────────────────────
    # update — fully overrides IQLAgent.update with CAPA routing
    # ──────────────────────────────────────────────────────────────────────

    def update(
        self,
        batch:      Dict[str, Tensor],            # mixed real+syn (for actor)
        real_batch: Optional[Dict[str, Tensor]] = None,   # real-only (for V/Q + BC)
        critic_only: bool = False,                # for UTD>1 (RLPD/EDAC-style)
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

        # ── 0.5 CAPA+: pull TRUSTED (low-uncertainty) syn rows for the critic ─
        # Only active when critic_syn_gate=True. gate = exp(-unc_beta·Q_std);
        # high-disagreement (OOD) syn rows get ~0 weight, so the critic only
        # absorbs synthetic transitions the ensemble is confident about.
        syn_for_critic = None
        if self.critic_syn_gate:
            b_obs    = batch["obs"]
            b_action = batch["action"]
            src = batch.get("source", torch.ones(b_obs.shape[0], device=self.device))
            m = src < 0.5
            if bool(m.any()):
                with torch.no_grad():
                    qa   = self.q_tgt.target.all(b_obs[m], b_action[m])   # (M, Bs)
                    gate = torch.exp(-self.unc_beta * qa.std(dim=0))       # (Bs,)
                syn_for_critic = dict(
                    obs=b_obs[m], action=b_action[m],
                    reward=batch["reward"][m], next_obs=batch["next_obs"][m],
                    done=batch["done"][m], gate=gate,
                )
                metrics["train/critic_gate_syn_mean"] = float(gate.mean().item())

        # ── 1. V update on REAL data ───────────────────────────────────────
        self.opt_v.zero_grad(set_to_none=True)
        with autocast(enabled=(self.device.type == "cuda")):
            with torch.no_grad():
                q_tgt = self.q_tgt.target.min(r_obs, r_action)
            v      = self.v(r_obs)
            v_diff = q_tgt - v
            v_loss = self._expectile_loss(v_diff)   # standard expectile, no SA-IQL

            # CAPA+: gated-syn expectile term (trusted syn extends V coverage)
            if syn_for_critic is not None:
                vs = self.v(syn_for_critic["obs"])
                with torch.no_grad():
                    qts = self.q_tgt.target.min(syn_for_critic["obs"],
                                                syn_for_critic["action"])
                dvs = qts - vs
                # NOVELTY: asymmetric expectile by uncertainty. Confident syn
                # (gate≈1) gets the standard upper-expectile τ; uncertain syn
                # (gate≈0) falls back to the median (τ=0.5) — more pessimistic
                # about the syn Q-target where the ensemble disagrees.
                if self.asym_expectile_syn:
                    tau_syn = 0.5 + (self.expectile - 0.5) * syn_for_critic["gate"]
                    ws = torch.where(dvs >= 0, tau_syn, 1.0 - tau_syn)
                else:
                    ws = torch.where(dvs >= 0,
                                     torch.full_like(dvs, self.expectile),
                                     torch.full_like(dvs, 1.0 - self.expectile))
                v_loss_syn = (syn_for_critic["gate"] * ws * dvs.pow(2)).mean()
                coef_now = self._current_critic_syn_coef()
                v_loss = v_loss + coef_now * v_loss_syn
                metrics["loss/v_syn"] = v_loss_syn.item()
                metrics["train/critic_syn_coef_live"] = coef_now

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

            # CAPA+: gated-syn TD term (trusted syn extends Q coverage). Uses the
            # syn reward+next-state — sound now that data is physics-consistent
            # and reward is exact-analytic; gate filters the untrusted rows.
            if syn_for_critic is not None:
                with torch.no_grad():
                    v_next_s = self.v(syn_for_critic["next_obs"])
                    target_s = (syn_for_critic["reward"]
                                + self.discount * (1.0 - syn_for_critic["done"]) * v_next_s)
                all_q_s  = self.q.all(syn_for_critic["obs"], syn_for_critic["action"])  # (M, Bs)
                q_err_s  = (all_q_s - target_s.unsqueeze(0)).pow(2)                      # (M, Bs)
                q_loss_syn = (syn_for_critic["gate"].unsqueeze(0) * q_err_s).mean()
                # Curriculum (NOVELTY): coef is ramped from 0 → target over the
                # warmup window, so syn enters the critic only once the gate signal
                # has stabilized. Same coefficient used for V and Q.
                coef_now_q = self._current_critic_syn_coef()
                q_loss = q_loss + coef_now_q * q_loss_syn
                metrics["loss/q_syn"] = q_loss_syn.item()

        self.scaler_q.scale(q_loss).backward()
        self.scaler_q.unscale_(self.opt_q)
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), 1.0)
        self.scaler_q.step(self.opt_q)
        self.scaler_q.update()
        self.q_tgt.update(self.q)

        metrics["loss/q"]       = q_loss.item()
        metrics["train/q_mean"] = all_q.mean().item()

        # ── 3. Actor AWR on MIXED batch, uncertainty-gated for syn rows ───
        # UTD>1 path: return early on critic-only calls (caller does utd-1 of these
        # then one full update — more critic gradient with same actor learning).
        if critic_only:
            return metrics
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

                # AWR weight. Two routes for combining the gate with the
                # AWR exponent:
                #   "scale"  (legacy)  : w = exp(β·adv) · gate
                #   "temper" (NOVELTY) : w = exp(β·gate·adv) — uncertainty
                #     flattens the AWR weight curve instead of multiplicatively
                #     suppressing it. Avoids the failure mode where a high-
                #     advantage but high-uncertainty syn row dominates after the
                #     exp blows it up and the gate barely dents it.
                if self.awr_gate_mode == "temper":
                    weight = torch.exp(self.temperature * gate * adv_for_weight).clamp(max=100.0)
                else:
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

            # NOVELTY — Generative BC anchor on HIGH-CONFIDENCE synthetic rows.
            # Adds an *offensive* signal-extraction term: when the ensemble agrees
            # that a syn (s, a) is in-distribution (gate >= gbc_gate_min), use it
            # as a BC target on the policy. This captures behavior the diffusion
            # invented that AWR misses because (a) its advantage is small or (b)
            # the weight is dominated by real high-advantage rows.
            if self.gbc_weight > 0.0:
                gbc_mask = (source < 0.5) & (gate >= self.gbc_gate_min)
                n_gbc = int(gbc_mask.sum().item())
                if n_gbc > 0:
                    log_prob_gbc = self.actor.log_prob(obs[gbc_mask], action_for_logp[gbc_mask])
                    # Weight by gate so confidence above the threshold still matters
                    g = gate[gbc_mask]
                    gbc_loss = -(g * log_prob_gbc).sum() / (g.sum() + 1e-8)
                    actor_loss = actor_loss + self.gbc_weight * gbc_loss
                    metrics["loss/gbc"]      = float(gbc_loss.item())
                    metrics["train/n_gbc"]   = float(n_gbc)
                else:
                    metrics["loss/gbc"]      = 0.0
                    metrics["train/n_gbc"]   = 0.0

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
