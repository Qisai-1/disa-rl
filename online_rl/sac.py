"""
Soft Actor-Critic (SAC) for online RL in DiSA-RL.

Key design decisions
--------------------
1. Identical actor/critic architecture to IQL (iql/networks.py)
   Weights transfer directly from offline IQL to online SAC.

2. load_from_iql()
   Transfers actor + critic weights from IQL checkpoint.
   Eliminates cold-start at the offline→online boundary.

3. align_entropy_with_iql()
   Initializes SAC temperature α to match the offline policy entropy.
   Prevents immediate policy destruction by too-high exploration noise.

4. critic_only flag in update()
   First N online steps: update critic only (value recalibration).
   IQL uses expectile regression (conservative). SAC uses Bellman
   (can overestimate). Recalibration lets Q-values settle before
   the actor starts following them.

5. conservative_weight in update()
   Blends behavior cloning penalty into actor loss at transition.
   Decays from 1.0 → 0.0 over first N steps.
   Prevents policy from deviating before Q-values are trustworthy.
"""

from __future__ import annotations
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from typing import Dict, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from iql.networks import GaussianActor, TwinQNetwork, QEnsemble, EMATarget


def create_sac_from_iql(iql_ckpt_path: str,
                        obs_dim: int,
                        action_dim: int,
                        device: torch.device = torch.device("cpu"),
                        actor_hidden_dims: Tuple[int, ...] = (256, 256),
                        q_hidden_dims: Tuple[int, ...] = (256, 256, 256, 256),
                        **sac_kwargs) -> "SACAgent":
    """
    Factory that builds a SACAgent matched to a DRC-IQL checkpoint and
    transfers weights in one step.

    Defaults match our v1/v2 sweep: actor = 2×256, Q = 4×256.
    Pass overrides for non-default geometries.

    Returns
    -------
    sac : SACAgent with actor, twin Q, and target Q all initialized from
          the IQL checkpoint (first two critics of the ensemble are used
          for the twin Q).
    """
    # Build SAC with the Q (wider) dims then patch the actor to match
    sac = SACAgent(obs_dim=obs_dim, action_dim=action_dim,
                    hidden_dims=q_hidden_dims, device=device, **sac_kwargs)
    if actor_hidden_dims != q_hidden_dims:
        sac.actor = GaussianActor(obs_dim, action_dim, actor_hidden_dims).to(device)
        sac.opt_actor = torch.optim.Adam(sac.actor.parameters(),
                                          lr=sac.opt_actor.param_groups[0]["lr"])
    sac.load_from_iql(iql_ckpt_path)
    return sac


class SACAgent:
    """
    SAC with offline-to-online transition support.

    Parameters
    ----------
    obs_dim, action_dim : environment dimensions
    hidden_dims         : must match IQL hidden_dims for weight transfer
    lr_actor/critic/alpha: per-network learning rates
    gamma               : discount
    tau                 : EMA rate for target critic
    target_entropy      : desired entropy (default = -action_dim)
    init_temperature    : initial α (overridden by align_entropy_with_iql)
    device              : torch device
    """

    def __init__(
        self,
        obs_dim:          int,
        action_dim:       int,
        hidden_dims:      Tuple[int, ...] = (256, 256),
        lr_actor:         float = 3e-4,
        lr_critic:        float = 3e-4,
        lr_alpha:         float = 3e-4,
        gamma:            float = 0.99,
        tau:              float = 0.005,
        target_entropy:   Optional[float] = None,
        init_temperature: float = 0.2,
        device:           torch.device = torch.device("cpu"),
        num_critics:      int = 2,           # 2 = TwinQ, >2 = QEnsemble (RLPD)
        critic_subset:    int = 2,           # used only when num_critics > 2
    ):
        self.obs_dim     = obs_dim
        self.action_dim  = action_dim
        self.gamma       = gamma
        self.device      = device
        self.total_steps = 0
        self.num_critics = num_critics

        # Same architecture as IQL — weights transfer directly
        self.actor = GaussianActor(obs_dim, action_dim, hidden_dims).to(device)
        if num_critics <= 2:
            self.critic = TwinQNetwork(obs_dim, action_dim, hidden_dims).to(device)
        else:
            self.critic = QEnsemble(obs_dim, action_dim, hidden_dims,
                                    num_critics=num_critics,
                                    subset_size=critic_subset).to(device)
        self.critic_target  = EMATarget(self.critic, tau=tau).to(device)

        # Learnable entropy temperature
        self.target_entropy = target_entropy if target_entropy else -float(action_dim)
        self.log_alpha = torch.tensor(
            np.log(init_temperature), dtype=torch.float32,
            device=device, requires_grad=True
        )

        self.opt_actor  = torch.optim.Adam(self.actor.parameters(),  lr=lr_actor)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.opt_alpha  = torch.optim.Adam([self.log_alpha],         lr=lr_alpha)

        self.scaler_actor  = GradScaler()
        self.scaler_critic = GradScaler()

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    # ──────────────────────────────────────────────────────────────────
    # Offline → Online connection
    # ──────────────────────────────────────────────────────────────────

    def load_from_iql(self, iql_ckpt_path: str) -> None:
        """
        Transfer actor and critic weights from an offline IQL checkpoint.
        Both networks start from the offline policy — no cold start.

        Handles both architectures:
          • Legacy TwinQ checkpoints (state_dict keys "q1.*", "q2.*")
          • New DRC-IQL QEnsemble checkpoints (keys "qs.0.*", "qs.1.*", ...)
        For QEnsemble, we pick the FIRST TWO critics into SAC's twin Q.
        """
        ckpt = torch.load(iql_ckpt_path, map_location=self.device,
                          weights_only=False)

        # Actor: direct copy — identical architecture
        self.actor.load_state_dict(ckpt["actor"])

        # Critic: detect whether the saved q is TwinQ or QEnsemble
        q_sd = ckpt["q"]
        tq_sd = ckpt.get("q_tgt", q_sd)

        if any(k.startswith("qs.") for k in q_sd):
            # Source = QEnsemble of M_src critics. Three cases:
            #   1. SAC critic is also QEnsemble — copy as many as match.
            #   2. SAC critic is TwinQ — take first two ensemble members.
            #   3. Sizes mismatch — partial load + warn.
            def extract(prefix: int, sd: dict) -> dict:
                p = f"qs.{prefix}."
                return {k[len(p):]: v for k, v in sd.items() if k.startswith(p)}

            n_src = len({int(k.split(".")[1]) for k in q_sd if k.startswith("qs.")})

            if isinstance(self.critic, QEnsemble):
                n_copy = min(n_src, self.critic.num_critics)
                for i in range(n_copy):
                    self.critic.qs[i].load_state_dict(extract(i, q_sd))
                    self.critic_target.target.qs[i].load_state_dict(extract(i, tq_sd))
                print(f"Loaded QEnsemble[0..{n_copy-1}] / src={n_src} → "
                      f"SAC QEnsemble (size {self.critic.num_critics}) | {iql_ckpt_path}")
                if n_copy < self.critic.num_critics:
                    print(f"  NB: {self.critic.num_critics - n_copy} extra critics "
                          f"randomly initialized (diversity bonus).")
            else:
                # Legacy: SAC is TwinQ → take first two
                try:
                    self.critic.q1.load_state_dict(extract(0, q_sd))
                    self.critic.q2.load_state_dict(extract(1, q_sd))
                    self.critic_target.target.q1.load_state_dict(extract(0, tq_sd))
                    self.critic_target.target.q2.load_state_dict(extract(1, tq_sd))
                    print(f"Loaded QEnsemble[0,1] → SAC TwinQ | {iql_ckpt_path}")
                except RuntimeError as e:
                    print(f"WARN: ensemble→twin shape mismatch. Cause: {e!s:.140s}")
        else:
            # Legacy TwinQ source → only works into TwinQ SAC
            if isinstance(self.critic, QEnsemble):
                # Replicate the twin into the first two ensemble members,
                # leave the rest random for diversity.
                self.critic.qs[0].load_state_dict(
                    {k.replace("q1.", ""): v for k, v in q_sd.items() if k.startswith("q1.")})
                self.critic.qs[1].load_state_dict(
                    {k.replace("q2.", ""): v for k, v in q_sd.items() if k.startswith("q2.")})
                self.critic_target.target.qs[0].load_state_dict(
                    {k.replace("q1.", ""): v for k, v in tq_sd.items() if k.startswith("q1.")})
                self.critic_target.target.qs[1].load_state_dict(
                    {k.replace("q2.", ""): v for k, v in tq_sd.items() if k.startswith("q2.")})
                print(f"Loaded TwinQ → SAC QEnsemble[0,1] (rest random) | {iql_ckpt_path}")
            else:
                self.critic.q1.load_state_dict(q_sd)
                self.critic.q2.load_state_dict(q_sd)
                self.critic_target.target.q1.load_state_dict(tq_sd)
                self.critic_target.target.q2.load_state_dict(tq_sd)
                print(f"Loaded IQL TwinQ → SAC TwinQ | {iql_ckpt_path}")

    def align_entropy_with_iql(
        self,
        sample_obs: torch.Tensor,
        n_samples:  int = 1024,
    ) -> float:
        """
        Set SAC temperature α to match the offline policy's natural entropy.

        Without this, SAC starts with default α=0.2 which may be completely
        wrong for this policy — too high causes immediate performance collapse,
        too low causes premature exploitation.

        Returns the computed initial α value.
        """
        self.actor.eval()
        with torch.no_grad():
            obs = sample_obs[:n_samples].to(self.device)
            _, log_prob = self.actor.get_action(obs)
            entropy = -log_prob.mean().item()   # E[-log π(a|s)]

        # Set α so entropy term starts at the right scale
        if entropy > 0:
            init_alpha = float(np.clip(
                abs(self.target_entropy) / (entropy + 1e-8), 0.01, 1.0
            ))
        else:
            init_alpha = 0.2

        with torch.no_grad():
            self.log_alpha.fill_(np.log(init_alpha))

        self.actor.train()
        print(f"Entropy aligned  |  "
              f"policy_entropy={entropy:.3f}  "
              f"target={self.target_entropy:.3f}  "
              f"init_alpha={init_alpha:.4f}")
        return init_alpha

    # ──────────────────────────────────────────────────────────────────
    # Training step
    # ──────────────────────────────────────────────────────────────────

    def update(
        self,
        batch:               Dict[str, torch.Tensor],
        critic_only:         bool  = False,
        conservative_weight: float = 0.0,
    ) -> Dict[str, float]:
        """
        One SAC gradient step.

        Parameters
        ----------
        batch               : {obs, action, reward, next_obs, done}
        critic_only         : skip actor update (value recalibration phase)
        conservative_weight : [0,1] blends BC penalty into actor loss
                              1.0 = pure BC, 0.0 = pure SAC
                              Decay this from 1.0→0.0 over first N steps
        """
        obs      = batch["obs"]
        action   = batch["action"]
        reward   = batch["reward"]
        next_obs = batch["next_obs"]
        done     = batch["done"]
        metrics  = {}

        # ── Critic update ─────────────────────────────────────────────
        # Use .all() / .min() API so the same code works for TwinQ
        # (M=2) and QEnsemble (M=num_critics, RLPD-style).
        self.opt_critic.zero_grad(set_to_none=True)
        with torch.no_grad():
            next_a, next_lp = self.actor.get_action(next_obs)
            # min over (possibly random) subset — REDQ-style for QEnsemble,
            # plain min(Q1,Q2) for TwinQ
            q_next   = self.critic_target.target.min(next_obs, next_a) - self.alpha * next_lp
            q_target = reward + self.gamma * (1.0 - done) * q_next

        all_q = self.critic.all(obs, action)                # (M, B)
        # Each critic gets its own TD loss; ensemble grad updates them all.
        critic_loss = F.mse_loss(all_q, q_target.unsqueeze(0).expand_as(all_q))

        self.scaler_critic.scale(critic_loss).backward()
        self.scaler_critic.unscale_(self.opt_critic)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.scaler_critic.step(self.opt_critic)
        self.scaler_critic.update()
        self.critic_target.update(self.critic)

        metrics["online/critic_loss"] = critic_loss.item()
        metrics["online/q_mean"]      = all_q.mean().item()
        metrics["online/q_target"]    = q_target.mean().item()
        metrics["online/q_std"]       = all_q.std(dim=0).mean().item()

        if critic_only:
            self.total_steps += 1
            return metrics

        # ── Actor update ───────────────────────────────────────────────
        self.opt_actor.zero_grad(set_to_none=True)
        new_a, log_prob = self.actor.get_action(obs)
        # Twin-style min for the actor target (RLPD uses subset-min too).
        q_pi = self.critic.min(obs, new_a)

        # SAC actor loss: maximize E[Q(s,a) - α log π(a|s)]
        sac_loss = (self.alpha.detach() * log_prob - q_pi).mean()

        # Conservative blend: BC penalty decays over transition window
        if conservative_weight > 0.0:
            _, offline_lp = self.actor.get_action(obs)
            bc_loss    = -offline_lp.mean()
            actor_loss = ((1.0 - conservative_weight) * sac_loss +
                           conservative_weight * bc_loss)
            metrics["online/bc_loss"]            = bc_loss.item()
            metrics["online/conservative_weight"] = conservative_weight
        else:
            actor_loss = sac_loss

        self.scaler_actor.scale(actor_loss).backward()
        self.scaler_actor.unscale_(self.opt_actor)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.scaler_actor.step(self.opt_actor)
        self.scaler_actor.update()

        metrics["online/actor_loss"] = actor_loss.item()
        metrics["online/entropy"]    = -log_prob.mean().item()

        # ── Alpha update ───────────────────────────────────────────────
        self.opt_alpha.zero_grad()
        alpha_loss = -(self.log_alpha *
                       (log_prob + self.target_entropy).detach()).mean()
        alpha_loss.backward()
        self.opt_alpha.step()

        metrics["online/alpha"]      = self.alpha.item()
        metrics["online/alpha_loss"] = alpha_loss.item()

        self.total_steps += 1
        return metrics

    # ──────────────────────────────────────────────────────────────────
    # Action selection
    # ──────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select action for env interaction. obs: (obs_dim,) numpy."""
        obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        return self.actor.act(obs_t, deterministic).squeeze(0).cpu().numpy()

    # ──────────────────────────────────────────────────────────────────
    # Checkpoint
    # ──────────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "actor":         self.actor.state_dict(),
            "critic":        self.critic.state_dict(),
            "critic_target": self.critic_target.target.state_dict(),
            "log_alpha":     self.log_alpha.data,
            "opt_actor":     self.opt_actor.state_dict(),
            "opt_critic":    self.opt_critic.state_dict(),
            "opt_alpha":     self.opt_alpha.state_dict(),
            "total_steps":   self.total_steps,
        }, path)
        print(f"SAC saved → {path}")

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.target.load_state_dict(ckpt["critic_target"])
        self.log_alpha.data = ckpt["log_alpha"]
        self.opt_actor.load_state_dict(ckpt["opt_actor"])
        self.opt_critic.load_state_dict(ckpt["opt_critic"])
        self.opt_alpha.load_state_dict(ckpt["opt_alpha"])
        self.total_steps = ckpt["total_steps"]
        print(f"SAC loaded from {path}  (step {self.total_steps:,})")
