"""
Trajectory generation interface for DiSA-RL.

Usage in the offline RL augmentation loop:
    gen    = TrajectoryGenerator.from_checkpoint("checkpoints/offline_final.pt", device)
    result = gen.generate(n_trajectories=512, target_return=5000.)
    # result["transitions"] is a list of (s, a, r, s', done) dicts

Usage in the online fine-tuning loop:
    loss   = gen.finetune_step(new_real_trajs)
    eroll  = gen.estimate_eroll(new_real_trajs)
    rho_max = eta * (1 - gamma) / eroll
"""

from __future__ import annotations
import sys, os as _os
_d = _os.path.dirname(_os.path.abspath(__file__))
if _d not in sys.path: sys.path.insert(0, _d)
if _os.path.dirname(_d) not in sys.path: sys.path.insert(0, _os.path.dirname(_d))
import numpy as np
import torch
from torch.cuda.amp import GradScaler
from dataclasses import dataclass
from typing import Dict, List, Optional

from model import TrajectoryDiT
from flow_matching import ConditionalFlowMatching
from data import DataNormalizer
from config import LossConfig


# ──────────────────────────────────────────────────────────────────────────────
# Generation config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class GenerationConfig:
    nfe:          int   = 20     # Heun steps  (cost ≈ 2 × nfe forward passes)
    cfg_scale:    float = 1.5    # Guidance strength  (1.0 = no guidance)
    clip_actions: bool  = True   # Hard-clip generated actions to [−1, 1]


# ──────────────────────────────────────────────────────────────────────────────
# Generator
# ──────────────────────────────────────────────────────────────────────────────

class TrajectoryGenerator:
    """
    High-level interface for generating synthetic RL trajectories and
    fine-tuning the diffusion model online with EWC protection.
    """

    def __init__(
        self,
        model:      TrajectoryDiT,
        cfm:        ConditionalFlowMatching,
        normalizer: DataNormalizer,
        device:     torch.device,
    ):
        self.model      = model
        self.cfm        = cfm
        self.normalizer = normalizer
        self.device     = device
        self.model.eval()

        self._ft_optimizer: Optional[torch.optim.Optimizer] = None
        self._ft_scaler = GradScaler()
        self._ewc       = None    # set via attach_ewc()

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(cls, path: str, device: torch.device) -> "TrajectoryGenerator":
        """Load generator from a training checkpoint.  Uses EMA weights."""
        ckpt = torch.load(path, map_location=device, weights_only=False)

        model = TrajectoryDiT(**ckpt["model_config"]).to(device)
        # Use EMA weights for generation (smoother, better quality)
        model.load_state_dict(ckpt["ema_state_dict"])
        model.eval()

        normalizer = DataNormalizer.from_dict(ckpt["normalizer"])
        cfm        = ConditionalFlowMatching(model, device)

        print(f"Generator loaded  |  step={ckpt['step']:,}  |  {path}")
        return cls(model, cfm, normalizer, device)

    def attach_ewc(self, ewc) -> None:
        """Attach an EWC instance for use during online fine-tuning."""
        self._ewc = ewc
        print("EWC attached to generator.")

    # ------------------------------------------------------------------
    # Condition helpers
    # ------------------------------------------------------------------

    def _build_cond(
        self,
        initial_states: np.ndarray,   # (B, obs_dim) unnormalised
        target_return:  float,
    ) -> torch.Tensor:
        norm_s0  = self.normalizer.obs.normalize(initial_states).astype(np.float32)
        norm_ret = np.full(
            (len(initial_states), 1),
            self.normalizer.normalize_return(target_return),
            dtype=np.float32,
        )
        cond = np.concatenate([norm_s0, norm_ret], axis=1)
        return torch.from_numpy(cond).to(self.device)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        n_trajectories: int                        = 64,
        initial_states: Optional[np.ndarray]       = None,
        target_return:  float                      = 3000.0,
        gen_cfg:        Optional[GenerationConfig] = None,
    ) -> Dict:
        """
        Generate a batch of denormalised trajectories.

        Parameters
        ----------
        n_trajectories : number of trajectories to generate
        initial_states : (B, obs_dim) or (obs_dim,) unnormalised observations.
                         None → use dataset mean (zeros in normalised space).
        target_return  : desired cumulative return for CFG conditioning.
                         Higher values bias generation toward better behaviour.
        gen_cfg        : NFE, guidance scale, action clipping.

        Returns
        -------
        dict with:
            observations : (B, T, obs_dim)    unnormalised
            actions      : (B, T, action_dim) unnormalised, clipped to [−1, 1]
            rewards      : (B, T)             unnormalised
            transitions  : list of dicts {obs, action, reward, next_obs, done}
        """
        if gen_cfg is None:
            gen_cfg = GenerationConfig()

        obs_dim    = self.model.obs_dim
        action_dim = self.model.action_dim

        # Build initial states
        if initial_states is None:
            initial_states = np.zeros((n_trajectories, obs_dim), dtype=np.float32)
        elif initial_states.ndim == 1:
            initial_states = np.tile(initial_states[None], (n_trajectories, 1))
        elif len(initial_states) < n_trajectories:
            reps           = int(np.ceil(n_trajectories / len(initial_states)))
            initial_states = np.tile(initial_states, (reps, 1))[:n_trajectories]

        cond = self._build_cond(initial_states, target_return)  # (B, cond_dim)

        # Generate in normalised space
        trajs_norm = self.cfm.heun_sample(
            batch_size = n_trajectories,
            cond       = cond,
            nfe        = gen_cfg.nfe,
            cfg_scale  = gen_cfg.cfg_scale,
        ).cpu().numpy()   # (B, T, D)

        # Denormalise
        trajs_raw = self.normalizer.denormalize_batch(trajs_norm, obs_dim, action_dim)

        obs     = trajs_raw[:, :, :obs_dim]
        actions = trajs_raw[:, :, obs_dim:obs_dim + action_dim]
        rewards = trajs_raw[:, :, -1]

        if gen_cfg.clip_actions:
            actions = np.clip(actions, -1.0, 1.0)

        return {
            "observations": obs,
            "actions":      actions,
            "rewards":      rewards,
            "transitions":  self._to_transitions(obs, actions, rewards),
        }

    # ------------------------------------------------------------------
    # Replay buffer formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _to_transitions(
        obs:     np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
    ) -> List[Dict]:
        """Flatten (B, T, *) arrays to a list of (s, a, r, s', done) dicts."""
        B, T, _ = obs.shape
        out = []
        for b in range(B):
            for t in range(T):
                done     = (t == T - 1)
                next_obs = obs[b, t] if done else obs[b, t + 1]
                out.append({
                    "obs":      obs[b, t],
                    "action":   actions[b, t],
                    "reward":   float(rewards[b, t]),
                    "next_obs": next_obs,
                    "done":     done,
                })
        return out

    # ------------------------------------------------------------------
    # Online fine-tuning
    # ------------------------------------------------------------------

    def finetune_step(
        self,
        real_trajs: np.ndarray,   # (B, T, D) unnormalised real trajectories
        lr:         float = 1e-5,
    ) -> Dict[str, float]:
        """
        One gradient step to adapt the diffusion model to new real data.

        Implements the "periodic re-train pψ" step from Algorithm 1.
        If EWC is attached, adds the forgetting-prevention penalty.

        Parameters
        ----------
        real_trajs : (B, T, D) unnormalised (obs, action, reward) from the env
        lr         : fine-tuning learning rate (typically 10× lower than pretraining)

        Returns
        -------
        dict of loss components for logging
        """
        self.model.train()
        obs_dim    = self.model.obs_dim
        action_dim = self.model.action_dim

        # Normalise
        norm = self.normalizer.normalize_batch(real_trajs, obs_dim, action_dim)
        x1   = torch.from_numpy(norm).float().to(self.device)

        # Build condition: s_0 normalised + actual return normalised
        s0       = real_trajs[:, 0, :obs_dim]
        ret      = real_trajs[:, :, -1].sum(axis=1)
        norm_ret = ((ret - self.normalizer.return_mean) /
                    (self.normalizer.return_std + 1e-8)).astype(np.float32)
        norm_s0  = self.normalizer.obs.normalize(s0).astype(np.float32)
        cond     = torch.from_numpy(
            np.concatenate([norm_s0, norm_ret[:, None]], axis=1)
        ).to(self.device)

        # Lazy-init optimizer
        if self._ft_optimizer is None:
            self._ft_optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=lr, weight_decay=1e-5
            )
        for pg in self._ft_optimizer.param_groups:
            pg["lr"] = lr

        self._ft_optimizer.zero_grad(set_to_none=True)
        fm_loss, metrics = self.cfm.loss(x1, cond)

        total_loss = fm_loss
        if self._ewc is not None:
            ewc_loss             = self._ewc.penalty(self.model)
            total_loss           = fm_loss + ewc_loss
            metrics["ewc_loss"]  = ewc_loss.item()

        self._ft_scaler.scale(total_loss).backward()
        self._ft_scaler.unscale_(self._ft_optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self._ft_scaler.step(self._ft_optimizer)
        self._ft_scaler.update()

        self.model.eval()
        return metrics

    # ------------------------------------------------------------------
    # εroll estimation  (used for adaptive ρ schedule)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def estimate_eroll(
        self,
        real_trajs: np.ndarray,   # (B, T, D) unnormalised
        n_samples:  int = 64,
    ) -> float:
        """
        Estimate εroll = TV(P̃_{π,ψ}, P_π).

        We use MMD (Maximum Mean Discrepancy) with an RBF kernel on the
        (obs, action) distribution as a proxy for TV distance.
        MMD is computationally cheap, statistically principled, and avoids
        the need for a density estimator.

        A value near 0 means the diffusion model closely matches the real
        distribution.  The adaptive ρ schedule uses:
            ρ_max = η · (1 − γ) / εroll

        Parameters
        ----------
        real_trajs : recently collected real trajectories (unnormalised)
        n_samples  : number of synthetic samples to generate for comparison

        Returns
        -------
        eroll_approx : float in [0, ∞)
        """
        obs_dim    = self.model.obs_dim
        action_dim = self.model.action_dim

        n = min(n_samples, len(real_trajs))
        s0 = real_trajs[:n, 0, :obs_dim]

        # Generate synthetic trajectories seeded from real starting states
        avg_return = float(real_trajs[:n, :, -1].sum(axis=1).mean())
        gen = self.generate(
            n_trajectories = n,
            initial_states = s0,
            target_return  = avg_return,
            gen_cfg        = GenerationConfig(nfe=10, cfg_scale=1.0),
        )

        # Flatten to (n*T, obs_dim + action_dim) for MMD
        real_flat = np.concatenate([
            real_trajs[:n, :, :obs_dim + action_dim].reshape(-1, obs_dim + action_dim)
        ], axis=0)
        gen_flat = np.concatenate([
            gen["observations"].reshape(-1, obs_dim),
            gen["actions"].reshape(-1, action_dim),
        ], axis=1)

        # Subsample for speed
        max_pts = 2000
        if len(real_flat) > max_pts:
            idx       = np.random.choice(len(real_flat), max_pts, replace=False)
            real_flat = real_flat[idx]
            gen_flat  = gen_flat[idx]

        eroll = self._mmd(real_flat, gen_flat)
        return float(eroll)

    @staticmethod
    def _mmd(x: np.ndarray, y: np.ndarray, bandwidth: float = 1.0) -> float:
        """
        Unbiased MMD² estimate with RBF kernel.

        MMD²(P, Q) = E[k(x,x')] − 2·E[k(x,y)] + E[k(y,y')]
        where k(a, b) = exp(−||a − b||² / (2 · bw²))
        """
        def rbf(a, b):
            diff = a[:, None] - b[None, :]          # (n, m, D)
            sq   = (diff ** 2).sum(axis=-1)          # (n, m)
            return np.exp(-sq / (2.0 * bandwidth ** 2))

        n, m = len(x), len(y)
        kxx  = rbf(x, x)
        kyy  = rbf(y, y)
        kxy  = rbf(x, y)

        # Unbiased: exclude diagonal for kxx and kyy
        np.fill_diagonal(kxx, 0.0)
        np.fill_diagonal(kyy, 0.0)

        mmd2 = (kxx.sum() / (n * (n-1))
              - 2 * kxy.mean()
              + kyy.sum() / (m * (m-1)))
        return float(max(mmd2, 0.0) ** 0.5)   # return MMD (not MMD²)


# ──────────────────────────────────────────────────────────────────────────────
# Quick generation test (no real checkpoint needed)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = "./checkpoints/halfcheetah/offline_final.pt"

    if not os.path.exists(ckpt_path):
        print(f"No checkpoint at {ckpt_path} — run train.py first.")
    else:
        gen = TrajectoryGenerator.from_checkpoint(ckpt_path, device)

        print("\nGenerating 16 trajectories …")
        result = gen.generate(n_trajectories=16, target_return=5_000.0)

        obs, act, rew = result["observations"], result["actions"], result["rewards"]
        print(f"  obs     : {obs.shape}  mean={obs.mean():.3f}  std={obs.std():.3f}")
        print(f"  actions : {act.shape}  range=[{act.min():.3f}, {act.max():.3f}]")
        print(f"  rewards : mean={rew.mean():.3f}  cumulative={rew.sum(1).mean():.1f}")
        print(f"  transitions: {len(result['transitions']):,}")