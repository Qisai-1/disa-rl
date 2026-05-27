"""
Q-Conditional Trajectory Diffusion (QCD) — Pillar 2 of DiSA-RL.

Replaces the standard return-conditioning of the trajectory diffusion model
with conditioning on the **offline IQL critic's value estimate**.

Standard (return-conditioned, what SynthER/GTA/GODA/Decision-Diffuser do):
    cond = [s_0_norm | (R − μ_R) / σ_R]      where R = sum of rewards in sub-traj

QCD (ours):
    cond = [s_0_norm | (Q* − μ_Q) / σ_Q]     where Q* = Q_φ(s_0, a_0)
                                              from a pretrained IQL critic.

Why this is novel:
  * Return is **raw** — same value can correspond to very different state
    distributions, and is bounded only by the dataset's empirical max.
  * Q* is the **learned** value — it reflects how the critic has assessed
    the (s, a) pair, which already includes pessimism on OOD actions.
  * Conditioning on Q* therefore pushes generation toward in-support,
    high-value regions in a single step. No prior diffusion-augmented
    offline RL paper does this.

At inference time, target_Q is sampled from the top quantile of the
training Q-distribution, giving us a principled way to ask the generator
for "high-value, in-support" trajectories.
"""

from __future__ import annotations
import os
import sys
from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor


def compute_q_targets_from_critic(
    iql_ckpt:    str,
    obs:         np.ndarray,        # (N, obs_dim) — initial states of each sub-trajectory
    actions:     np.ndarray,        # (N, action_dim) — initial actions
    device:      torch.device,
    use_v:       bool = False,      # True → use V(s_0) instead of Q(s_0, a_0)
    batch_size:  int  = 4096,
) -> np.ndarray:
    """
    Load a pretrained IQL agent and compute the Q-target (or V-target)
    for every sub-trajectory's starting (s_0, a_0).

    Returns a (N,) numpy array of float32 values.
    """
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    from iql.agent import IQLAgent

    obs_dim    = obs.shape[1]
    action_dim = actions.shape[1]

    if not os.path.exists(iql_ckpt):
        raise FileNotFoundError(
            f"IQL critic checkpoint not found: {iql_ckpt}\n"
            f"Train one first, e.g. with offline_only mode."
        )

    print(f"  QCD: loading IQL critic from {iql_ckpt}")
    agent = IQLAgent(obs_dim=obs_dim, action_dim=action_dim, device=device)
    # Try to load with default (TwinQ) first; if mismatch, agent.load will
    # fall back to actor-only mode — but we *need* Q/V here so we re-try
    # with num_critics=10 if the strict load fails.
    try:
        agent.load(iql_ckpt)
    except Exception:
        agent = IQLAgent(obs_dim=obs_dim, action_dim=action_dim,
                          q_hidden_dims=(256, 256, 256, 256),
                          num_critics=10, device=device)
        agent.load(iql_ckpt)

    agent.q.eval(); agent.v.eval()
    targets = np.empty(len(obs), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(obs), batch_size):
            e = min(s + batch_size, len(obs))
            ob = torch.from_numpy(obs[s:e]).float().to(device)
            if use_v:
                targets[s:e] = agent.v(ob).cpu().numpy()
            else:
                ac = torch.from_numpy(actions[s:e]).float().to(device)
                # min of the twin/ensemble — what IQL uses as the target
                targets[s:e] = agent.q.min(ob, ac).cpu().numpy()

    label = "V(s_0)" if use_v else "Q(s_0, a_0)"
    print(f"  QCD: {label} stats — "
          f"mean={targets.mean():.3f}  std={targets.std():.3f}  "
          f"min={targets.min():.3f}  max={targets.max():.3f}")
    return targets


def sample_q_target(
    q_distribution: np.ndarray,
    quantile:       float = 0.9,
    batch_size:     int   = 64,
    rng:            Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Sample target Q-values for generation: draw from the top-quantile of
    the training Q-distribution. This is the conditioning analog of
    "give me a high-value but realistic trajectory".

    Returns a (batch_size,) numpy array of float32 values.
    """
    rng = rng or np.random.default_rng()
    pool = q_distribution[q_distribution >= np.quantile(q_distribution, quantile)]
    return rng.choice(pool, size=batch_size).astype(np.float32)


class QConditioning:
    """
    Stateful conditioning helper: stores Q-targets per sub-trajectory plus
    the (mean, std) statistics used to z-score them, mirroring how
    DataNormalizer stores return_mean / return_std.

    Plumbed into TrajectoryDataset so the existing condition vector path
    [s_0_norm | scalar_norm] works unchanged — we just swap the scalar.
    """

    def __init__(self, q_targets: np.ndarray):
        self.q_targets = q_targets.astype(np.float32)
        self.mean      = float(self.q_targets.mean())
        self.std       = float(self.q_targets.std() + 1e-8)

    def normalize(self, q: np.ndarray) -> np.ndarray:
        return (q - self.mean) / self.std

    def denormalize(self, qn: np.ndarray) -> np.ndarray:
        return qn * self.std + self.mean

    def __len__(self) -> int:
        return len(self.q_targets)

    def __getitem__(self, idx) -> float:
        return float(self.q_targets[idx])

    def as_dict(self) -> dict:
        return {"q_mean": np.array([self.mean]), "q_std": np.array([self.std])}

    @classmethod
    def from_dict(cls, d: dict, q_targets: Optional[np.ndarray] = None):
        # Reconstruct without storing targets (for inference-only loading)
        obj = cls.__new__(cls)
        obj.q_targets = q_targets if q_targets is not None else np.zeros(0, dtype=np.float32)
        obj.mean = float(np.array(d["q_mean"]).flatten()[0])
        obj.std  = float(np.array(d["q_std"]).flatten()[0])
        return obj
