"""
v2 data pipeline — includes REWARD as a generated channel.

Differences vs data.py (v1):
  - TrajectoryDataset keeps the reward column in the trajectory tensor
    (v1 explicitly dropped it at line 331 of data.py).
  - DataNormalizer methods extended: normalize_batch_with_reward and
    denormalize_batch_with_reward handle all three channels (obs, action,
    reward), each with per-channel z-score normalization. The reward stats
    were ALREADY tracked in v1's DataNormalizer — they just weren't being
    applied to the trajectory tensor.
  - TrajectoryDataset supports `train_noise` for Gaussian data augmentation
    on (obs, action, reward) during training.

Design rationale (decided 2026-05-26):
  - Per-channel z-score normalization fixes the "reward scale dominates obs
    scale" problem that originally motivated keeping reward out of diffusion.
  - The existing DataNormalizer infrastructure already had reward stats —
    this file just USES them.
  - Train noise σ=0.01 is a standard regularizer for small datasets
    (medium-replay envs have only ~100-200k transitions vs medium's 1M).
"""

from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple

# Reuse the v1 building blocks
from data import (
    DataNormalizer,
    ChannelStats,
    load_npz,
    split_to_episodes,
    make_subtrajectories,
)


# ──────────────────────────────────────────────────────────────────────────────
# Normalizer — patch DataNormalizer with reward-aware batch methods
# ──────────────────────────────────────────────────────────────────────────────

def _normalize_batch_with_reward(self, trajs: np.ndarray, obs_dim: int, action_dim: int) -> np.ndarray:
    """Normalize (obs, action, reward) trajectories — all 3 channels."""
    out = trajs.copy().astype(np.float32)
    out[..., :obs_dim] = self.obs.normalize(trajs[..., :obs_dim])
    out[..., obs_dim:obs_dim + action_dim] = self.action.normalize(trajs[..., obs_dim:obs_dim + action_dim])
    # Reward is the last channel; stats were stored as (1,)-shape arrays in v1
    r_real = trajs[..., obs_dim + action_dim:obs_dim + action_dim + 1]
    out[..., obs_dim + action_dim:obs_dim + action_dim + 1] = self.reward.normalize(r_real)
    return out


def _denormalize_batch_with_reward(self, trajs: np.ndarray, obs_dim: int, action_dim: int) -> np.ndarray:
    """Denormalize (obs, action, reward) trajectories — all 3 channels."""
    out = trajs.copy().astype(np.float32)
    out[..., :obs_dim] = self.obs.denormalize(trajs[..., :obs_dim])
    out[..., obs_dim:obs_dim + action_dim] = self.action.denormalize(trajs[..., obs_dim:obs_dim + action_dim])
    r_norm = trajs[..., obs_dim + action_dim:obs_dim + action_dim + 1]
    out[..., obs_dim + action_dim:obs_dim + action_dim + 1] = self.reward.denormalize(r_norm)
    return out


# Monkey-patch DataNormalizer so existing instances + new ones gain the methods.
# (Pure additions — does not change v1 behaviour.)
DataNormalizer.normalize_batch_with_reward   = _normalize_batch_with_reward
DataNormalizer.denormalize_batch_with_reward = _denormalize_batch_with_reward


# ──────────────────────────────────────────────────────────────────────────────
# v2 Dataset
# ──────────────────────────────────────────────────────────────────────────────

class TrajectoryDatasetV2(Dataset):
    """
    v2 dataset — keeps reward channel; optional Gaussian noise data aug.

    Each sample is a dict:
        'trajectory' : (T, obs_dim + action_dim + 1)   normalised (obs, action, reward)
        'condition'  : (cond_dim,)                     [norm_s0 | norm_return]

    Args:
        train_noise : if > 0, adds Gaussian noise N(0, train_noise²) to the
                      normalised trajectory at sampling time. Standard
                      regularizer for small datasets. Use 0.0 to disable
                      (e.g. for validation set).
    """

    def __init__(
        self,
        trajs:           np.ndarray,         # (N, T, obs_dim+action_dim+1)  — INCLUDES reward column
        returns:         np.ndarray,         # (N,) — sub-traj returns
        normalizer:      DataNormalizer,
        obs_dim:         int  = 17,
        action_dim:      int  = 6,
        use_return_cond: bool = True,
        train_noise:     float = 0.0,
        q_targets:       Optional[np.ndarray] = None,
    ):
        self.obs_dim         = obs_dim
        self.action_dim      = action_dim
        self.train_noise     = float(train_noise)
        self.normalizer      = normalizer
        self.use_return_cond = use_return_cond
        self.cond_kind       = getattr(normalizer, "cond_kind", "return")

        # Keep ALL 3 channels (obs, action, reward) and normalize each
        assert trajs.shape[-1] >= obs_dim + action_dim + 1, \
            f"Expected obs+action+reward columns, got shape {trajs.shape}"
        trajs_oar = trajs[..., :obs_dim + action_dim + 1]
        self.trajs = normalizer.normalize_batch_with_reward(trajs_oar, obs_dim, action_dim)

        if self.cond_kind == "q":
            if q_targets is None:
                raise ValueError("cond_kind='q' but q_targets=None.")
            self.norm_scalars = (
                (q_targets - normalizer.q_mean) / (normalizer.q_std + 1e-8)
            ).astype(np.float32)
            self.raw_scalars = q_targets.astype(np.float32)
        else:
            self.norm_scalars = (
                (returns - normalizer.return_mean) / (normalizer.return_std + 1e-8)
            ).astype(np.float32)
            self.raw_scalars = returns.astype(np.float32)

        # Legacy aliases (other code reads these)
        self.norm_returns = self.norm_scalars
        self.raw_returns  = self.raw_scalars

    def __len__(self) -> int:
        return len(self.trajs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        traj = torch.from_numpy(self.trajs[idx])              # (T, obs_dim+action_dim+1)
        # Data augmentation: small Gaussian noise on normalised trajectory
        if self.train_noise > 0:
            traj = traj + torch.randn_like(traj) * self.train_noise
        s0   = traj[0, :self.obs_dim]                         # (obs_dim,)
        if self.use_return_cond:
            scalar = torch.tensor([self.norm_scalars[idx]])
            cond   = torch.cat([s0, scalar], dim=0)
        else:
            cond = s0
        return {"trajectory": traj, "condition": cond}


# ──────────────────────────────────────────────────────────────────────────────
# Top-level builder
# ──────────────────────────────────────────────────────────────────────────────

def build_datasets_v2(
    dataset_name:      str           = "halfcheetah-medium-v2",
    data_path:         Optional[str] = None,
    trajectory_length: int           = 100,
    stride:            int           = 50,
    val_fraction:      float         = 0.05,
    obs_dim:           int           = 17,
    action_dim:        int           = 6,
    use_return_cond:   bool          = True,
    seed:              int           = 42,
    train_noise:       float         = 0.01,   # v2 default — standard for small datasets
) -> Tuple[TrajectoryDatasetV2, TrajectoryDatasetV2, DataNormalizer]:
    """
    v2 pipeline: load → split episodes → sliding window → normalize → split.
    Returns (train_ds, val_ds, normalizer). Validation uses train_noise=0.
    """
    rng = np.random.default_rng(seed)

    raw = load_npz(data_path) if data_path else None
    if raw is None:
        raise RuntimeError(f"data_path required (no d4rl loader in v2 path).")

    if "pre_split_trajectories" in raw:
        all_trajs   = raw["pre_split_trajectories"]
        all_returns = all_trajs[:, :, -1].sum(axis=1)
    else:
        episodes = split_to_episodes(**{k: raw[k] for k in
                   ["observations", "actions", "rewards", "terminals", "timeouts"]})
        all_trajs, all_returns = make_subtrajectories(episodes, trajectory_length, stride)

    print(f"[v2] Sub-trajectories: {len(all_trajs):,}  |  shape: {all_trajs.shape}  "
          f"|  return range: [{all_returns.min():.0f}, {all_returns.max():.0f}]")

    # Fit normalizer on (obs, action, reward) — DataNormalizer already does this.
    normalizer = DataNormalizer.from_trajectories(all_trajs, obs_dim, action_dim)
    print(f"[v2] Reward normalization: mean={float(normalizer.reward.mean):.3f}  "
          f"std={float(normalizer.reward.std):.3f}")

    # Train / val split
    n     = len(all_trajs)
    n_val = max(1, int(n * val_fraction))
    perm  = rng.permutation(n)
    val_i, train_i = perm[:n_val], perm[n_val:]

    train_ds = TrajectoryDatasetV2(
        all_trajs[train_i], all_returns[train_i], normalizer, obs_dim, action_dim,
        use_return_cond=use_return_cond, train_noise=train_noise,
    )
    val_ds = TrajectoryDatasetV2(
        all_trajs[val_i], all_returns[val_i], normalizer, obs_dim, action_dim,
        use_return_cond=use_return_cond, train_noise=0.0,   # no aug on val
    )

    print(f"[v2] Train: {len(train_ds):,}   Val: {len(val_ds):,}   "
          f"train_noise={train_noise}")
    return train_ds, val_ds, normalizer


# Quick smoke test
if __name__ == "__main__":
    # Synthetic data round-trip
    T, N = 100, 200
    obs_dim, action_dim = 17, 6
    D_full = obs_dim + action_dim + 1   # includes reward

    fake = np.random.randn(N, T, D_full).astype(np.float32)
    fake_returns = fake[:, :, -1].sum(axis=1)

    norm = DataNormalizer.from_trajectories(fake, obs_dim, action_dim)
    normed   = norm.normalize_batch_with_reward(fake, obs_dim, action_dim)
    denormed = norm.denormalize_batch_with_reward(normed, obs_dim, action_dim)
    assert np.allclose(fake, denormed, atol=1e-5), "v2 normalize round-trip failed"
    print("[v2] Normalizer with reward — round-trip OK")

    ds = TrajectoryDatasetV2(fake, fake_returns, norm, obs_dim, action_dim,
                             train_noise=0.01)
    s = ds[0]
    assert s["trajectory"].shape == (T, D_full), \
        f"Expected ({T}, {D_full}), got {s['trajectory'].shape}"
    assert s["condition"].shape == (obs_dim + 1,)
    print(f"[v2] Dataset sample: trajectory={tuple(s['trajectory'].shape)}  "
          f"condition={tuple(s['condition'].shape)}")
    print("[v2] All tests passed.")
