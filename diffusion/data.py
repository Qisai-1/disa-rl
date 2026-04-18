"""
Data loading and preprocessing for D4RL trajectory datasets.

Supports two input formats:
  1. D4RL via gym + d4rl library (online, requires pip install d4rl)
  2. Pre-saved .npz file (offline, no d4rl needed)

Pipeline:
  raw transitions → split episodes → sliding-window sub-trajectories
  → fit normalizer → TrajectoryDataset (normalised, with condition vector)

Normalisation strategy:
  Per-modality z-score: obs, action, reward are normalised independently.
  This prevents reward scale (often 0-10) from being swamped by observation
  scale (often -10 to +10 across 17 dimensions) in the MSE loss.

Condition vector:
  [norm_s0 (obs_dim) | norm_return (1)]  used for CFG conditioning.
  norm_return = (R - μ_R) / σ_R  so the model learns relative return quality.
"""

from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# Normalizer
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ChannelStats:
    mean: np.ndarray
    std:  np.ndarray

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / (self.std + 1e-8)

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        return x * (self.std + 1e-8) + self.mean


class DataNormalizer:
    """
    Per-modality z-score normalizer for (obs, action, reward) trajectories.
    Also tracks cumulative return statistics for the CFG condition vector.
    """

    def __init__(
        self,
        obs:         ChannelStats,
        action:      ChannelStats,
        reward:      ChannelStats,
        return_mean: float = 0.0,
        return_std:  float = 1.0,
    ):
        self.obs    = obs
        self.action = action
        self.reward = reward
        self.return_mean = float(return_mean)
        self.return_std  = float(return_std)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_trajectories(
        cls,
        trajs:      np.ndarray,   # (N, T, D)
        obs_dim:    int,
        action_dim: int,
    ) -> "DataNormalizer":
        flat_obs    = trajs[:, :, :obs_dim].reshape(-1, obs_dim)
        flat_action = trajs[:, :, obs_dim:obs_dim + action_dim].reshape(-1, action_dim)
        flat_reward = trajs[:, :, -1].reshape(-1, 1)
        returns     = trajs[:, :, -1].sum(axis=1)
        return cls(
            obs    = ChannelStats(flat_obs.mean(0),    flat_obs.std(0)    + 1e-8),
            action = ChannelStats(flat_action.mean(0), flat_action.std(0) + 1e-8),
            reward = ChannelStats(flat_reward.mean(0), flat_reward.std(0) + 1e-8),
            return_mean = float(returns.mean()),
            return_std  = float(returns.std() + 1e-8),
        )

    # ------------------------------------------------------------------
    # Normalize / Denormalize  (works on numpy arrays of any leading shape)
    # ------------------------------------------------------------------

    def normalize_batch(self, trajs: np.ndarray, obs_dim: int, action_dim: int) -> np.ndarray:
        out = trajs.copy().astype(np.float32)
        out[..., :obs_dim]                     = self.obs.normalize(trajs[..., :obs_dim])
        out[..., obs_dim:obs_dim + action_dim] = self.action.normalize(trajs[..., obs_dim:obs_dim + action_dim])
        out[..., -1:]                          = self.reward.normalize(trajs[..., -1:])
        return out

    def denormalize_batch(self, trajs: np.ndarray, obs_dim: int, action_dim: int) -> np.ndarray:
        out = trajs.copy().astype(np.float32)
        out[..., :obs_dim]                     = self.obs.denormalize(trajs[..., :obs_dim])
        out[..., obs_dim:obs_dim + action_dim] = self.action.denormalize(trajs[..., obs_dim:obs_dim + action_dim])
        out[..., -1:]                          = self.reward.denormalize(trajs[..., -1:])
        return out

    def normalize_return(self, r: float) -> float:
        return (r - self.return_mean) / (self.return_std + 1e-8)

    def denormalize_return(self, r: float) -> float:
        return r * (self.return_std + 1e-8) + self.return_mean

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def as_dict(self) -> Dict:
        return dict(
            obs_mean    = self.obs.mean,    obs_std    = self.obs.std,
            action_mean = self.action.mean, action_std = self.action.std,
            reward_mean = self.reward.mean, reward_std = self.reward.std,
            return_mean = np.array([self.return_mean]),
            return_std  = np.array([self.return_std]),
        )

    @classmethod
    def from_dict(cls, d: Dict) -> "DataNormalizer":
        return cls(
            obs    = ChannelStats(np.array(d["obs_mean"]),    np.array(d["obs_std"])),
            action = ChannelStats(np.array(d["action_mean"]), np.array(d["action_std"])),
            reward = ChannelStats(np.array(d["reward_mean"]), np.array(d["reward_std"])),
            return_mean = float(np.array(d["return_mean"]).flatten()[0]),
            return_std  = float(np.array(d["return_std"]).flatten()[0]),
        )

    def save(self, path: str) -> None:
        np.savez(path, **self.as_dict())

    @classmethod
    def load(cls, path: str) -> "DataNormalizer":
        return cls.from_dict(dict(np.load(path)))


# ──────────────────────────────────────────────────────────────────────────────
# Raw data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_d4rl(dataset_name: str) -> Dict[str, np.ndarray]:
    """
    Load a D4RL dataset.

    mujoco_py (required by the old d4rl library) does not compile on
    Python 3.10+ with recent Cython versions.

    Use download_data.py instead — it downloads the raw .hdf5 files
    directly from the D4RL host without needing mujoco_py:

        python download_data.py --datasets halfcheetah-medium-v2

    Then point data_path to the saved .npz:

        data = DataConfig(data_path="./data/halfcheetah-medium-v2.npz")
    """
    raise RuntimeError(
        f"\n\nDirect d4rl loading is disabled (mujoco_py Cython issue).\n"
        f"Download the dataset with:\n\n"
        f"    python download_data.py --datasets {dataset_name}\n\n"
        f"Then update train.py:\n"
        f"    data = DataConfig(data_path='./data/{dataset_name}.npz')\n"
    )


def load_npz(path: str) -> Dict[str, np.ndarray]:
    """
    Load from a .npz file.  Supports two formats:

    Format A — flat transitions (preferred):
        keys: observations, actions, rewards, terminals, [timeouts]

    Format B — pre-split trajectories (legacy):
        key:  'trajectories' or 'arr_0'  with shape (N, T, D)
    """
    data = np.load(path, allow_pickle=True)
    keys = list(data.keys())

    if "observations" in keys:
        return dict(
            observations = data["observations"].astype(np.float32),
            actions      = data["actions"].astype(np.float32),
            rewards      = data["rewards"].astype(np.float32),
            terminals    = data["terminals"].astype(bool),
            timeouts     = data.get("timeouts",
                           np.zeros(len(data["rewards"]), dtype=bool)).astype(bool),
        )

    traj_key = "trajectories" if "trajectories" in keys else "arr_0"
    if traj_key not in keys:
        raise ValueError(f"Unrecognised .npz format.  Keys: {keys}")

    return {"pre_split_trajectories": data[traj_key].astype(np.float32)}


# ──────────────────────────────────────────────────────────────────────────────
# Episode splitting  &  sliding-window sub-trajectories
# ──────────────────────────────────────────────────────────────────────────────

def split_to_episodes(
    observations: np.ndarray,
    actions:      np.ndarray,
    rewards:      np.ndarray,
    terminals:    np.ndarray,
    timeouts:     np.ndarray,
) -> List[np.ndarray]:
    """
    Concatenate (obs, action, reward) and split at episode boundaries.
    Returns list of float32 arrays, each shape (ep_len, D).
    """
    done      = np.logical_or(terminals, timeouts)
    end_idxs  = np.where(done)[0] + 1
    starts    = np.concatenate([[0], end_idxs[:-1]])

    episodes = []
    for s, e in zip(starts, end_idxs):
        if e - s < 2:
            continue
        ep = np.concatenate(
            [observations[s:e], actions[s:e], rewards[s:e, None]], axis=1
        ).astype(np.float32)
        episodes.append(ep)
    return episodes


def make_subtrajectories(
    episodes:          List[np.ndarray],
    trajectory_length: int,
    stride:            int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sliding-window sub-trajectories.

    Returns
    -------
    trajs   : (N, T, D)  float32
    returns : (N,)       float32   cumulative reward per sub-trajectory
    """
    trajs, returns = [], []
    for ep in episodes:
        for start in range(0, len(ep) - trajectory_length + 1, stride):
            sub = ep[start : start + trajectory_length]
            trajs.append(sub)
            returns.append(float(sub[:, -1].sum()))

    if not trajs:
        raise ValueError(
            f"No sub-trajectories created — check that episodes are longer "
            f"than trajectory_length={trajectory_length}."
        )
    return np.array(trajs, dtype=np.float32), np.array(returns, dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset
# ──────────────────────────────────────────────────────────────────────────────

class TrajectoryDataset(Dataset):
    """
    Dataset of normalised fixed-length trajectory sub-sequences.

    Each sample is a dict:
        'trajectory' : (T, D)        normalised (obs, action, reward)
        'condition'  : (cond_dim,)   [norm_s0 | norm_return]
    """

    def __init__(
        self,
        trajs:           np.ndarray,    # (N, T, D) raw unnormalised
        returns:         np.ndarray,    # (N,)
        normalizer:      DataNormalizer,
        obs_dim:         int  = 17,
        action_dim:      int  = 6,
        use_return_cond: bool = True,
    ):
        self.obs_dim         = obs_dim
        self.action_dim      = action_dim
        self.normalizer      = normalizer
        self.use_return_cond = use_return_cond

        # Normalise all trajectories up front (keeps __getitem__ cheap)
        self.trajs = normalizer.normalize_batch(trajs, obs_dim, action_dim)

        # Normalised returns for conditioning
        self.norm_returns = (
            (returns - normalizer.return_mean) / (normalizer.return_std + 1e-8)
        ).astype(np.float32)

        # Keep raw returns for reward-quality reporting
        self.raw_returns = returns.astype(np.float32)

    def __len__(self) -> int:
        return len(self.trajs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        traj = torch.from_numpy(self.trajs[idx])          # (T, D)
        s0   = traj[0, :self.obs_dim]                     # (obs_dim,)

        if self.use_return_cond:
            ret  = torch.tensor([self.norm_returns[idx]])
            cond = torch.cat([s0, ret], dim=0)            # (obs_dim + 1,)
        else:
            cond = s0

        return {"trajectory": traj, "condition": cond}


# ──────────────────────────────────────────────────────────────────────────────
# Top-level builder
# ──────────────────────────────────────────────────────────────────────────────

def build_datasets(
    dataset_name:      str           = "halfcheetah-medium-v2",
    data_path:         Optional[str] = None,
    trajectory_length: int           = 100,
    stride:            int           = 50,
    val_fraction:      float         = 0.05,
    obs_dim:           int           = 17,
    action_dim:        int           = 6,
    use_return_cond:   bool          = True,
    seed:              int           = 42,
) -> Tuple[TrajectoryDataset, TrajectoryDataset, DataNormalizer]:
    """
    Full pipeline: load → split episodes → sliding window → normalize → split.

    Returns
    -------
    train_ds, val_ds : TrajectoryDataset
    normalizer       : DataNormalizer   (needed at generation time)
    """
    rng = np.random.default_rng(seed)

    # 1. Load
    raw = load_npz(data_path) if data_path else load_d4rl(dataset_name)

    if "pre_split_trajectories" in raw:
        all_trajs   = raw["pre_split_trajectories"]
        all_returns = all_trajs[:, :, -1].sum(axis=1)
    else:
        episodes               = split_to_episodes(**{k: raw[k] for k in
                                 ["observations", "actions", "rewards", "terminals", "timeouts"]})
        all_trajs, all_returns = make_subtrajectories(episodes, trajectory_length, stride)

    print(f"Sub-trajectories: {len(all_trajs):,}  |  shape: {all_trajs.shape}  "
          f"|  return range: [{all_returns.min():.0f}, {all_returns.max():.0f}]")

    # 2. Fit normalizer on full dataset (before train/val split)
    normalizer = DataNormalizer.from_trajectories(all_trajs, obs_dim, action_dim)

    # 3. Train / val split
    n      = len(all_trajs)
    n_val  = max(1, int(n * val_fraction))
    perm   = rng.permutation(n)
    val_i, train_i = perm[:n_val], perm[n_val:]

    train_ds = TrajectoryDataset(all_trajs[train_i], all_returns[train_i],
                                 normalizer, obs_dim, action_dim, use_return_cond)
    val_ds   = TrajectoryDataset(all_trajs[val_i],   all_returns[val_i],
                                 normalizer, obs_dim, action_dim, use_return_cond)

    print(f"Train: {len(train_ds):,}   Val: {len(val_ds):,}")
    return train_ds, val_ds, normalizer


# ──────────────────────────────────────────────────────────────────────────────
# Quick test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile, os

    # Synthetic data test (no d4rl needed)
    T, D, N = 200, 24, 500
    obs_dim, action_dim = 17, 6

    fake_trajs = np.random.randn(N, T, D).astype(np.float32)
    fake_returns = fake_trajs[:, :, -1].sum(axis=1)

    normalizer = DataNormalizer.from_trajectories(fake_trajs, obs_dim, action_dim)

    # Normalizer round-trip
    normed   = normalizer.normalize_batch(fake_trajs, obs_dim, action_dim)
    denormed = normalizer.denormalize_batch(normed, obs_dim, action_dim)
    assert np.allclose(fake_trajs, denormed, atol=1e-5), "Normalizer round-trip failed"
    print("Normalizer round-trip: OK")

    # Save / load normalizer
    with tempfile.TemporaryDirectory() as tmp:
        npath = os.path.join(tmp, "norm.npz")
        normalizer.save(npath)
        normalizer2 = DataNormalizer.load(npath)
        assert np.allclose(normalizer.obs.mean, normalizer2.obs.mean)
        print("Normalizer save/load: OK")

    # Dataset
    ds = TrajectoryDataset(fake_trajs, fake_returns, normalizer, obs_dim, action_dim)
    sample = ds[0]
    assert sample["trajectory"].shape == (T, D)
    assert sample["condition"].shape  == (obs_dim + 1,)
    print(f"Dataset sample shapes: trajectory={sample['trajectory'].shape}  "
          f"condition={sample['condition'].shape}")

    # as_dict / from_dict round-trip
    d  = normalizer.as_dict()
    n3 = DataNormalizer.from_dict(d)
    assert np.allclose(normalizer.obs.mean, n3.obs.mean)
    print("as_dict/from_dict: OK")

    print("\nAll data tests passed.")