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

    QCD (Pillar 2) extension: optionally tracks q_mean / q_std for
    Q-conditional diffusion. When q stats are set, callers should use
    `normalize_q` instead of `normalize_return` for the condition scalar.
    """

    def __init__(
        self,
        obs:         ChannelStats,
        action:      ChannelStats,
        reward:      ChannelStats,
        return_mean: float = 0.0,
        return_std:  float = 1.0,
        q_mean:      float = 0.0,
        q_std:       float = 1.0,
        cond_kind:   str   = "return",     # "return" or "q"
    ):
        self.obs    = obs
        self.action = action
        self.reward = reward
        self.return_mean = float(return_mean)
        self.return_std  = float(return_std)
        self.q_mean      = float(q_mean)
        self.q_std       = float(q_std)
        self.cond_kind   = cond_kind

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
        """Normalise (obs, action) trajectories — no reward dimension."""
        out = trajs.copy().astype(np.float32)
        out[..., :obs_dim]                     = self.obs.normalize(trajs[..., :obs_dim])
        out[..., obs_dim:obs_dim + action_dim] = self.action.normalize(trajs[..., obs_dim:obs_dim + action_dim])
        return out

    def denormalize_batch(self, trajs: np.ndarray, obs_dim: int, action_dim: int) -> np.ndarray:
        """Denormalise (obs, action) trajectories — no reward dimension."""
        out = trajs.copy().astype(np.float32)
        out[..., :obs_dim]                     = self.obs.denormalize(trajs[..., :obs_dim])
        out[..., obs_dim:obs_dim + action_dim] = self.action.denormalize(trajs[..., obs_dim:obs_dim + action_dim])
        return out

    def normalize_return(self, r: float) -> float:
        return (r - self.return_mean) / (self.return_std + 1e-8)

    def denormalize_return(self, r: float) -> float:
        return r * (self.return_std + 1e-8) + self.return_mean

    def normalize_q(self, q: float) -> float:
        return (q - self.q_mean) / (self.q_std + 1e-8)

    def denormalize_q(self, q: float) -> float:
        return q * (self.q_std + 1e-8) + self.q_mean

    def normalize_scalar(self, x: float) -> float:
        """Dispatch to return or Q normalization based on cond_kind."""
        return self.normalize_q(x) if self.cond_kind == "q" else self.normalize_return(x)

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
            q_mean      = np.array([self.q_mean]),
            q_std       = np.array([self.q_std]),
            cond_kind   = np.array([self.cond_kind]),
        )

    @classmethod
    def from_dict(cls, d: Dict) -> "DataNormalizer":
        # Backwards-compat: older checkpoints have no q_* / cond_kind keys
        q_mean    = float(np.array(d["q_mean"]).flatten()[0])    if "q_mean"   in d else 0.0
        q_std     = float(np.array(d["q_std"]).flatten()[0])     if "q_std"    in d else 1.0
        cond_kind = str(np.array(d["cond_kind"]).flatten()[0])   if "cond_kind" in d else "return"
        return cls(
            obs    = ChannelStats(np.array(d["obs_mean"]),    np.array(d["obs_std"])),
            action = ChannelStats(np.array(d["action_mean"]), np.array(d["action_std"])),
            reward = ChannelStats(np.array(d["reward_mean"]), np.array(d["reward_std"])),
            return_mean = float(np.array(d["return_mean"]).flatten()[0]),
            return_std  = float(np.array(d["return_std"]).flatten()[0]),
            q_mean      = q_mean,
            q_std       = q_std,
            cond_kind   = cond_kind,
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
        'trajectory' : (T, D)        normalised (obs, action)
        'condition'  : (cond_dim,)   [norm_s0 | norm_scalar]

    The conditioning scalar is either:
      - the **return** of the sub-trajectory (standard, what Decision Diffuser
        / SynthER / GTA do), or
      - a **Q-target** Q_φ(s_0, a_0) from a pretrained offline IQL critic
        (our QCD novelty — Pillar 2).

    Which scalar is used is governed by `normalizer.cond_kind`:
      "return" → use `returns`        (the original behavior)
      "q"      → use `q_targets`      (QCD)
    """

    def __init__(
        self,
        trajs:           np.ndarray,         # (N, T, D_full) — D_full includes reward
        returns:         np.ndarray,         # (N,) — sub-traj returns
        normalizer:      DataNormalizer,
        obs_dim:         int  = 17,
        action_dim:      int  = 6,
        use_return_cond: bool = True,
        q_targets:       Optional[np.ndarray] = None,   # (N,) — required if cond_kind=="q"
    ):
        self.obs_dim         = obs_dim
        self.action_dim      = action_dim
        self.normalizer      = normalizer
        self.use_return_cond = use_return_cond
        self.cond_kind       = getattr(normalizer, "cond_kind", "return")

        # Keep only (obs, action) — drop reward column before normalising
        trajs_oa = trajs[..., :obs_dim + action_dim]
        self.trajs = normalizer.normalize_batch(trajs_oa, obs_dim, action_dim)

        # Normalised conditioning scalar — return-based or Q-based
        if self.cond_kind == "q":
            if q_targets is None:
                raise ValueError(
                    "cond_kind='q' but q_targets=None. Pass q_targets to TrajectoryDataset."
                )
            self.norm_scalars = (
                (q_targets - normalizer.q_mean) / (normalizer.q_std + 1e-8)
            ).astype(np.float32)
            self.raw_scalars  = q_targets.astype(np.float32)
        else:
            self.norm_scalars = (
                (returns - normalizer.return_mean) / (normalizer.return_std + 1e-8)
            ).astype(np.float32)
            self.raw_scalars  = returns.astype(np.float32)

        # Legacy aliases (other code reads these)
        self.norm_returns = self.norm_scalars
        self.raw_returns  = self.raw_scalars

    def __len__(self) -> int:
        return len(self.trajs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        traj = torch.from_numpy(self.trajs[idx])          # (T, D)
        s0   = traj[0, :self.obs_dim]                     # (obs_dim,)

        if self.use_return_cond:
            scalar = torch.tensor([self.norm_scalars[idx]])
            cond   = torch.cat([s0, scalar], dim=0)       # (obs_dim + 1,)
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
    qcd_iql_ckpt:      Optional[str] = None,        # QCD: path to IQL critic
    qcd_use_v:         bool          = False,       # True → V(s_0); False → Q(s_0, a_0)
    qcd_device:        Optional["torch.device"] = None,
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

    # 2b. QCD (Pillar 2): compute Q-targets for each sub-trajectory if a
    # critic was provided. Q-targets become the conditioning scalar
    # in place of returns. The model arch is unchanged.
    all_q_targets = None
    if qcd_iql_ckpt is not None:
        import torch
        from diffusion.q_conditional import compute_q_targets_from_critic
        device = qcd_device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        s0 = all_trajs[:, 0, :obs_dim]
        a0 = all_trajs[:, 0, obs_dim:obs_dim + action_dim]
        all_q_targets = compute_q_targets_from_critic(
            iql_ckpt=qcd_iql_ckpt, obs=s0, actions=a0,
            device=device, use_v=qcd_use_v,
        )
        # Stash Q stats in the normalizer so generation can denormalize.
        normalizer.q_mean    = float(all_q_targets.mean())
        normalizer.q_std     = float(all_q_targets.std() + 1e-8)
        normalizer.cond_kind = "q"
        print(f"  QCD: cond_kind set to 'q'  "
              f"(Q stats: mean={normalizer.q_mean:.3f}  std={normalizer.q_std:.3f})")

    # 3. Train / val split
    n      = len(all_trajs)
    n_val  = max(1, int(n * val_fraction))
    perm   = rng.permutation(n)
    val_i, train_i = perm[:n_val], perm[n_val:]

    q_train = all_q_targets[train_i] if all_q_targets is not None else None
    q_val   = all_q_targets[val_i]   if all_q_targets is not None else None
    train_ds = TrajectoryDataset(all_trajs[train_i], all_returns[train_i],
                                 normalizer, obs_dim, action_dim, use_return_cond,
                                 q_targets=q_train)
    val_ds   = TrajectoryDataset(all_trajs[val_i],   all_returns[val_i],
                                 normalizer, obs_dim, action_dim, use_return_cond,
                                 q_targets=q_val)

    print(f"Train: {len(train_ds):,}   Val: {len(val_ds):,}  "
          f"cond_kind={normalizer.cond_kind}")
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