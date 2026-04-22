"""
Replay buffers for DiSA-RL offline training.

Two buffer types:

1. ReplayBuffer
   Loads real D4RL transitions from a .npz file.

2. AugmentedReplayBuffer
   Mixes real D4RL transitions with pre-generated synthetic transitions
   loaded from disk (produced by generate_synthetic_data.py).

   Each sample() call draws:
     n_real = round(batch_size * alpha)  from the real buffer
     n_syn  = batch_size - n_real        from the synthetic buffer

   alpha=1.0 → pure real (identical to plain ReplayBuffer)
   alpha=0.5 → 50% real, 50% synthetic

Design: synthetic data is pre-generated and loaded once at startup.
This is ~10x faster than on-the-fly generation and simpler to debug.
"""

from __future__ import annotations
import numpy as np
import torch
from torch import Tensor
from typing import Dict, Optional


class ReplayBuffer:
    """Real D4RL transition buffer loaded from .npz."""

    def __init__(self, data_path: str, device: torch.device, terminal_penalty: bool = True):
        self.device = device
        data = np.load(data_path, allow_pickle=True)

        obs       = data["observations"].astype(np.float32)
        actions   = data["actions"].astype(np.float32)
        rewards   = data["rewards"].astype(np.float32)
        terminals = data["terminals"].astype(np.float32)
        timeouts  = data.get("timeouts", np.zeros_like(terminals)).astype(np.float32)

        if "next_observations" in data:
            next_obs = data["next_observations"].astype(np.float32)
        else:
            next_obs = np.concatenate([obs[1:], obs[-1:]], axis=0)

        done = np.clip(terminals + timeouts, 0, 1).astype(np.float32)

        if terminal_penalty:
            rewards = rewards - terminals

        self.obs      = obs
        self.actions  = actions
        self.rewards  = rewards
        self.next_obs = next_obs
        self.done     = done
        self.size     = len(obs)

        print(f"ReplayBuffer (real)  : {self.size:,} transitions  "
              f"obs={obs.shape}  act={actions.shape}  "
              f"r=[{rewards.min():.2f}, {rewards.max():.2f}]")

    def sample(self, batch_size: int) -> Dict[str, Tensor]:
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs":      torch.from_numpy(self.obs[idx]).to(self.device),
            "action":   torch.from_numpy(self.actions[idx]).to(self.device),
            "reward":   torch.from_numpy(self.rewards[idx]).to(self.device),
            "next_obs": torch.from_numpy(self.next_obs[idx]).to(self.device),
            "done":     torch.from_numpy(self.done[idx]).to(self.device),
        }

    def __len__(self) -> int:
        return self.size


class SyntheticBuffer:
    """Pre-generated synthetic transition buffer loaded from disk."""

    def __init__(self, data_path: str, device: torch.device):
        self.device = device
        data = np.load(data_path, allow_pickle=True)

        self.obs      = data["observations"].astype(np.float32)
        self.actions  = data["actions"].astype(np.float32)
        self.rewards  = data["rewards"].astype(np.float32)
        self.next_obs = data["next_observations"].astype(np.float32)
        self.done     = data["terminals"].astype(np.float32)
        self.size     = len(self.obs)

        print(f"SyntheticBuffer      : {self.size:,} transitions  "
              f"obs={self.obs.shape}  act={self.actions.shape}  "
              f"r=[{self.rewards.min():.2f}, {self.rewards.max():.2f}]")

    def sample(self, batch_size: int) -> Dict[str, Tensor]:
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs":      torch.from_numpy(self.obs[idx]).to(self.device),
            "action":   torch.from_numpy(self.actions[idx]).to(self.device),
            "reward":   torch.from_numpy(self.rewards[idx]).to(self.device),
            "next_obs": torch.from_numpy(self.next_obs[idx]).to(self.device),
            "done":     torch.from_numpy(self.done[idx]).to(self.device),
        }

    def __len__(self) -> int:
        return self.size


class AugmentedReplayBuffer:
    """
    Mixes real D4RL transitions with pre-generated synthetic transitions.

    Parameters
    ----------
    real_buffer      : ReplayBuffer of real D4RL data
    synthetic_buffer : SyntheticBuffer of pre-generated data (None = pure real)
    alpha            : fraction of real data  (0.5 = 50% real, 50% synthetic)
    """

    def __init__(
        self,
        real_buffer:      ReplayBuffer,
        synthetic_buffer: Optional[SyntheticBuffer] = None,
        alpha:            float = 0.5,
    ):
        self.real      = real_buffer
        self.synthetic = synthetic_buffer
        self.alpha     = float(np.clip(alpha, 0.0, 1.0))

        if synthetic_buffer is not None:
            print(f"AugmentedReplayBuffer: alpha={self.alpha:.2f}  "
                  f"({(1-self.alpha)*100:.0f}% synthetic)")
        else:
            print("AugmentedReplayBuffer: no synthetic data — pure real")

    def set_alpha(self, alpha: float) -> None:
        self.alpha = float(np.clip(alpha, 0.0, 1.0))

    def sample(self, batch_size: int) -> Dict[str, Tensor]:
        if self.synthetic is None or self.alpha >= 1.0:
            return self.real.sample(batch_size)

        n_real = max(1, round(batch_size * self.alpha))
        n_syn  = batch_size - n_real

        real_batch = self.real.sample(n_real)
        syn_batch  = self.synthetic.sample(n_syn)

        return {k: torch.cat([real_batch[k], syn_batch[k]], dim=0) for k in real_batch}

    def __len__(self) -> int:
        return len(self.real)