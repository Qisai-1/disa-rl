"""
Replay buffers for DiSA-RL offline training.

DiSA-RL improvements over standard buffer:

1. Return-weighted synthetic sampling:
   Transitions from high-return synthetic episodes are sampled more
   frequently. This biases learning toward the best synthetic behavior
   rather than treating all synthetic transitions equally.

2. Separate real batch access:
   AugmentedReplayBuffer.sample_real() returns a batch of ONLY real
   transitions — used by the BC anchor in agent.py.

Buffer types:
  ReplayBuffer          — real D4RL transitions from .npz
  SyntheticBuffer       — pre-generated synthetic transitions with
                          return-weighted sampling
  AugmentedReplayBuffer — mixes real + synthetic at ratio alpha
"""

from __future__ import annotations
import numpy as np
import torch
from torch import Tensor
from typing import Dict, Optional


class ReplayBuffer:
    """Real D4RL transition buffer loaded from .npz."""

    def __init__(self, data_path: str, device: torch.device,
                 terminal_penalty: bool = True):
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
    """
    Pre-generated synthetic transition buffer with return-weighted sampling.

    DiSA-RL improvement: transitions from high-return synthetic episodes
    are sampled proportionally more often than low-return ones.
    This means the policy learns from the BEST synthetic behavior,
    not an average of all synthetic transitions.

    Parameters
    ----------
    data_path        : path to synthetic_transitions.npz
    device           : torch device
    return_weighting : if True, sample proportional to episode return
    temperature      : β for softmax weighting (lower = more uniform)
    """

    def __init__(
        self,
        data_path:        str,
        device:           torch.device,
        return_weighting: bool  = True,
        temperature:      float = 1.0,
    ):
        self.device = device
        data = np.load(data_path, allow_pickle=True)

        self.obs      = data["observations"].astype(np.float32)
        self.actions  = data["actions"].astype(np.float32)
        self.rewards  = data["rewards"].astype(np.float32)
        self.next_obs = data["next_observations"].astype(np.float32)
        self.done     = data["terminals"].astype(np.float32)
        self.size     = len(self.obs)

        # ── Return-weighted sampling ───────────────────────────────────────
        # Compute per-transition weight based on the episode return it belongs to.
        # Transitions in high-return episodes are sampled more often.
        self._weights = None
        if return_weighting:
            self._weights = self._compute_weights(temperature)
            print(f"SyntheticBuffer      : {self.size:,} transitions  "
                  f"obs={self.obs.shape}  "
                  f"r=[{self.rewards.min():.2f}, {self.rewards.max():.2f}]  "
                  f"[return-weighted sampling]")
        else:
            print(f"SyntheticBuffer      : {self.size:,} transitions  "
                  f"obs={self.obs.shape}  "
                  f"r=[{self.rewards.min():.2f}, {self.rewards.max():.2f}]")

    def _compute_weights(self, temperature: float) -> np.ndarray:
        """
        Compute per-transition sampling weights based on episode return.

        Algorithm:
        1. Split transitions into episodes using done flags
        2. Compute total return per episode
        3. Softmax over episode returns → episode weights
        4. Assign each transition the weight of its episode
        """
        weights   = np.ones(self.size, dtype=np.float32)
        ep_start  = 0
        ep_returns = []
        ep_ranges  = []

        for i in range(self.size):
            if self.done[i] > 0.5 or i == self.size - 1:
                ep_return = self.rewards[ep_start:i+1].sum()
                ep_returns.append(ep_return)
                ep_ranges.append((ep_start, i+1))
                ep_start = i + 1

        if len(ep_returns) == 0:
            return weights

        # Softmax over episode returns
        ep_returns = np.array(ep_returns)
        # Normalize before softmax for numerical stability
        ep_returns_norm = (ep_returns - ep_returns.mean()) / (ep_returns.std() + 1e-8)
        ep_weights = np.exp(ep_returns_norm / temperature)
        ep_weights = ep_weights / ep_weights.sum()

        # Assign episode weight to each transition
        for (start, end), w in zip(ep_ranges, ep_weights):
            weights[start:end] = w * (end - start)  # scale by ep length

        # Normalize to sum to 1
        weights = weights / weights.sum()
        return weights

    def sample(self, batch_size: int) -> Dict[str, Tensor]:
        if self._weights is not None:
            # Return-weighted sampling
            idx = np.random.choice(self.size, size=batch_size,
                                   replace=True, p=self._weights)
        else:
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

    DiSA-RL improvements:
    1. sample() returns mixed batch for Q/V learning
    2. sample_real() returns ONLY real transitions for BC anchor

    Parameters
    ----------
    real_buffer      : ReplayBuffer of real D4RL data
    synthetic_buffer : SyntheticBuffer (None = pure real)
    alpha            : fraction of real data (0.5 = 50% real, 50% synthetic)
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
        """Sample a mixed minibatch for Q/V/actor training."""
        if self.synthetic is None or self.alpha >= 1.0:
            return self.real.sample(batch_size)

        n_real = max(1, round(batch_size * self.alpha))
        n_syn  = batch_size - n_real

        real_batch = self.real.sample(n_real)
        syn_batch  = self.synthetic.sample(n_syn)

        return {
            k: torch.cat([real_batch[k], syn_batch[k]], dim=0)
            for k in real_batch
        }

    def sample_real(self, batch_size: int) -> Dict[str, Tensor]:
        """
        Sample ONLY real transitions — used for BC anchor in agent.py.
        Always returns real data regardless of alpha.
        """
        return self.real.sample(batch_size)

    def __len__(self) -> int:
        return len(self.real)

    @property
    def real_size(self) -> int:
        return len(self.real)