"""
Replay buffers for DiSA-RL offline training.

DiSA-RL improvements:

1. Return-weighted synthetic sampling:
   Transitions from high-return synthetic episodes are sampled more
   frequently. This biases learning toward the best synthetic behavior
   rather than treating all synthetic transitions equally.

2. Reward normalization for synthetic data:
   Synthetic rewards are normalized to match real data distribution.
   This prevents Q-value miscalibration caused by OOD synthetic rewards
   (e.g. synthetic mean=9.5 vs real mean=4.77).

3. Reward-based OOD filtering:
   Synthetic transitions with rewards far outside the real data range
   are discarded before training begins.

4. Separate real batch access:
   AugmentedReplayBuffer.sample_real() returns ONLY real transitions
   — used by the BC anchor in agent.py.
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

        # Store real reward stats for synthetic normalization
        self.reward_mean = float(rewards.mean())
        self.reward_std  = float(rewards.std() + 1e-8)

        print(f"ReplayBuffer (real)  : {self.size:,} transitions  "
              f"obs={obs.shape}  act={actions.shape}  "
              f"r=[{rewards.min():.2f}, {rewards.max():.2f}]  "
              f"mean={self.reward_mean:.3f}  std={self.reward_std:.3f}")

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
    Pre-generated synthetic transition buffer.

    Improvements:
    1. Reward normalization — scales synthetic rewards to match real distribution
    2. OOD reward filter — removes transitions with unrealistic rewards
    3. Return-weighted sampling — high-return episodes sampled more often

    Parameters
    ----------
    data_path        : path to synthetic_transitions.npz
    device           : torch device
    real_reward_mean : mean of real data rewards (for normalization)
    real_reward_std  : std of real data rewards (for normalization)
    normalize_rewards: if True, normalize synthetic rewards to match real
    filter_sigma     : remove transitions outside ±N sigma of real reward
                       (None = no filtering)
    return_weighting : if True, sample proportional to episode return
    temperature      : β for softmax return weighting
    """

    def __init__(
        self,
        data_path:         str,
        device:            torch.device,
        real_reward_mean:  float = 0.0,
        real_reward_std:   float = 1.0,
        normalize_rewards: bool  = True,
        filter_sigma:      Optional[float] = 3.0,
        return_weighting:  bool  = True,
        temperature:       float = 1.0,
    ):
        self.device = device
        data = np.load(data_path, allow_pickle=True)

        obs      = data["observations"].astype(np.float32)
        actions  = data["actions"].astype(np.float32)
        rewards  = data["rewards"].astype(np.float32)
        next_obs = data["next_observations"].astype(np.float32)
        done     = data["terminals"].astype(np.float32)

        n_original = len(obs)

        # ── Step 1: OOD reward filter ──────────────────────────────────────
        # Remove transitions with rewards far outside the real data range.
        # These correspond to physically impossible states.
        if filter_sigma is not None:
            r_min = real_reward_mean - filter_sigma * real_reward_std
            r_max = real_reward_mean + filter_sigma * real_reward_std
            mask  = (rewards >= r_min) & (rewards <= r_max)
            keep  = np.where(mask)[0]

            if len(keep) > 0:
                obs      = obs[keep]
                actions  = actions[keep]
                rewards  = rewards[keep]
                next_obs = next_obs[keep]
                done     = done[keep]
                pct = 100 * len(keep) / n_original
                print(f"  OOD filter (±{filter_sigma}σ): kept "
                      f"{len(keep):,}/{n_original:,} ({pct:.1f}%)")
            else:
                print(f"  OOD filter: ALL transitions filtered — "
                      f"syn reward range [{rewards.min():.2f}, {rewards.max():.2f}] "
                      f"vs real range [{r_min:.2f}, {r_max:.2f}]")
                print(f"  Disabling filter and using reward normalization only.")

        # ── Step 2: Reward normalization ───────────────────────────────────
        # Scale synthetic rewards to match real reward distribution.
        # This prevents Q-value miscalibration even if synthetic rewards
        # are slightly off.
        if normalize_rewards and real_reward_std > 0:
            syn_mean = float(rewards.mean())
            syn_std  = float(rewards.std() + 1e-8)
            rewards  = (rewards - syn_mean) / syn_std * real_reward_std + real_reward_mean
            print(f"  Reward normalization: "
                  f"syn [{syn_mean:.3f}±{syn_std:.3f}] → "
                  f"real [{real_reward_mean:.3f}±{real_reward_std:.3f}]")

        self.obs      = obs
        self.actions  = actions
        self.rewards  = rewards
        self.next_obs = next_obs
        self.done     = done
        self.size     = len(obs)

        # ── Step 3: Return-weighted sampling ──────────────────────────────
        self._weights = None
        if return_weighting and self.size > 0:
            self._weights = self._compute_weights(temperature)

        print(f"SyntheticBuffer      : {self.size:,} transitions  "
              f"r=[{self.rewards.min():.2f}, {self.rewards.max():.2f}]  "
              f"mean={self.rewards.mean():.3f}  "
              f"{'[weighted]' if self._weights is not None else ''}")

    def _compute_weights(self, temperature: float) -> np.ndarray:
        weights    = np.ones(self.size, dtype=np.float32)
        ep_start   = 0
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

        ep_returns      = np.array(ep_returns)
        ep_returns_norm = (ep_returns - ep_returns.mean()) / (ep_returns.std() + 1e-8)
        ep_weights      = np.exp(ep_returns_norm / temperature)
        ep_weights      = ep_weights / ep_weights.sum()

        for (start, end), w in zip(ep_ranges, ep_weights):
            weights[start:end] = w * (end - start)

        return weights / weights.sum()

    def sample(self, batch_size: int) -> Dict[str, Tensor]:
        if self.size == 0:
            raise RuntimeError("SyntheticBuffer is empty after filtering")

        if self._weights is not None:
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

    sample()      → mixed batch for Q/V/actor training
    sample_real() → real-only batch for BC anchor
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

        # If synthetic buffer is empty after filtering, fall back to pure real
        if synthetic_buffer is not None and len(synthetic_buffer) == 0:
            print("AugmentedReplayBuffer: synthetic buffer empty — using pure real")
            self.synthetic = None
            self.alpha     = 1.0

        if self.synthetic is not None:
            print(f"AugmentedReplayBuffer: alpha={self.alpha:.2f}  "
                  f"({self.alpha*100:.0f}% real + {(1-self.alpha)*100:.0f}% synthetic)")
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

        return {
            k: torch.cat([real_batch[k], syn_batch[k]], dim=0)
            for k in real_batch
        }

    def sample_real(self, batch_size: int) -> Dict[str, Tensor]:
        """Always returns real transitions — used for BC anchor."""
        return self.real.sample(batch_size)

    def __len__(self) -> int:
        return len(self.real)

    @property
    def real_size(self) -> int:
        return len(self.real)