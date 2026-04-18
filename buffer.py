"""
Augmented replay buffer for DiSA-RL offline training.

Two buffer types:

1. ReplayBuffer — standard flat transition buffer loaded from D4RL .npz
2. AugmentedReplayBuffer — wraps ReplayBuffer and mixes in synthetic
   transitions from the diffusion generator at a configurable ratio α.

The mixing ratio α is controlled externally by the εroll adaptive schedule:
  - α = 1.0 → pure real data (start of training or when εroll is high)
  - α = 0.5 → 50% real, 50% synthetic
  - α decreases as diffusion model improves (εroll drops)

Design decision: synthetic transitions are generated on-the-fly per batch
rather than pre-generated and stored.  This ensures synthetic data always
reflects the current (periodically fine-tuned) diffusion model rather than
a stale snapshot.
"""

from __future__ import annotations
import numpy as np
import torch
from torch import Tensor
from typing import Dict, Optional


# ──────────────────────────────────────────────────────────────────────────────
# Base replay buffer
# ──────────────────────────────────────────────────────────────────────────────

class ReplayBuffer:
    """
    Flat transition buffer loaded from D4RL .npz dataset.

    Normalises observations and actions using the diffusion model's normalizer
    so IQL and the diffusion model operate in the same space.

    Note: IQL is trained on UNNORMALISED data — it handles its own internal
    normalisation via LayerNorm.  The normalizer is only used here when
    synthetic data (which comes normalised from the diffusion model) needs
    to be denormalised for mixing.
    """

    def __init__(
        self,
        data_path: str,
        device:    torch.device,
        terminal_penalty: bool = True,
    ):
        """
        Parameters
        ----------
        data_path        : path to .npz with keys observations, actions,
                           rewards, terminals, timeouts
        device           : torch device for returned batches
        terminal_penalty : if True, apply -1 reward penalty at episode end
                           (standard D4RL preprocessing)
        """
        self.device = device
        data = np.load(data_path, allow_pickle=True)

        obs      = data["observations"].astype(np.float32)
        actions  = data["actions"].astype(np.float32)
        rewards  = data["rewards"].astype(np.float32)
        terminals = data["terminals"].astype(np.float32)
        timeouts  = data.get("timeouts", np.zeros_like(terminals)).astype(np.float32)

        # next_obs: use provided or shift by 1
        if "next_observations" in data:
            next_obs = data["next_observations"].astype(np.float32)
        else:
            next_obs = np.concatenate([obs[1:], obs[-1:]], axis=0)

        # Done = terminal OR timeout
        done = np.clip(terminals + timeouts, 0, 1).astype(np.float32)

        # Optional terminal reward penalty (-1 at end of episode)
        if terminal_penalty:
            rewards = rewards - terminals.astype(np.float32)

        self.obs      = obs
        self.actions  = actions
        self.rewards  = rewards
        self.next_obs = next_obs
        self.done     = done
        self.size     = len(obs)

        print(f"ReplayBuffer loaded: {self.size:,} transitions  "
              f"obs={obs.shape}  act={actions.shape}  "
              f"r=[{rewards.min():.2f}, {rewards.max():.2f}]")

    def sample(self, batch_size: int) -> Dict[str, Tensor]:
        """Sample a random minibatch of transitions."""
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


# ──────────────────────────────────────────────────────────────────────────────
# Augmented replay buffer
# ──────────────────────────────────────────────────────────────────────────────

class AugmentedReplayBuffer:
    """
    Mixes real D4RL transitions with synthetic diffusion-generated ones.

    Each call to sample() draws:
      - n_real   = round(batch_size * alpha) transitions from the real buffer
      - n_syn    = batch_size - n_real        transitions generated on the fly

    The generator is only called when alpha < 1.0.  When alpha == 1.0 this
    is identical to plain ReplayBuffer sampling.

    Parameters
    ----------
    real_buffer    : loaded ReplayBuffer
    generator      : TrajectoryGenerator instance (generate.py)
    normalizer     : DataNormalizer — needed to denormalise synthetic data
    alpha          : fraction of real data (1.0 = no augmentation)
    target_return  : return conditioning for CFG guidance (use p90 of dataset)
    """

    def __init__(
        self,
        real_buffer:   ReplayBuffer,
        generator,                      # TrajectoryGenerator
        normalizer,                     # DataNormalizer
        alpha:         float = 1.0,
        target_return: float = 3000.0,
    ):
        self.real_buffer   = real_buffer
        self.generator     = generator
        self.normalizer    = normalizer
        self.alpha         = alpha
        self.target_return = target_return

        # Cache a small batch of synthetic transitions for efficiency
        # (generating one transition at a time is very slow)
        self._syn_cache:        Optional[Dict[str, np.ndarray]] = None
        self._syn_cache_ptr:    int = 0
        self._syn_cache_size:   int = 1024    # pre-generate this many at once

    def set_alpha(self, alpha: float) -> None:
        """Update the real/synthetic mixing ratio.  Call from training loop."""
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        # Invalidate cache when ratio changes significantly
        self._syn_cache = None

    def _refill_cache(self) -> None:
        """Generate a fresh batch of synthetic transitions."""
        obs_dim    = self.real_buffer.obs.shape[1]
        action_dim = self.real_buffer.actions.shape[1]

        # Seed from random real starting states
        idx     = np.random.randint(0, len(self.real_buffer), size=32)
        s0      = self.real_buffer.obs[idx]

        result  = self.generator.generate(
            n_trajectories = 32,
            initial_states = s0,
            target_return  = self.target_return,
        )
        # Flatten trajectories to transitions
        trans   = result["transitions"]
        n       = len(trans)

        self._syn_cache = {
            "obs":      np.stack([t["obs"]      for t in trans]).astype(np.float32),
            "action":   np.stack([t["action"]   for t in trans]).astype(np.float32),
            "reward":   np.array([t["reward"]   for t in trans], dtype=np.float32),
            "next_obs": np.stack([t["next_obs"] for t in trans]).astype(np.float32),
            "done":     np.array([t["done"]     for t in trans], dtype=np.float32),
        }
        self._syn_cache_ptr  = 0
        self._syn_cache_size = n

    def _sample_synthetic(self, n: int) -> Dict[str, Tensor]:
        """Draw n synthetic transitions from the cache, refilling as needed."""
        results = {k: [] for k in ["obs", "action", "reward", "next_obs", "done"]}
        remaining = n

        while remaining > 0:
            if self._syn_cache is None or self._syn_cache_ptr >= self._syn_cache_size:
                self._refill_cache()

            avail = self._syn_cache_size - self._syn_cache_ptr
            take  = min(remaining, avail)

            for k in results:
                results[k].append(
                    self._syn_cache[k][self._syn_cache_ptr : self._syn_cache_ptr + take]
                )
            self._syn_cache_ptr += take
            remaining -= take

        return {
            k: torch.from_numpy(np.concatenate(v, axis=0)).to(self.real_buffer.device)
            for k, v in results.items()
        }

    def sample(self, batch_size: int) -> Dict[str, Tensor]:
        """
        Sample a mixed minibatch.

        If alpha == 1.0 → pure real data (no diffusion needed).
        Otherwise mixes n_real real + n_syn synthetic transitions.
        """
        if self.alpha >= 1.0 or self.generator is None:
            return self.real_buffer.sample(batch_size)

        n_real = max(1, round(batch_size * self.alpha))
        n_syn  = batch_size - n_real

        real_batch = self.real_buffer.sample(n_real)
        syn_batch  = self._sample_synthetic(n_syn)

        return {
            k: torch.cat([real_batch[k], syn_batch[k]], dim=0)
            for k in real_batch
        }

    def __len__(self) -> int:
        return len(self.real_buffer)

    @property
    def real_size(self) -> int:
        return len(self.real_buffer)


# ──────────────────────────────────────────────────────────────────────────────
# Smoke test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile, os

    # Create a fake .npz file
    N = 10_000
    obs_dim, action_dim = 17, 6
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "fake.npz")

    np.savez(
        path,
        observations      = np.random.randn(N, obs_dim).astype(np.float32),
        actions           = np.random.randn(N, action_dim).astype(np.float32),
        rewards           = np.random.randn(N).astype(np.float32),
        terminals         = np.zeros(N, dtype=bool),
        timeouts          = np.zeros(N, dtype=bool),
        next_observations = np.random.randn(N, obs_dim).astype(np.float32),
    )

    device = torch.device("cpu")
    buf    = ReplayBuffer(path, device)

    batch = buf.sample(256)
    assert batch["obs"].shape      == (256, obs_dim)
    assert batch["action"].shape   == (256, action_dim)
    assert batch["reward"].shape   == (256,)
    assert batch["next_obs"].shape == (256, obs_dim)
    assert batch["done"].shape     == (256,)

    print(f"ReplayBuffer sample: obs={batch['obs'].shape}  OK")

    # Test AugmentedReplayBuffer without generator (alpha=1.0)
    aug = AugmentedReplayBuffer(buf, generator=None, normalizer=None, alpha=1.0)
    b2  = aug.sample(256)
    assert b2["obs"].shape == (256, obs_dim)
    print(f"AugmentedReplayBuffer (pure real): OK")
    print("All buffer tests passed.")
