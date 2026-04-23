"""
Unified replay buffer for DiSA-RL online training.

Three data sources, one buffer:

1. Offline synthetic (pre-generated, fixed)
   Loaded from ./data/synthetic/<env>/synthetic_transitions.npz
   Generated before online training by generate_synthetic_data.py
   Always available — never empty

2. Real environment data (grows during online training)
   Collected by the online SAC policy interacting with MuJoCo
   Starts empty, grows throughout online training

3. Fresh synthetic (async, optional)
   Generated on-the-fly by the background diffusion process
   Reflects the fine-tuned diffusion model
   Falls back to offline synthetic if queue is empty

Sampling strategy (adaptive ρ):
   Each sample() call draws:
     n_real = round(batch_size * real_fraction)
     n_syn  = batch_size - n_real
   where real_fraction = 1 / (1 + rho)
   rho is updated externally via set_rho() based on εroll

   Early online (rho low):    mostly synthetic  → stabilizes training
   Late online  (rho high):   mostly real       → adapts to environment
"""

from __future__ import annotations
import numpy as np
import torch
from typing import Dict, Optional
from torch import Tensor


# ──────────────────────────────────────────────────────────────────────────────
# Circular buffer for real environment data
# ──────────────────────────────────────────────────────────────────────────────

class RealEnvBuffer:
    """
    Circular buffer for transitions collected from the real environment.
    Grows during online training up to max_size, then overwrites oldest.
    """

    def __init__(
        self,
        obs_dim:    int,
        action_dim: int,
        max_size:   int = 1_000_000,
        device:     torch.device = torch.device("cpu"),
    ):
        self.obs_dim    = obs_dim
        self.action_dim = action_dim
        self.max_size   = max_size
        self.device     = device
        self.ptr        = 0      # next write position
        self.size       = 0      # current number of transitions

        # Pre-allocate numpy arrays (faster than growing lists)
        self.obs      = np.zeros((max_size, obs_dim),    dtype=np.float32)
        self.actions  = np.zeros((max_size, action_dim), dtype=np.float32)
        self.rewards  = np.zeros((max_size,),            dtype=np.float32)
        self.next_obs = np.zeros((max_size, obs_dim),    dtype=np.float32)
        self.done     = np.zeros((max_size,),            dtype=np.float32)

    def add(
        self,
        obs:      np.ndarray,
        action:   np.ndarray,
        reward:   float,
        next_obs: np.ndarray,
        done:     bool,
    ) -> None:
        """Add a single transition."""
        self.obs[self.ptr]      = obs
        self.actions[self.ptr]  = action
        self.rewards[self.ptr]  = reward
        self.next_obs[self.ptr] = next_obs
        self.done[self.ptr]     = float(done)
        self.ptr  = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def add_batch(self, transitions: Dict[str, np.ndarray]) -> None:
        """Add a batch of transitions at once."""
        n = len(transitions["obs"])
        for i in range(n):
            self.add(
                transitions["obs"][i],
                transitions["action"][i],
                transitions["reward"][i],
                transitions["next_obs"][i],
                bool(transitions["done"][i]),
            )

    def sample(self, batch_size: int) -> Dict[str, Tensor]:
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs":      torch.from_numpy(self.obs[idx]).to(self.device),
            "action":   torch.from_numpy(self.actions[idx]).to(self.device),
            "reward":   torch.from_numpy(self.rewards[idx]).to(self.device),
            "next_obs": torch.from_numpy(self.next_obs[idx]).to(self.device),
            "done":     torch.from_numpy(self.done[idx]).to(self.device),
        }

    def recent(self, n: int) -> Dict[str, np.ndarray]:
        """Return the n most recently added transitions (for εroll estimation)."""
        if self.size < n:
            n = self.size
        # Circular buffer — compute indices going backwards from ptr
        end   = self.ptr
        start = (end - n) % self.max_size
        if start < end:
            idx = np.arange(start, end)
        else:
            idx = np.concatenate([np.arange(start, self.max_size), np.arange(0, end)])
        return {
            "obs":    self.obs[idx],
            "action": self.actions[idx],
        }

    def __len__(self) -> int:
        return self.size


# ──────────────────────────────────────────────────────────────────────────────
# Offline synthetic buffer (static, loaded from disk)
# ──────────────────────────────────────────────────────────────────────────────

class OfflineSyntheticBuffer:
    """
    Pre-generated synthetic transitions loaded from disk.
    Static throughout online training — provides stable background data.
    """

    def __init__(self, data_path: str, device: torch.device):
        data = np.load(data_path, allow_pickle=True)
        self.obs      = data["observations"].astype(np.float32)
        self.actions  = data["actions"].astype(np.float32)
        self.rewards  = data["rewards"].astype(np.float32)
        self.next_obs = data["next_observations"].astype(np.float32)
        self.done     = data["terminals"].astype(np.float32)
        self.size     = len(self.obs)
        self.device   = device
        print(f"OfflineSyntheticBuffer: {self.size:,} transitions  "
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


# ──────────────────────────────────────────────────────────────────────────────
# Unified online buffer
# ──────────────────────────────────────────────────────────────────────────────

class OnlineBuffer:
    """
    Unified buffer that mixes real env data with synthetic data.

    Sampling strategy:
        real_fraction = 1 / (1 + rho)
        n_real = round(batch_size * real_fraction)
        n_syn  = batch_size - n_real

    rho is updated externally via set_rho() based on εroll.
    Higher rho = more synthetic data.

    Synthetic priority:
        1. Fresh synthetic queue (from fine-tuned pψ)
        2. Offline synthetic buffer (from original pψ)

    Parameters
    ----------
    real_buffer     : RealEnvBuffer (grows during online training)
    offline_syn     : OfflineSyntheticBuffer (static, always available)
    fresh_syn_queue : multiprocessing.Queue from async_generator (optional)
    rho             : initial synthetic ratio (default 1.0 = 50/50)
    device          : torch device
    """

    def __init__(
        self,
        real_buffer:      RealEnvBuffer,
        offline_syn:      OfflineSyntheticBuffer,
        fresh_syn_queue   = None,   # mp.Queue or None
        rho:              float = 1.0,
        device:           torch.device = torch.device("cpu"),
    ):
        self.real       = real_buffer
        self.offline    = offline_syn
        self.fresh_q    = fresh_syn_queue
        self.rho        = rho
        self.device     = device

        # Fresh synthetic cache — we cache a batch from the queue
        # to avoid overhead on every sample() call
        self._fresh_cache:     Optional[Dict[str, np.ndarray]] = None
        self._fresh_cache_ptr: int = 0

        print(f"OnlineBuffer initialized  |  "
              f"offline_syn={len(offline_syn):,}  "
              f"rho={rho:.2f}  "
              f"async_gen={'enabled' if fresh_syn_queue else 'disabled'}")

    def set_rho(self, rho: float) -> None:
        """Update synthetic ratio. Called by train_online when εroll changes."""
        self.rho = float(np.clip(rho, 0.0, 10.0))

    @property
    def real_fraction(self) -> float:
        """Fraction of batch that comes from real env data."""
        return 1.0 / (1.0 + self.rho)

    def _get_synthetic(self, n: int) -> Dict[str, Tensor]:
        """
        Get n synthetic transitions.
        Prefers fresh queue, falls back to offline buffer.
        """
        # Try fresh queue first
        if self.fresh_q is not None:
            try:
                # Refill cache from queue if empty
                if self._fresh_cache is None or \
                   self._fresh_cache_ptr >= len(self._fresh_cache["obs"]):
                    batch = self.fresh_q.get_nowait()
                    self._fresh_cache     = batch
                    self._fresh_cache_ptr = 0

                # Draw from cache
                ptr  = self._fresh_cache_ptr
                take = min(n, len(self._fresh_cache["obs"]) - ptr)
                result = {
                    k: torch.from_numpy(
                        self._fresh_cache[k][ptr:ptr+take]
                    ).to(self.device)
                    for k in ["obs", "action", "reward", "next_obs", "done"]
                }
                self._fresh_cache_ptr += take

                # If we need more, fill from offline buffer
                if take < n:
                    extra = self.offline.sample(n - take)
                    result = {
                        k: torch.cat([result[k], extra[k]], dim=0)
                        for k in result
                    }
                return result

            except Exception:
                pass  # Queue empty or error — fall through to offline

        # Fallback: offline synthetic buffer always available
        return self.offline.sample(n)

    def sample(self, batch_size: int) -> Dict[str, Tensor]:
        """
        Sample a mixed batch according to current rho.

        If real buffer is empty (early online training), use pure synthetic.
        As real data accumulates, transitions to mostly real.
        """
        # If no real data yet, use pure synthetic
        if len(self.real) == 0:
            return self._get_synthetic(batch_size)

        n_real = max(1, round(batch_size * self.real_fraction))
        n_syn  = batch_size - n_real

        real_batch = self.real.sample(n_real)

        if n_syn == 0:
            return real_batch

        syn_batch = self._get_synthetic(n_syn)

        return {
            k: torch.cat([real_batch[k], syn_batch[k]], dim=0)
            for k in real_batch
        }

    def stats(self) -> Dict[str, float]:
        return {
            "buffer/real_size":    len(self.real),
            "buffer/offline_size": len(self.offline),
            "buffer/rho":          self.rho,
            "buffer/real_fraction": self.real_fraction,
        }
