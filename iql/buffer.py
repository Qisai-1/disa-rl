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


def corl_reward_norm_factor(
    rewards: np.ndarray,
    terminals: np.ndarray,
    timeouts: np.ndarray,
    max_episode_steps: int = 1000,
) -> float:
    """
    Standard CORL/IQL reward normalization factor for D4RL locomotion:
        factor = max_episode_steps / (max_ep_return - min_ep_return)

    Per-step rewards multiplied by this factor put episode returns into
    a roughly [0, max_episode_steps] range, matching the temperature/expectile
    tuning in the IQL paper (Kostrikov et al., ICLR 2022).
    Reference: CORL `modify_reward` for D4RL halfcheetah/hopper/walker2d.
    """
    done = (terminals + timeouts) > 0
    ep_returns, ret = [], 0.0
    for r, d in zip(rewards, done):
        ret += float(r)
        if d:
            ep_returns.append(ret)
            ret = 0.0
    if ret != 0.0:
        ep_returns.append(ret)
    if not ep_returns:
        return 1.0
    rmin, rmax = min(ep_returns), max(ep_returns)
    span = rmax - rmin
    if span <= 0:
        return 1.0
    return float(max_episode_steps) / float(span)


class ReplayBuffer:
    """
    Real D4RL transition buffer loaded from .npz.

    All tensors live on GPU after construction — sample() just does index
    arithmetic on device, no host→device copy per batch. ~3-5x speedup on
    A40 for 1M-step IQL training.
    """

    def __init__(self, data_path: str, device: torch.device,
                 terminal_penalty: bool = True,
                 gpu_resident: bool = True,
                 reward_scale: float = 1.0,
                 reward_norm: str = "none"):
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

        # Standard CORL/IQL reward normalization (D4RL locomotion). Scales
        # per-step rewards by 1000/(max_ep_return-min_ep_return). Our raw
        # rewards (mean~2.4) are ~3× the scale the IQL paper's temperature
        # β=3 and expectile τ=0.7 were tuned for; without this the AWR
        # weight exp(β·adv) becomes too peaky → policy underfits.
        self.reward_norm = reward_norm
        self._corl_factor = 1.0
        if reward_norm == "corl":
            self._corl_factor = corl_reward_norm_factor(rewards, terminals, timeouts)
            rewards = rewards * self._corl_factor
            print(f"  CORL reward norm: ×{self._corl_factor:.4f}  "
                  f"(per-step mean→{rewards.mean():.3f})")
        elif reward_norm != "none":
            raise ValueError(f"unknown reward_norm={reward_norm!r}; "
                             f"expected 'none' or 'corl'")

        # PARS-style reward scaling. Multiplies the TD-target magnitude
        # which, combined with LayerNorm in the critic, sharpens Q-value
        # discrimination of OOD actions. Common values for D4RL MuJoCo:
        #   halfcheetah: 5     hopper: 10    walker2d: 10    ant: 5-10
        # Default 1.0 = no scaling, fully backward-compatible.
        self.reward_scale = float(reward_scale)
        if self.reward_scale != 1.0:
            rewards = rewards * self.reward_scale
            print(f"  reward scaling ×{self.reward_scale}")

        self.size = len(obs)

        # Stats stay as Python floats — used for syn-data normalization
        self.reward_mean = float(rewards.mean())
        self.reward_std  = float(rewards.std() + 1e-8)

        # Move all buffers to GPU once (or keep on CPU if memory is tight)
        if gpu_resident and device.type == "cuda":
            self.obs      = torch.from_numpy(obs).to(device)
            self.actions  = torch.from_numpy(actions).to(device)
            self.rewards  = torch.from_numpy(rewards).to(device)
            self.next_obs = torch.from_numpy(next_obs).to(device)
            self.done     = torch.from_numpy(done).to(device)
            self._on_gpu  = True
        else:
            self.obs      = obs
            self.actions  = actions
            self.rewards  = rewards
            self.next_obs = next_obs
            self.done     = done
            self._on_gpu  = False

        loc = "GPU" if self._on_gpu else "CPU"
        print(f"ReplayBuffer (real)  : {self.size:,} transitions  [{loc}]  "
              f"obs={obs.shape}  act={actions.shape}  "
              f"r=[{rewards.min():.2f}, {rewards.max():.2f}]  "
              f"mean={self.reward_mean:.3f}  std={self.reward_std:.3f}")

    def sample(self, batch_size: int) -> Dict[str, Tensor]:
        if self._on_gpu:
            idx = torch.randint(0, self.size, (batch_size,), device=self.device)
            return {
                "obs":      self.obs[idx],
                "action":   self.actions[idx],
                "reward":   self.rewards[idx],
                "next_obs": self.next_obs[idx],
                "done":     self.done[idx],
                # Source tag: 1.0 = real, 0.0 = synthetic. Used by SA-IQL.
                "source":   torch.ones(batch_size, device=self.device),
            }
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs":      torch.from_numpy(self.obs[idx]).to(self.device),
            "action":   torch.from_numpy(self.actions[idx]).to(self.device),
            "reward":   torch.from_numpy(self.rewards[idx]).to(self.device),
            "next_obs": torch.from_numpy(self.next_obs[idx]).to(self.device),
            "done":     torch.from_numpy(self.done[idx]).to(self.device),
            "source":   torch.ones(batch_size, device=self.device),
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
        gpu_resident:      bool  = True,
        reward_scale:      float = 1.0,
    ):
        self.device = device
        data = np.load(data_path, allow_pickle=True)

        obs      = data["observations"].astype(np.float32)
        actions  = data["actions"].astype(np.float32)
        rewards  = data["rewards"].astype(np.float32)
        next_obs = data["next_observations"].astype(np.float32)
        done     = data["terminals"].astype(np.float32)

        # Apply same reward scaling as the real buffer (must match!)
        # Reward normalization below will rescale to match real distribution,
        # so the absolute scale doesn't matter for syn — but if the
        # `normalize_rewards=False` path is used, we need consistent scale.
        if reward_scale != 1.0:
            rewards = rewards * reward_scale

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

        self.size = len(obs)

        # ── Step 3: Return-weighted sampling ──────────────────────────────
        # NOTE: weights computed BEFORE moving arrays to GPU (numpy code below)
        self._weights = None
        if return_weighting and self.size > 0:
            tmp_done    = done
            tmp_rewards = rewards
            self.done   = tmp_done
            self.rewards = tmp_rewards
            self._weights = self._compute_weights(temperature)

        # ── Move to GPU after weight computation ──────────────────────────
        if gpu_resident and device.type == "cuda":
            self.obs      = torch.from_numpy(obs).to(device)
            self.actions  = torch.from_numpy(actions).to(device)
            self.rewards  = torch.from_numpy(rewards).to(device)
            self.next_obs = torch.from_numpy(next_obs).to(device)
            self.done     = torch.from_numpy(done).to(device)
            self._on_gpu  = True
            # Pre-build a cumulative-prob tensor for fast torch.multinomial-style draw
            if self._weights is not None:
                self._weights_t = torch.from_numpy(self._weights).to(device)
            else:
                self._weights_t = None
        else:
            self.obs      = obs
            self.actions  = actions
            self.rewards  = rewards
            self.next_obs = next_obs
            self.done     = done
            self._on_gpu  = False
            self._weights_t = None

        loc = "GPU" if self._on_gpu else "CPU"
        rmin = float(self.rewards.min()) if not isinstance(self.rewards, np.ndarray) else self.rewards.min()
        rmax = float(self.rewards.max()) if not isinstance(self.rewards, np.ndarray) else self.rewards.max()
        rmean = float(self.rewards.mean()) if not isinstance(self.rewards, np.ndarray) else self.rewards.mean()
        print(f"SyntheticBuffer      : {self.size:,} transitions  [{loc}]  "
              f"r=[{rmin:.2f}, {rmax:.2f}]  mean={rmean:.3f}  "
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

        if self._on_gpu:
            if self._weights_t is not None:
                idx = torch.multinomial(self._weights_t, batch_size, replacement=True)
            else:
                idx = torch.randint(0, self.size, (batch_size,), device=self.device)
            return {
                "obs":      self.obs[idx],
                "action":   self.actions[idx],
                "reward":   self.rewards[idx],
                "next_obs": self.next_obs[idx],
                "done":     self.done[idx],
                "source":   torch.zeros(batch_size, device=self.device),   # 0 = syn
            }

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
            "source":   torch.zeros(batch_size, device=self.device),
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
        alpha_warmup:     int   = 0,         # NEW: steps of pure-real (alpha=1)
        alpha_ramp:       int   = 0,         # NEW: steps to ramp from 1.0 → alpha
    ):
        self.real      = real_buffer
        self.synthetic = synthetic_buffer
        self.alpha_target = float(np.clip(alpha, 0.0, 1.0))
        self.alpha     = self.alpha_target
        self.alpha_warmup = int(alpha_warmup)
        self.alpha_ramp   = int(alpha_ramp)
        self._step = 0

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
        self.alpha_target = float(np.clip(alpha, 0.0, 1.0))
        self.alpha = self.alpha_target

    def step(self) -> None:
        """Advance the warmup/ramp schedule. Call once per training step."""
        self._step += 1
        if self.synthetic is None or (self.alpha_warmup == 0 and self.alpha_ramp == 0):
            return  # no schedule, nothing to do
        if self._step <= self.alpha_warmup:
            self.alpha = 1.0
        elif self._step <= self.alpha_warmup + self.alpha_ramp:
            t = (self._step - self.alpha_warmup) / max(1, self.alpha_ramp)
            self.alpha = float(1.0 + t * (self.alpha_target - 1.0))
        else:
            self.alpha = self.alpha_target

    def sample(self, batch_size: int) -> Dict[str, Tensor]:
        if self.synthetic is None or self.alpha >= 1.0:
            return self.real.sample(batch_size)
        if self.alpha <= 0.0:
            return self.synthetic.sample(batch_size)

        # Clamp both halves to ≥1 so we always have something to sample.
        # Without this, rounding can produce n_real=batch_size, n_syn=0 when
        # alpha is just barely < 1.0 (e.g. 0.998 during alpha_warmup ramp),
        # and torch.multinomial refuses to sample 0 items.
        n_real = max(1, min(batch_size - 1, round(batch_size * self.alpha)))
        n_syn  = max(1, batch_size - n_real)

        real_batch = self.real.sample(n_real)
        syn_batch  = self.synthetic.sample(n_syn)

        # Concatenate every shared key. Both buffers must produce same keys;
        # `source` field is 1.0 for real / 0.0 for syn → preserved here.
        return {
            k: torch.cat([real_batch[k], syn_batch[k]], dim=0)
            for k in real_batch.keys() & syn_batch.keys()
        }

    def sample_real(self, batch_size: int) -> Dict[str, Tensor]:
        """Always returns real transitions — used for BC anchor."""
        return self.real.sample(batch_size)

    def __len__(self) -> int:
        return len(self.real)

    @property
    def real_size(self) -> int:
        return len(self.real)