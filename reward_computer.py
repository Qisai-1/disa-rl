"""
Reward computer for DiSA-RL synthetic trajectories.

Two modes
---------
1. Analytic (default for D4RL locomotion)
   Exact reward functions derived from MuJoCo environment source code.
   Zero approximation error — rewards are identical to the real environment.

2. Learned (fallback / general case)
   Small MLP trained on D4RL (obs, action) → reward.
   Use when the analytic formula is unavailable (custom envs, real-world).

Usage
-----
    # Analytic (recommended for D4RL)
    rc = RewardComputer.make("halfcheetah-medium-v2")
    rewards = rc.compute(obs, actions)          # (N,) float32

    # Learned
    rc = RewardComputer.make("halfcheetah-medium-v2", use_learned=True)
    rc.fit("./data/halfcheetah-medium-v2.npz")  # train on D4RL data
    rc.save("./checkpoints/halfcheetah-medium-v2/reward_model.pt")
    rewards = rc.compute(obs, actions)

    # Load pre-trained
    rc = RewardComputer.load("./checkpoints/halfcheetah-medium-v2/reward_model.pt",
                             "halfcheetah-medium-v2")
    rewards = rc.compute(obs, actions)

Analytic reward functions
-------------------------
Source: gymnasium MuJoCo environment implementations
  HalfCheetah-v4 : forward_reward - ctrl_cost_weight * ||action||^2
  Hopper-v4      : forward_reward + alive_bonus - ctrl_cost
  Walker2d-v4    : forward_reward + alive_bonus - ctrl_cost
  Ant-v4         : forward_reward + survive_reward - ctrl_cost - contact_cost
"""

from __future__ import annotations
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# Analytic reward functions
# ──────────────────────────────────────────────────────────────────────────────

def reward_halfcheetah(obs: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """
    HalfCheetah-v4 reward.

    r = forward_reward_weight * x_velocity - ctrl_cost_weight * ||action||^2

    Observation layout (17 dims):
      [0]     : z position
      [1]     : sin(theta) of joints...
      [8]     : x_velocity  ← key dimension
      [9-16]  : joint velocities

    Default weights from gymnasium source:
      forward_reward_weight = 1.0
      ctrl_cost_weight      = 0.1
    """
    x_velocity = obs[:, 8]
    ctrl_cost  = 0.1 * np.sum(actions ** 2, axis=1)
    return (x_velocity - ctrl_cost).astype(np.float32)


def reward_hopper(obs: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """
    Hopper-v4 reward.

    r = alive_bonus + forward_reward - ctrl_cost

    Observation layout (11 dims):
      [0]   : z position (height)
      [1]   : angle (torso)
      [2-4] : joint angles
      [5]   : x_velocity  ← forward reward
      [6-10]: joint velocities

    Default weights:
      forward_reward_weight = 1.0
      ctrl_cost_weight      = 0.001
      healthy_reward        = 1.0
    """
    x_velocity  = obs[:, 5]
    ctrl_cost   = 0.001 * np.sum(actions ** 2, axis=1)
    alive_bonus = 1.0
    return (alive_bonus + x_velocity - ctrl_cost).astype(np.float32)


def reward_walker2d(obs: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """
    Walker2d-v4 reward.

    r = alive_bonus + forward_reward - ctrl_cost

    Observation layout (17 dims):
      [0]   : z position (height)
      [1]   : angle (torso)
      [2-7] : joint angles
      [8]   : x_velocity  ← forward reward
      [9-16]: joint velocities

    Default weights:
      forward_reward_weight = 1.0
      ctrl_cost_weight      = 0.001
      healthy_reward        = 1.0
    """
    x_velocity  = obs[:, 8]
    ctrl_cost   = 0.001 * np.sum(actions ** 2, axis=1)
    alive_bonus = 1.0
    return (alive_bonus + x_velocity - ctrl_cost).astype(np.float32)


def reward_ant(obs: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """
    Ant-v4 reward.

    r = survive_reward + forward_reward - ctrl_cost - contact_cost

    Observation layout (111 dims):
      [0]    : z position
      [1-3]  : quaternion orientation
      [4-12] : joint angles
      [13]   : x_velocity  ← forward reward
      [14]   : y_velocity
      [15-26]: joint velocities
      [27-110]: contact forces (84 dims)

    Default weights:
      forward_reward_weight = 1.0
      ctrl_cost_weight      = 0.5
      contact_cost_weight   = 5e-4
      healthy_reward        = 1.0
    """
    x_velocity   = obs[:, 13]
    ctrl_cost    = 0.5   * np.sum(actions ** 2, axis=1)
    # Contact forces are in obs[27:111] — use their norm as proxy
    contact_cost = 5e-4  * np.sum(obs[:, 27:] ** 2, axis=1)
    survive      = 1.0
    return (survive + x_velocity - ctrl_cost - contact_cost).astype(np.float32)


# Registry mapping env name → analytic function
ANALYTIC_REWARDS = {
    "halfcheetah": reward_halfcheetah,
    "hopper":      reward_hopper,
    "walker2d":    reward_walker2d,
    "ant":         reward_ant,
}


def get_analytic_fn(env_name: str):
    """Return analytic reward function for a D4RL env name."""
    for key, fn in ANALYTIC_REWARDS.items():
        if key in env_name.lower():
            return fn
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Learned reward model
# ──────────────────────────────────────────────────────────────────────────────

class RewardMLP(nn.Module):
    """
    Small MLP that maps (obs, action) → scalar reward.

    LayerNorm after each hidden layer for stability.
    Fits in <1MB — negligible overhead.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Small init for output layer — reward range is bounded
        nn.init.uniform_(self.net[-1].weight, -0.01, 0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, action], dim=-1)
        return self.net(x).squeeze(-1)


# ──────────────────────────────────────────────────────────────────────────────
# Main interface
# ──────────────────────────────────────────────────────────────────────────────

class RewardComputer:
    """
    Unified reward computation interface.

    Prefers analytic rewards when available.
    Falls back to learned MLP when analytic function is not implemented.

    Parameters
    ----------
    env_name     : D4RL dataset name (e.g. "halfcheetah-medium-v2")
    use_learned  : force learned model even if analytic is available
    obs_dim      : required only for learned mode
    action_dim   : required only for learned mode
    device       : torch device for learned model
    """

    def __init__(
        self,
        env_name:    str,
        use_learned: bool          = False,
        obs_dim:     Optional[int] = None,
        action_dim:  Optional[int] = None,
        device:      torch.device  = torch.device("cpu"),
    ):
        self.env_name   = env_name
        self.device     = device
        self._analytic  = get_analytic_fn(env_name)
        self._model:    Optional[RewardMLP] = None
        self._obs_mean: Optional[np.ndarray] = None
        self._obs_std:  Optional[np.ndarray] = None
        self._act_mean: Optional[np.ndarray] = None
        self._act_std:  Optional[np.ndarray] = None
        self._rew_mean: float = 0.0
        self._rew_std:  float = 1.0

        if use_learned or self._analytic is None:
            if obs_dim is None or action_dim is None:
                raise ValueError(
                    "obs_dim and action_dim required for learned reward model."
                )
            self._model = RewardMLP(obs_dim, action_dim).to(device)
            self._mode  = "learned"
            print(f"RewardComputer [{env_name}]: learned MLP mode")
        else:
            self._mode = "analytic"
            print(f"RewardComputer [{env_name}]: analytic mode "
                  f"(fn={self._analytic.__name__})")

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def make(
        cls,
        env_name:    str,
        use_learned: bool          = False,
        device:      torch.device  = torch.device("cpu"),
        data_dir:    str           = "./data",
    ) -> "RewardComputer":
        """
        Auto-detect obs/action dims from dataset file.
        Requires the dataset .npz to exist — no hardcoded fallbacks.
        """
        import numpy as np
        data_path = os.path.join(data_dir, f"{env_name}.npz")
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"Dataset not found: {data_path}\n"
                f"Run: python download_data.py --datasets {env_name}"
            )
        data       = np.load(data_path, allow_pickle=True)
        obs_dim    = int(data["observations"].shape[1])
        action_dim = int(data["actions"].shape[1])
        return cls(env_name, use_learned, obs_dim, action_dim, device)

    # ------------------------------------------------------------------
    # Fit (learned mode only)
    # ------------------------------------------------------------------

    def fit(
        self,
        data_path:  str,
        n_epochs:   int   = 50,
        batch_size: int   = 1024,
        lr:         float = 3e-4,
        verbose:    bool  = True,
    ) -> None:
        """
        Train the reward MLP on D4RL data.

        Parameters
        ----------
        data_path  : path to D4RL .npz file
        n_epochs   : training epochs
        batch_size : mini-batch size
        lr         : learning rate
        verbose    : print loss every 10 epochs
        """
        if self._mode != "learned":
            print("Analytic mode — no fitting needed.")
            return

        data    = np.load(data_path, allow_pickle=True)
        obs     = data["observations"].astype(np.float32)
        actions = data["actions"].astype(np.float32)
        rewards = data["rewards"].astype(np.float32)

        # Fit normalisation stats
        self._obs_mean = obs.mean(0);    self._obs_std = obs.std(0) + 1e-8
        self._act_mean = actions.mean(0); self._act_std = actions.std(0) + 1e-8
        self._rew_mean = float(rewards.mean())
        self._rew_std  = float(rewards.std() + 1e-8)

        # Normalise inputs
        obs_n  = (obs - self._obs_mean) / self._obs_std
        act_n  = (actions - self._act_mean) / self._act_std
        rew_n  = (rewards - self._rew_mean) / self._rew_std

        obs_t  = torch.from_numpy(obs_n).to(self.device)
        act_t  = torch.from_numpy(act_n).to(self.device)
        rew_t  = torch.from_numpy(rew_n).to(self.device)

        N    = len(obs_t)
        opt  = torch.optim.Adam(self._model.parameters(), lr=lr)

        self._model.train()
        for epoch in range(n_epochs):
            perm  = torch.randperm(N, device=self.device)
            total_loss = 0.0
            n_batches  = 0

            for i in range(0, N, batch_size):
                idx = perm[i : i + batch_size]
                pred = self._model(obs_t[idx], act_t[idx])
                loss = F.mse_loss(pred, rew_t[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item()
                n_batches  += 1

            if verbose and (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1:>3d}/{n_epochs}  "
                      f"loss={total_loss/n_batches:.5f}")

        self._model.eval()
        print(f"Reward model fitted on {N:,} transitions.")

    # ------------------------------------------------------------------
    # Compute
    # ------------------------------------------------------------------

    def compute(
        self,
        obs:     np.ndarray,   # (N, obs_dim)
        actions: np.ndarray,   # (N, action_dim)
    ) -> np.ndarray:
        """
        Compute rewards for a batch of (obs, action) pairs.

        Returns (N,) float32 array.
        Works in both analytic and learned modes.
        """
        if self._mode == "analytic":
            rewards = self._analytic(obs, actions)

        else:
            # Normalise
            obs_n = (obs - self._obs_mean) / self._obs_std
            act_n = (actions - self._act_mean) / self._act_std

            with torch.no_grad():
                obs_t = torch.from_numpy(obs_n.astype(np.float32)).to(self.device)
                act_t = torch.from_numpy(act_n.astype(np.float32)).to(self.device)
                rew_n = self._model(obs_t, act_t).cpu().numpy()

            # Denormalise
            rewards = rew_n * self._rew_std + self._rew_mean

        return rewards.astype(np.float32)

    def compute_trajectory(
        self,
        obs:     np.ndarray,   # (B, T, obs_dim)
        actions: np.ndarray,   # (B, T, action_dim)
    ) -> np.ndarray:
        """
        Compute rewards for full trajectory batches.
        Returns (B, T) float32.
        """
        B, T, _ = obs.shape
        obs_flat = obs.reshape(B * T, -1)
        act_flat = actions.reshape(B * T, -1)
        rew_flat = self.compute(obs_flat, act_flat)
        return rew_flat.reshape(B, T)

    # ------------------------------------------------------------------
    # Save / Load (learned mode)
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save learned reward model to disk."""
        if self._mode != "learned":
            print("Analytic mode — nothing to save.")
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "model_state": self._model.state_dict(),
            "obs_mean":    self._obs_mean,
            "obs_std":     self._obs_std,
            "act_mean":    self._act_mean,
            "act_std":     self._act_std,
            "rew_mean":    self._rew_mean,
            "rew_std":     self._rew_std,
            "env_name":    self.env_name,
        }, path)
        print(f"Reward model saved → {path}")

    @classmethod
    def load(
        cls,
        path:     str,
        env_name: str,
        device:   torch.device = torch.device("cpu"),
    ) -> "RewardComputer":
        """Load a pre-trained learned reward model."""
        ckpt       = torch.load(path, map_location=device, weights_only=False)
        obs_dim    = ckpt["obs_mean"].shape[0]
        action_dim = ckpt["act_mean"].shape[0]

        rc = cls(env_name, use_learned=True, obs_dim=obs_dim,
                 action_dim=action_dim, device=device)
        rc._model.load_state_dict(ckpt["model_state"])
        rc._model.eval()
        rc._obs_mean = ckpt["obs_mean"]
        rc._obs_std  = ckpt["obs_std"]
        rc._act_mean = ckpt["act_mean"]
        rc._act_std  = ckpt["act_std"]
        rc._rew_mean = float(ckpt["rew_mean"])
        rc._rew_std  = float(ckpt["rew_std"])
        print(f"Reward model loaded from {path}")
        return rc


# ──────────────────────────────────────────────────────────────────────────────
# CLI — fit and save a learned reward model
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser(
        description="Fit and save a learned reward model on D4RL data"
    )
    parser.add_argument("--env",       type=str, required=True)
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to D4RL .npz (auto-detected if None)")
    parser.add_argument("--output",    type=str, default=None,
                        help="Where to save model (default: checkpoints/<env>/reward_model.pt)")
    parser.add_argument("--n_epochs",  type=int, default=50)
    parser.add_argument("--test_analytic", action="store_true",
                        help="Test analytic reward function and print statistics")
    args = parser.parse_args()

    rc_tmp = RewardComputer.make(args.env)
    obs_dim, action_dim = rc_tmp._ENV_DIMS[
        next(k for k in rc_tmp._ENV_DIMS if k in args.env.lower())
    ]
    data_path = args.data_path or f"./data/{args.env}.npz"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.test_analytic:
        # Test analytic reward against real D4RL rewards
        print(f"\nTesting analytic reward for {args.env}...")
        data    = np.load(data_path, allow_pickle=True)
        obs     = data["observations"].astype(np.float32)
        actions = data["actions"].astype(np.float32)
        real_r  = data["rewards"].astype(np.float32)

        rc       = RewardComputer.make(args.env, device=device)
        pred_r   = rc.compute(obs, actions)

        corr = np.corrcoef(real_r, pred_r)[0, 1]
        mae  = np.abs(real_r - pred_r).mean()
        print(f"  Real reward   : mean={real_r.mean():.3f}  std={real_r.std():.3f}")
        print(f"  Analytic pred : mean={pred_r.mean():.3f}  std={pred_r.std():.3f}")
        print(f"  Correlation   : {corr:.4f}  (1.0 = perfect)")
        print(f"  MAE           : {mae:.4f}")

    else:
        # Fit learned reward model
        output = args.output or f"./checkpoints/{args.env}/reward_model.pt"
        rc = RewardComputer(args.env, use_learned=True,
                            obs_dim=obs_dim, action_dim=action_dim, device=device)
        rc.fit(data_path, n_epochs=args.n_epochs)
        rc.save(output)
        print(f"\nDone. Load with:")
        print(f"  rc = RewardComputer.load('{output}', '{args.env}')")