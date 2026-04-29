"""
MuJoCo evaluator for offline RL policies.

Rolls out the actor in the real environment and reports:
  - Raw episode return
  - D4RL normalized score (0 = random policy, 100 = expert policy)

D4RL normalization reference scores (from the original D4RL paper):
  These allow comparing across different environments on the same scale.
  Score = (agent_return - random_return) / (expert_return - random_return) * 100
"""

from __future__ import annotations
import numpy as np
import torch
from typing import Dict, Optional, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# D4RL reference scores for normalization
# Source: Fu et al. "D4RL: Datasets for Deep Data-Driven Reinforcement Learning"
# ──────────────────────────────────────────────────────────────────────────────

D4RL_REF_SCORES = {
    # (random_score, expert_score)
    "halfcheetah": (-280.178953, 12135.0),
    "hopper":      (20.272305,   3234.3),
    "walker2d":    (1.629008,    4592.3),
    "ant":         (-325.6,      3879.7),
}


def get_normalized_score(env_name: str, episode_return: float) -> float:
    """
    Compute D4RL normalized score.

    env_name: e.g. "halfcheetah-medium-v2" or just "halfcheetah"
    """
    # Extract base env name
    base = None
    for k in D4RL_REF_SCORES:
        if k in env_name.lower():
            base = k
            break
    if base is None:
        return episode_return   # Unknown env → return raw score

    random_score, expert_score = D4RL_REF_SCORES[base]
    score = (episode_return - random_score) / (expert_score - random_score) * 100.0
    return float(score)


# ──────────────────────────────────────────────────────────────────────────────
# Evaluator
# ──────────────────────────────────────────────────────────────────────────────

class Evaluator:
    """
    Rolls out a policy in the MuJoCo environment and reports normalized scores.

    Uses gymnasium (not the old gym) which is the maintained fork.
    Falls back gracefully if gymnasium is not installed — returns None.

    Parameters
    ----------
    env_name   : gymnasium env name, e.g. "HalfCheetah-v4"
    dataset_name : D4RL dataset name for score normalization
    n_episodes : number of evaluation episodes (10 is standard)
    device     : device for actor forward pass
    seed       : fixed seed for reproducible evaluation
    """

    def __init__(
        self,
        env_name:     str,
        dataset_name: str,
        n_episodes:   int           = 10,
        device:       torch.device  = torch.device("cpu"),
        seed:         int           = 0,
    ):
        self.env_name    = env_name
        self.dataset_name = dataset_name
        self.n_episodes  = n_episodes
        self.device      = device
        self.seed        = seed
        self._env        = None

        # Try to build env
        self._env = self._make_env()

    def _make_env(self):
        try:
            import gymnasium as gym
            if "ant" in self.env_name.lower() or "Ant" in self.env_name:
                env = gym.make(self.env_name, use_contact_forces=True)
            else:
                env = gym.make(self.env_name)
            print(f"Evaluator: {self.env_name}  obs={env.observation_space.shape}  "
                  f"act={env.action_space.shape}")
            return env
        except Exception as e:
            print(f"Warning: Could not create {self.env_name}: {e}")
            print("Evaluation will be skipped. Install: pip install gymnasium mujoco")
            return None

    @torch.no_grad()
    def evaluate(self, actor) -> Optional[Dict[str, float]]:
        """
        Roll out the actor for n_episodes and return metrics.

        Parameters
        ----------
        actor : GaussianActor with .act(obs_tensor) method

        Returns
        -------
        dict with:
            eval/return_mean    raw mean episode return
            eval/return_std     std of returns across episodes
            eval/normalized     D4RL normalized score
        or None if environment not available.
        """
        if self._env is None:
            return None

        actor.eval()
        returns = []

        for ep in range(self.n_episodes):
            obs, _ = self._env.reset(seed=self.seed + ep)
            ep_return = 0.0
            done = False

            while not done:
                obs_t  = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
                action = actor.act(obs_t, deterministic=True).squeeze(0).cpu().numpy()
                action = np.clip(action, -1.0, 1.0)

                obs, reward, terminated, truncated, _ = self._env.step(action)
                ep_return += reward
                done = terminated or truncated

            returns.append(ep_return)

        actor.train()

        mean_return = float(np.mean(returns))
        std_return  = float(np.std(returns))
        norm_score  = get_normalized_score(self.dataset_name, mean_return)

        return {
            "eval/return_mean":  mean_return,
            "eval/return_std":   std_return,
            "eval/normalized":   norm_score,
        }

    def close(self) -> None:
        if self._env is not None:
            self._env.close()


# ──────────────────────────────────────────────────────────────────────────────
# Environment name mapping
# ──────────────────────────────────────────────────────────────────────────────

# Maps D4RL dataset names to gymnasium env names
DATASET_TO_GYM = {
    "halfcheetah-medium-v2":        "HalfCheetah-v4",
    "halfcheetah-medium-replay-v2": "HalfCheetah-v4",
    "halfcheetah-expert-v2":        "HalfCheetah-v4",
    "halfcheetah-random-v2":        "HalfCheetah-v4",
    "hopper-medium-v2":             "Hopper-v4",
    "hopper-medium-replay-v2":      "Hopper-v4",
    "hopper-expert-v2":             "Hopper-v4",
    "walker2d-medium-v2":           "Walker2d-v4",
    "walker2d-medium-replay-v2":    "Walker2d-v4",
    "walker2d-expert-v2":           "Walker2d-v4",
    "ant-medium-v2":                "Ant-v4",
    "ant-medium-replay-v2":         "Ant-v4",
}


def make_evaluator(
    dataset_name: str,
    device:       torch.device,
    n_episodes:   int = 10,
) -> Evaluator:
    gym_name = DATASET_TO_GYM.get(dataset_name, dataset_name)
    return Evaluator(
        env_name     = gym_name,
        dataset_name = dataset_name,
        n_episodes   = n_episodes,
        device       = device,
    )
