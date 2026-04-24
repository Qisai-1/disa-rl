"""
Neural networks for IQL (Implicit Q-Learning).

Architecture choices for NeurIPS-quality results:
  - LayerNorm after every hidden layer (critical for offline RL stability —
    without it Q-values diverge on OOD actions in the D4RL dataset)
  - Twin Q-networks (take min at target, prevents overestimation)
  - Gaussian actor with tanh squashing (matches D4RL action space [-1, 1])
  - Orthogonal weight initialisation (better gradient flow than Xavier for RL)

Reference: Kostrikov et al. "Offline Reinforcement Learning with
Implicit Q-Learning" (ICLR 2022).
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple


# ──────────────────────────────────────────────────────────────────────────────
# Shared MLP backbone
# ──────────────────────────────────────────────────────────────────────────────

def build_mlp(
    input_dim:   int,
    output_dim:  int,
    hidden_dims: Tuple[int, ...] = (256, 256),
    use_ln:      bool = True,
    activate_last: bool = False,
) -> nn.Sequential:
    """
    Standard MLP with optional LayerNorm.

    LayerNorm is applied AFTER the activation (post-norm) which is more
    stable for offline RL than pre-norm.  The final layer never has LN.
    """
    layers = []
    dims   = (input_dim,) + hidden_dims

    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if use_ln:
            layers.append(nn.LayerNorm(dims[i + 1]))
        layers.append(nn.ReLU())

    layers.append(nn.Linear(dims[-1], output_dim))
    if activate_last:
        layers.append(nn.ReLU())

    return nn.Sequential(*layers)


def orthogonal_init(module: nn.Module, gain: float = math.sqrt(2)) -> nn.Module:
    """Apply orthogonal initialisation to all Linear layers."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=gain)
            nn.init.zeros_(m.bias)
    return module


# ──────────────────────────────────────────────────────────────────────────────
# Value network  V(s)
# ──────────────────────────────────────────────────────────────────────────────

class ValueNetwork(nn.Module):
    """
    State value function V(s).
    Trained with expectile regression on Q(s,a) targets.
    Output is a scalar — no activation on final layer.
    """

    def __init__(
        self,
        obs_dim:     int,
        hidden_dims: Tuple[int, ...] = (256, 256),
    ):
        super().__init__()
        self.net = build_mlp(obs_dim, 1, hidden_dims, use_ln=True)
        orthogonal_init(self.net)

    def forward(self, obs: Tensor) -> Tensor:
        """obs: (B, obs_dim) → (B,)"""
        return self.net(obs).squeeze(-1)


# ──────────────────────────────────────────────────────────────────────────────
# Q-network  Q(s, a)
# ──────────────────────────────────────────────────────────────────────────────

class QNetwork(nn.Module):
    """
    State-action value function Q(s, a).
    Takes (obs, action) concatenated as input.
    We instantiate two of these for the twin-Q trick.
    """

    def __init__(
        self,
        obs_dim:     int,
        action_dim:  int,
        hidden_dims: Tuple[int, ...] = (256, 256),
    ):
        super().__init__()
        self.net = build_mlp(obs_dim + action_dim, 1, hidden_dims, use_ln=True)
        orthogonal_init(self.net)

    def forward(self, obs: Tensor, action: Tensor) -> Tensor:
        """obs: (B, obs_dim), action: (B, act_dim) → (B,)"""
        x = torch.cat([obs, action], dim=-1)
        return self.net(x).squeeze(-1)


class TwinQNetwork(nn.Module):
    """
    Two independent Q-networks.  Use min(Q1, Q2) as the target to prevent
    overestimation — standard practice in off-policy RL since SAC.
    """

    def __init__(
        self,
        obs_dim:     int,
        action_dim:  int,
        hidden_dims: Tuple[int, ...] = (256, 256),
    ):
        super().__init__()
        self.q1 = QNetwork(obs_dim, action_dim, hidden_dims)
        self.q2 = QNetwork(obs_dim, action_dim, hidden_dims)

    def forward(self, obs: Tensor, action: Tensor) -> Tuple[Tensor, Tensor]:
        """Returns (q1, q2) each (B,)"""
        return self.q1(obs, action), self.q2(obs, action)

    def min(self, obs: Tensor, action: Tensor) -> Tensor:
        """min(Q1, Q2) — used for target computation"""
        q1, q2 = self.forward(obs, action)
        return torch.min(q1, q2)


# ──────────────────────────────────────────────────────────────────────────────
# Actor  π(a | s)
# ──────────────────────────────────────────────────────────────────────────────

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


class GaussianActor(nn.Module):
    """
    Diagonal Gaussian policy with tanh squashing.

    Output: mean and log_std → reparameterised sample → tanh → action in [-1, 1]

    The tanh squashing is important because D4RL environments clip actions
    to [-1, 1].  Without squashing the actor learns to predict actions
    slightly outside this range and the evaluator clips them, causing a
    subtle mismatch between training and evaluation.

    Log-std is clamped to [LOG_STD_MIN, LOG_STD_MAX] for numerical stability.
    """

    def __init__(
        self,
        obs_dim:     int,
        action_dim:  int,
        hidden_dims: Tuple[int, ...] = (256, 256),
    ):
        super().__init__()
        self.action_dim = action_dim
        self.trunk = build_mlp(obs_dim, hidden_dims[-1], hidden_dims[:-1], use_ln=True, activate_last=True)
        self.mean_head    = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_head = nn.Linear(hidden_dims[-1], action_dim)

        orthogonal_init(self.trunk)
        # Smaller init for output heads → better initial exploration
        nn.init.orthogonal_(self.mean_head.weight,    gain=0.01)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.orthogonal_(self.log_std_head.weight, gain=0.01)
        nn.init.zeros_(self.log_std_head.bias)

    def forward(self, obs: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Returns (mean, log_std) in pre-squash space.
        obs: (B, obs_dim) → mean: (B, act_dim), log_std: (B, act_dim)
        """
        h       = self.trunk(obs)
        mean    = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def get_action(
        self,
        obs:         Tensor,
        deterministic: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """
        Sample an action and compute its log-probability.

        Returns
        -------
        action   : (B, act_dim) in [-1, 1] after tanh squashing
        log_prob : (B,) log π(a|s) with tanh correction applied
        """
        mean, log_std = self.forward(obs)
        std = log_std.exp()

        if deterministic:
            # Deterministic = mean of the Gaussian, then squash
            action   = torch.tanh(mean)
            log_prob = torch.zeros(obs.shape[0], device=obs.device)
            return action, log_prob

        # Reparameterisation trick
        eps     = torch.randn_like(mean)
        x_t     = mean + std * eps          # pre-squash sample
        action  = torch.tanh(x_t)

        # Log-prob with tanh correction:
        #   log π(a|s) = log N(x_t; μ, σ) - Σ log(1 - tanh²(x_t))
        log_prob = (
            -0.5 * ((x_t - mean) / std).pow(2)
            - log_std
            - 0.5 * math.log(2 * math.pi)
        ).sum(dim=-1)
        # Tanh correction (numerically stable version)
        log_prob -= (2.0 * (math.log(2) - x_t - F.softplus(-2 * x_t))).sum(dim=-1)

        return action, log_prob

    def log_prob(self, obs: Tensor, action: Tensor) -> Tensor:
        """
        Compute log π(action | obs) for given (obs, action) pairs.
        Used in AWR actor update — evaluates log prob of DATASET actions.

        obs    : (B, obs_dim)
        action : (B, action_dim) — actions from the dataset (already in [-1,1])
        Returns: (B,) log probabilities
        """
        mean, log_std = self.forward(obs)
        std = log_std.exp()

        # Inverse tanh to get pre-squash values
        # action = tanh(x) → x = atanh(action)
        action_clamped = action.clamp(-1 + 1e-6, 1 - 1e-6)
        x_t = torch.atanh(action_clamped)

        # Log prob of Gaussian at x_t
        log_prob = (
            -0.5 * ((x_t - mean) / std).pow(2)
            - log_std
            - 0.5 * math.log(2 * math.pi)
        ).sum(dim=-1)

        # Tanh correction
        log_prob -= (2.0 * (math.log(2) - x_t - F.softplus(-2 * x_t))).sum(dim=-1)
        return log_prob

    @torch.no_grad()
    def act(self, obs: Tensor, deterministic: bool = True) -> Tensor:
        """Single action for environment interaction (no grad, deterministic by default)."""
        action, _ = self.get_action(obs, deterministic=deterministic)
        return action


# ──────────────────────────────────────────────────────────────────────────────
# EMA target network helper
# ──────────────────────────────────────────────────────────────────────────────

class EMATarget:
    """
    Exponential moving average of a network's parameters.
    Used for the target Q-network in IQL.
    """

    def __init__(self, source: nn.Module, tau: float = 0.005):
        import copy
        self.tau    = tau
        self.target = copy.deepcopy(source)
        for p in self.target.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, source: nn.Module) -> None:
        for t, s in zip(self.target.parameters(), source.parameters()):
            t.data.mul_(1.0 - self.tau).add_(s.data, alpha=self.tau)

    def __call__(self, *args, **kwargs):
        return self.target(*args, **kwargs)

    def to(self, device):
        self.target.to(device)
        return self


# ──────────────────────────────────────────────────────────────────────────────
# Smoke test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    obs_dim, action_dim, B = 17, 6, 32

    obs    = torch.randn(B, obs_dim)
    action = torch.randn(B, action_dim)

    v  = ValueNetwork(obs_dim)
    q  = TwinQNetwork(obs_dim, action_dim)
    pi = GaussianActor(obs_dim, action_dim)

    v_out          = v(obs)
    q1_out, q2_out = q(obs, action)
    a, lp          = pi.get_action(obs)

    assert v_out.shape          == (B,)
    assert q1_out.shape         == (B,)
    assert q2_out.shape         == (B,)
    assert a.shape              == (B, action_dim)
    assert lp.shape             == (B,)
    assert a.abs().max().item() <= 1.0 + 1e-5, "actions not in [-1, 1]"

    # Test deterministic
    a_det, _ = pi.get_action(obs, deterministic=True)
    assert a_det.abs().max().item() <= 1.0 + 1e-5

    print(f"V:      {v_out.shape}     mean={v_out.mean():.3f}")
    print(f"Q1, Q2: {q1_out.shape}   min(Q)={q.min(obs, action).mean():.3f}")
    print(f"Actor:  {a.shape}  log_prob mean={lp.mean():.3f}")
    print("All network tests passed.")