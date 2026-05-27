"""
v2 diffusion model — joint (obs, action, reward) generation.

Differences vs model.py (v1):
  - Output dim D = obs_dim + action_dim + 1   (was obs_dim + action_dim).
  - New MultiModalEmbeddingV2 splits the input into 3 modalities (obs,
    action, reward) and projects each separately, mirroring v1's 2-modality
    pattern.
  - Reward stats are normalised per-channel by data_v2.DataNormalizer
    BEFORE this model sees them — so reward enters as a roughly-unit-variance
    extra channel, on the same scale as obs/action.

Reuses v1's RoPE, DiTBlock, FinalLayer, TimestepEmbedder, ConditionEmbedder.

Design rationale:
  - Equal-weight loss on the 3 channels (handled in flow_matching_v2.py).
  - Backwards-compatible state dicts: NOT loadable from v1 weights (output
    dim differs). v2 trains from scratch.
"""

from __future__ import annotations
import torch
import torch.nn as nn

# Reuse v1 building blocks
from model import (
    RoPEAttention,
    DiTBlock,
    FinalLayer,
    TimestepEmbedder,
    ConditionEmbedder,
    modulate,
)


# ──────────────────────────────────────────────────────────────────────────────
# 3-modality input embedding
# ──────────────────────────────────────────────────────────────────────────────

class MultiModalEmbeddingV2(nn.Module):
    """
    Separate projections for obs, action, reward — summed to hidden_size.
    Reward is a single scalar channel (dim=1) per timestep.
    """
    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int):
        super().__init__()
        self.obs_dim    = obs_dim
        self.action_dim = action_dim
        self.obs_proj    = nn.Linear(obs_dim,    hidden_size)
        self.action_proj = nn.Linear(action_dim, hidden_size)
        self.reward_proj = nn.Linear(1,          hidden_size)
        nn.init.xavier_uniform_(self.obs_proj.weight,    gain=1.0)
        nn.init.xavier_uniform_(self.action_proj.weight, gain=1.0)
        nn.init.xavier_uniform_(self.reward_proj.weight, gain=1.0)

    def forward(self, x):
        # x: (B, T, obs_dim + action_dim + 1)
        o = x[..., :self.obs_dim]
        a = x[..., self.obs_dim:self.obs_dim + self.action_dim]
        r = x[..., self.obs_dim + self.action_dim:self.obs_dim + self.action_dim + 1]
        return self.obs_proj(o) + self.action_proj(a) + self.reward_proj(r)


# ──────────────────────────────────────────────────────────────────────────────
# v2 main model
# ──────────────────────────────────────────────────────────────────────────────

class TrajectoryDiTV2(nn.Module):
    """
    v2 — Conditional Flow Matching DiT for (obs, action, reward) trajectory
    generation.

    D = obs_dim + action_dim + 1   (24 for HalfCheetah, 15 for Hopper, etc.)
    """

    def __init__(
        self,
        obs_dim:           int   = 17,
        action_dim:        int   = 6,
        trajectory_length: int   = 100,
        hidden_size:       int   = 512,
        depth:             int   = 8,
        num_heads:         int   = 8,
        mlp_ratio:         float = 4.0,
        rope_theta:        float = 10_000.0,
        max_seq_len:       int   = 1_024,
        use_return_cond:   bool  = True,
        cfg_dropout_prob:  float = 0.10,
        mlp_dropout:       float = 0.15,
    ):
        super().__init__()
        self.obs_dim           = obs_dim
        self.action_dim        = action_dim
        self.reward_dim        = 1
        self.D                 = obs_dim + action_dim + 1   # +1 reward
        self.T                 = trajectory_length
        self.hidden_size       = hidden_size
        self.depth             = depth
        self.num_heads         = num_heads
        self.mlp_ratio         = mlp_ratio
        self.rope_theta        = rope_theta
        self.max_seq_len       = max_seq_len
        self.use_return_cond   = use_return_cond
        self.cfg_dropout_prob  = cfg_dropout_prob
        self.mlp_dropout       = mlp_dropout

        cond_dim = obs_dim + (1 if use_return_cond else 0)

        self.x_emb    = MultiModalEmbeddingV2(obs_dim, action_dim, hidden_size)
        self.tau_emb  = TimestepEmbedder(hidden_size)
        self.cond_emb = ConditionEmbedder(cond_dim, hidden_size)
        self.blocks   = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio, max_seq_len, rope_theta, mlp_dropout)
            for _ in range(depth)
        ])
        self.head = FinalLayer(hidden_size, self.D)

        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"TrajectoryDiTV2 | {n/1e6:.1f}M params | "
              f"depth={depth} hidden={hidden_size} heads={num_heads} "
              f"T={trajectory_length} D={self.D} (obs+action+reward)")

    def forward(self, x_tau, tau, cond, drop_mask=None):
        """
        x_tau : (B, T, D)   noisy (obs, action, reward) trajectory
        tau   : (B,)         flow time
        cond  : (B, cond_dim)
        Returns: (B, T, D)  velocity field
        """
        x = self.x_emb(x_tau)
        c = self.tau_emb(tau) + self.cond_emb(cond, drop_mask)
        for block in self.blocks:
            x = block(x, c)
        return self.head(x, c)

    def config_dict(self):
        return dict(
            obs_dim           = self.obs_dim,
            action_dim        = self.action_dim,
            trajectory_length = self.T,
            hidden_size       = self.hidden_size,
            depth             = self.depth,
            num_heads         = self.num_heads,
            mlp_ratio         = self.mlp_ratio,
            rope_theta        = self.rope_theta,
            max_seq_len       = self.max_seq_len,
            use_return_cond   = self.use_return_cond,
            cfg_dropout_prob  = self.cfg_dropout_prob,
            mlp_dropout       = self.mlp_dropout,
            arch_version      = "v2",
            includes_reward   = True,
        )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # v2 default config: bigger model (512/8/8), reward channel.
    model = TrajectoryDiTV2(obs_dim=17, action_dim=6, trajectory_length=100,
                            hidden_size=512, depth=8, num_heads=8,
                            mlp_dropout=0.15).to(device)
    B, T, D = 4, 100, 24    # 17 obs + 6 action + 1 reward
    x   = torch.randn(B, T, D, device=device)
    tau = torch.rand(B, device=device)
    c   = torch.randn(B, 18, device=device)  # s0(17) + return(1)
    v   = model(x, tau, c)
    assert v.shape == (B, T, D), f"Expected ({B},{T},{D}), got {v.shape}"
    print(f"v2 forward OK | output shape: {v.shape}")
