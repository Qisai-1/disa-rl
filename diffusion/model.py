"""
TrajectoryDiT — diffusion transformer for RL trajectory generation.

Generates (obs, action) trajectories only — NO reward dimension.
Rewards are computed separately by reward_computer.py using analytic
functions or a learned MLP.

Key design:
  - RoPE positional encoding (relative temporal distance)
  - Multi-modal input embedding (obs and action projected separately)
  - Single output head for (obs, action) velocity
  - adaLN-Zero conditioning on flow time τ and return target R
  - Flash Attention via F.scaled_dot_product_attention
  - Pure float32 training (no autocast — avoids NaN with torch.compile)

Feature dim D = obs_dim + action_dim (23 for HalfCheetah)
"""

from __future__ import annotations
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# RoPE
# ──────────────────────────────────────────────────────────────────────────────

def build_rope_cache(seq_len, head_dim, theta=10_000.0, device=torch.device("cpu")):
    half  = head_dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, half, device=device).float() / half))
    pos   = torch.arange(seq_len, device=device).float()
    angles = torch.outer(pos, freqs)
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    x_even, x_odd = x[..., 0::2], x[..., 1::2]
    rot_even = x_even * cos - x_odd * sin
    rot_odd  = x_even * sin + x_odd * cos
    return torch.stack([rot_even, rot_odd], dim=-1).flatten(-2)


# ──────────────────────────────────────────────────────────────────────────────
# Attention
# ──────────────────────────────────────────────────────────────────────────────

class RoPEAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, max_seq_len=1024, rope_theta=10_000.0):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = hidden_size // num_heads
        self.qkv  = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.proj = nn.Linear(hidden_size, hidden_size)
        cos, sin = build_rope_cache(max_seq_len, self.head_dim, rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x):
        B, T, H = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q, k, v = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)
        cos, sin = self.rope_cos[:T], self.rope_sin[:T]
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.proj(out.transpose(1,2).reshape(B, T, H))


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def modulate(x, shift, scale):
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, freq_dim=256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_size), nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        nn.init.normal_(self.mlp[0].weight, std=0.02)
        nn.init.normal_(self.mlp[2].weight, std=0.02)

    def forward(self, tau):
        half  = self.freq_dim // 2
        freqs = torch.exp(-math.log(10_000) * torch.arange(half, device=tau.device) / half)
        args  = tau[:, None].float() * freqs[None]
        emb   = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(emb)


class ConditionEmbedder(nn.Module):
    def __init__(self, cond_dim, hidden_size):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(cond_dim, hidden_size), nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.null_emb = nn.Parameter(torch.zeros(hidden_size))
        nn.init.normal_(self.null_emb, std=0.02)

    def forward(self, cond, drop_mask=None):
        emb = self.proj(cond)
        if drop_mask is not None and drop_mask.any():
            null = self.null_emb.expand(emb.shape[0], -1)
            emb  = torch.where(drop_mask[:, None], null, emb)
        return emb


class MultiModalEmbedding(nn.Module):
    """Separate projections for obs and action — summed to hidden_size."""
    def __init__(self, obs_dim, action_dim, hidden_size):
        super().__init__()
        self.obs_dim    = obs_dim
        self.action_dim = action_dim
        self.obs_proj    = nn.Linear(obs_dim,    hidden_size)
        self.action_proj = nn.Linear(action_dim, hidden_size)
        nn.init.xavier_uniform_(self.obs_proj.weight,    gain=1.0)
        nn.init.xavier_uniform_(self.action_proj.weight, gain=1.0)

    def forward(self, x):
        obs    = x[..., :self.obs_dim]
        action = x[..., self.obs_dim:]
        return self.obs_proj(obs) + self.action_proj(action)


# ──────────────────────────────────────────────────────────────────────────────
# DiT block
# ──────────────────────────────────────────────────────────────────────────────

class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0,
                 max_seq_len=1024, rope_theta=10_000.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn  = RoPEAttention(hidden_size, num_heads, max_seq_len, rope_theta)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_dim    = int(hidden_size * mlp_ratio)
        self.mlp   = nn.Sequential(
            nn.Linear(hidden_size, mlp_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_dim, hidden_size),
        )
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6*hidden_size))
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x, c):
        s_a, sc_a, g_a, s_m, sc_m, g_m = self.adaLN(c).chunk(6, dim=1)
        x = x + g_a.unsqueeze(1) * self.attn(modulate(self.norm1(x), s_a, sc_a))
        x = x + g_m.unsqueeze(1) * self.mlp(modulate(self.norm2(x), s_m, sc_m))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_dim):
        super().__init__()
        self.norm   = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_dim)
        self.adaLN  = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2*hidden_size))
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x, c):
        shift, scale = self.adaLN(c).chunk(2, dim=1)
        return self.linear(modulate(self.norm(x), shift, scale))


# ──────────────────────────────────────────────────────────────────────────────
# Main model
# ──────────────────────────────────────────────────────────────────────────────

class TrajectoryDiT(nn.Module):
    """
    Conditional Flow Matching DiT for (obs, action) trajectory generation.

    NO reward dimension — rewards are computed by reward_computer.py.
    D = obs_dim + action_dim  (23 for HalfCheetah, 14 for Hopper, etc.)
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
    ):
        super().__init__()
        self.obs_dim           = obs_dim
        self.action_dim        = action_dim
        self.D                 = obs_dim + action_dim   # NO reward
        self.T                 = trajectory_length
        self.hidden_size       = hidden_size
        self.depth             = depth
        self.num_heads         = num_heads
        self.mlp_ratio         = mlp_ratio
        self.rope_theta        = rope_theta
        self.max_seq_len       = max_seq_len
        self.use_return_cond   = use_return_cond
        self.cfg_dropout_prob  = cfg_dropout_prob

        cond_dim = obs_dim + (1 if use_return_cond else 0)

        self.x_emb    = MultiModalEmbedding(obs_dim, action_dim, hidden_size)
        self.tau_emb  = TimestepEmbedder(hidden_size)
        self.cond_emb = ConditionEmbedder(cond_dim, hidden_size)
        self.blocks   = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio, max_seq_len, rope_theta)
            for _ in range(depth)
        ])
        self.head = FinalLayer(hidden_size, self.D)

        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"TrajectoryDiT | {n/1e6:.1f}M params | "
              f"depth={depth} hidden={hidden_size} heads={num_heads} "
              f"T={trajectory_length} D={self.D} (obs+action, no reward)")

    def forward(self, x_tau, tau, cond, drop_mask=None):
        """
        x_tau : (B, T, D)   noisy (obs, action) trajectory
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
        )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = TrajectoryDiT(obs_dim=17, action_dim=6, trajectory_length=100,
                            hidden_size=512, depth=8, num_heads=8).to(device)
    B, T, D = 4, 100, 23
    x   = torch.randn(B, T, D, device=device)
    tau = torch.rand(B, device=device)
    c   = torch.randn(B, 18, device=device)
    v   = model(x, tau, c)
    assert v.shape == (B, T, D)
    print(f"Output shape: {v.shape}  OK")