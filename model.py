"""
TrajectoryDiT — improved diffusion transformer for RL trajectory generation.

Key upgrades over the baseline:

1. RoPE (Rotary Position Embeddings)
   Applied to Q and K inside every attention block.
   Encodes *relative* temporal distance rather than absolute position,
   which is the right inductive bias for trajectories (gait patterns
   repeat at different positions; what matters is dt, not t).
   Uses F.scaled_dot_product_attention which dispatches to Flash Attention
   automatically when the CUDA kernel is available — ~2-3x faster than
   nn.MultiheadAttention for T=100.

2. Multi-modal input embedding
   obs, action, reward are projected separately then summed.
   Each modality gets its own weight initialisation scale, which matters
   because their ranges differ significantly even after z-score normalisation.

3. Per-modality output heads
   Separate Linear layers for obs / action / reward velocities.
   Enables per-modality loss weighting in flow_matching.py without any
   index-slicing hacks — the model itself knows which output is which.

4. Zero-init final layers
   All output weights and adaLN modulation weights are zero-initialised.
   At step 0 every block is an identity map and the model outputs zero
   velocity — a neutral starting point that trains faster than random init.
"""

from __future__ import annotations
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# RoPE utilities
# ──────────────────────────────────────────────────────────────────────────────

def build_rope_cache(
    seq_len:   int,
    head_dim:  int,
    theta:     float = 10_000.0,
    device:    torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pre-compute RoPE cos/sin tables.

    Returns cos, sin each of shape (seq_len, head_dim // 2).
    We only need half the head_dim because RoPE pairs dimensions:
    (d_0, d_1), (d_2, d_3), …  — each pair gets one rotation angle.
    """
    half = head_dim // 2
    # Frequencies: θ_i = 1 / (theta^(2i/head_dim))
    freqs = 1.0 / (theta ** (torch.arange(0, half, device=device).float() / half))
    positions = torch.arange(seq_len, device=device).float()
    # Outer product: (seq_len, half)
    angles = torch.outer(positions, freqs)
    return angles.cos(), angles.sin()


def apply_rope(
    x:   torch.Tensor,    # (B, n_heads, T, head_dim)
    cos: torch.Tensor,    # (T, head_dim // 2)
    sin: torch.Tensor,    # (T, head_dim // 2)
) -> torch.Tensor:
    """
    Apply rotary position embedding to query or key tensor.

    The rotation pairs even and odd dimensions:
        x_rot[..., 2i]   = x[..., 2i] * cos[t, i] - x[..., 2i+1] * sin[t, i]
        x_rot[..., 2i+1] = x[..., 2i] * sin[t, i] + x[..., 2i+1] * cos[t, i]
    """
    # Reshape cos/sin for broadcasting: (1, 1, T, head_dim//2)
    cos = cos[None, None, :, :]   # (1, 1, T, D//2)
    sin = sin[None, None, :, :]

    x_even = x[..., 0::2]         # (B, H, T, D//2)
    x_odd  = x[..., 1::2]

    # Complex-number rotation
    rot_even = x_even * cos - x_odd * sin
    rot_odd  = x_even * sin + x_odd * cos

    # Interleave back: stack on last dim then flatten
    return torch.stack([rot_even, rot_odd], dim=-1).flatten(-2)


# ──────────────────────────────────────────────────────────────────────────────
# Attention with RoPE + Flash Attention
# ──────────────────────────────────────────────────────────────────────────────

class RoPEAttention(nn.Module):
    """
    Multi-head self-attention with Rotary Position Embeddings.

    Uses torch.nn.functional.scaled_dot_product_attention which
    automatically dispatches to Flash Attention 2 when available,
    giving ~2-3× speedup over standard attention for T=100-500.

    RoPE is applied only to Q and K (not V) which is the standard practice.
    """

    def __init__(self, hidden_size: int, num_heads: int, max_seq_len: int = 1024, rope_theta: float = 10_000.0):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        self.num_heads  = num_heads
        self.head_dim   = hidden_size // num_heads
        self.scale      = self.head_dim ** -0.5

        self.qkv  = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.proj = nn.Linear(hidden_size, hidden_size)

        # RoPE cache — registered as buffer so it moves with .to(device)
        cos, sin = build_rope_cache(max_seq_len, self.head_dim, theta=rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)   # (max_T, D//2)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, T, H)
        Returns: (B, T, H)
        """
        B, T, H = x.shape

        # Project and split into heads
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)               # each (B, T, num_heads, head_dim)

        # Move heads before sequence: (B, num_heads, T, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply RoPE to Q and K only (standard practice)
        cos = self.rope_cos[:T]    # (T, head_dim//2)
        sin = self.rope_sin[:T]
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Flash Attention (falls back to standard if not available)
        out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)  # (B, H, T, head_dim)

        # Merge heads back
        out = out.transpose(1, 2).reshape(B, T, H)
        return self.proj(out)


# ──────────────────────────────────────────────────────────────────────────────
# Modulation helpers
# ──────────────────────────────────────────────────────────────────────────────

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """adaLN modulation.  shift/scale: (B, H) → broadcast over T."""
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# ──────────────────────────────────────────────────────────────────────────────
# Embedders
# ──────────────────────────────────────────────────────────────────────────────

class TimestepEmbedder(nn.Module):
    """Scalar flow time τ ∈ [0,1] → ℝ^H via sinusoidal basis + MLP."""

    def __init__(self, hidden_size: int, freq_dim: int = 256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        nn.init.normal_(self.mlp[0].weight, std=0.02)
        nn.init.normal_(self.mlp[2].weight, std=0.02)

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        """tau: (B,) → (B, H)"""
        half  = self.freq_dim // 2
        freqs = torch.exp(
            -math.log(10_000) * torch.arange(half, device=tau.device) / half
        )
        args  = tau[:, None].float() * freqs[None]
        emb   = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(emb)


class ConditionEmbedder(nn.Module):
    """
    [s_0 (obs_dim) | R (1)] → ℝ^H  with learnable CFG null embedding.

    During training the condition is replaced by `null_emb` with probability
    cfg_dropout_prob.  At inference, running with drop_mask=all-True gives
    the unconditional velocity needed for CFG interpolation.
    """

    def __init__(self, cond_dim: int, hidden_size: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(cond_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.null_emb = nn.Parameter(torch.zeros(hidden_size))
        nn.init.normal_(self.null_emb, std=0.02)

    def forward(
        self,
        cond:      torch.Tensor,
        drop_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        emb = self.proj(cond)
        if drop_mask is not None and drop_mask.any():
            null = self.null_emb.expand(emb.shape[0], -1)
            emb  = torch.where(drop_mask[:, None], null, emb)
        return emb


class MultiModalEmbedding(nn.Module):
    """
    Separate linear projections for obs, action, reward — summed to hidden_size.

    Why separate projections instead of one big Linear(D, H)?
      - Each modality has different scale/statistics even after normalisation
      - Separate weights allow the optimiser to scale gradients per modality
      - Easier to ablate or freeze individual modality embeddings later
    """

    def __init__(self, obs_dim: int, action_dim: int, reward_dim: int, hidden_size: int):
        super().__init__()
        self.obs_dim    = obs_dim
        self.action_dim = action_dim

        self.obs_proj    = nn.Linear(obs_dim,    hidden_size)
        self.action_proj = nn.Linear(action_dim, hidden_size)
        self.reward_proj = nn.Linear(reward_dim, hidden_size)

        # Slightly different init scales matching typical value ranges
        nn.init.xavier_uniform_(self.obs_proj.weight,    gain=1.0)
        nn.init.xavier_uniform_(self.action_proj.weight, gain=1.0)
        nn.init.xavier_uniform_(self.reward_proj.weight, gain=0.5)   # reward is scalar

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) → (B, T, H)"""
        obs    = x[..., :self.obs_dim]
        action = x[..., self.obs_dim:self.obs_dim + self.action_dim]
        reward = x[..., -1:]
        return self.obs_proj(obs) + self.action_proj(action) + self.reward_proj(reward)


# ──────────────────────────────────────────────────────────────────────────────
# DiT block (RoPE version)
# ──────────────────────────────────────────────────────────────────────────────

class DiTBlock(nn.Module):
    """
    DiT block with adaLN-Zero conditioning and RoPE self-attention.

    adaLN-Zero: all 6 modulation parameters (shift/scale/gate × 2)
    are initialised to zero, making each block start as an identity
    transformation.  This is critical for stable training of deep networks.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads:   int,
        mlp_ratio:   float = 4.0,
        max_seq_len: int   = 1024,
        rope_theta:  float = 10_000.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn  = RoPEAttention(hidden_size, num_heads, max_seq_len, rope_theta)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        mlp_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_dim, hidden_size),
        )

        # 6 modulation parameters: [shift_a, scale_a, gate_a, shift_m, scale_m, gate_m]
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )
        # Zero-init → identity at t=0
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        x : (B, T, H)  trajectory tokens
        c : (B, H)     conditioning vector (τ_emb + cond_emb)
        """
        s_a, sc_a, g_a, s_m, sc_m, g_m = self.adaLN(c).chunk(6, dim=1)

        # Self-attention with adaLN modulation
        x = x + g_a.unsqueeze(1) * self.attn(modulate(self.norm1(x), s_a, sc_a))
        # MLP with adaLN modulation
        x = x + g_m.unsqueeze(1) * self.mlp(modulate(self.norm2(x), s_m, sc_m))
        return x


# ──────────────────────────────────────────────────────────────────────────────
# Per-modality output head
# ──────────────────────────────────────────────────────────────────────────────

class MultiModalHead(nn.Module):
    """
    Final output layer with separate projections for each modality.

    adaLN final normalisation (shared across all three heads) followed by
    three independent linear projections.  The combined output is the full
    velocity vector (B, T, D), but the three components can be weighted
    independently in the loss.

    All weights zero-initialised so initial velocity prediction = 0.
    """

    def __init__(self, hidden_size: int, obs_dim: int, action_dim: int, reward_dim: int):
        super().__init__()
        self.obs_dim    = obs_dim
        self.action_dim = action_dim

        self.norm  = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

        self.obs_head    = nn.Linear(hidden_size, obs_dim)
        self.action_head = nn.Linear(hidden_size, action_dim)
        self.reward_head = nn.Linear(hidden_size, reward_dim)

        # Zero-init all output heads
        for layer in [self.adaLN[-1], self.obs_head, self.action_head, self.reward_head]:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        """
        Returns: (velocity, obs_vel, action_vel, reward_vel)
          velocity   : (B, T, D) — full concatenated velocity
          *_vel      : per-modality slices for weighted loss computation
        """
        shift, scale = self.adaLN(c).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)

        obs_vel    = self.obs_head(x)     # (B, T, obs_dim)
        action_vel = self.action_head(x)  # (B, T, action_dim)
        reward_vel = self.reward_head(x)  # (B, T, reward_dim)

        velocity = torch.cat([obs_vel, action_vel, reward_vel], dim=-1)   # (B, T, D)
        return velocity, obs_vel, action_vel, reward_vel


# ──────────────────────────────────────────────────────────────────────────────
# Main model
# ──────────────────────────────────────────────────────────────────────────────

class TrajectoryDiT(nn.Module):
    """
    Conditional Flow Matching DiT for RL trajectory generation.

    Models the velocity field:
        v_θ(x_τ, τ, s_0, R)  ∈  ℝ^{T × D}

    Used in the ODE:  dx/dτ = v_θ(x_τ, τ, s_0, R)

    Where x = (obs, action, reward) trajectory of shape (T, D).

    Key design choices (see module docstring):
      - RoPE in every attention block
      - Multi-modal input embedding (obs + action + reward projected separately)
      - Per-modality output heads for weighted loss
      - Flash Attention via F.scaled_dot_product_attention
      - Zero-init adaLN and output heads
    """

    def __init__(
        self,
        obs_dim:           int   = 17,
        action_dim:        int   = 6,
        reward_dim:        int   = 1,
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

        # Saved as attributes for checkpoint reconstruction
        self.obs_dim           = obs_dim
        self.action_dim        = action_dim
        self.reward_dim        = reward_dim
        self.D                 = obs_dim + action_dim + reward_dim
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

        # Embeddings
        self.x_emb    = MultiModalEmbedding(obs_dim, action_dim, reward_dim, hidden_size)
        self.tau_emb  = TimestepEmbedder(hidden_size)
        self.cond_emb = ConditionEmbedder(cond_dim, hidden_size)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio, max_seq_len, rope_theta)
            for _ in range(depth)
        ])

        # Per-modality output head
        self.head = MultiModalHead(hidden_size, obs_dim, action_dim, reward_dim)

        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"TrajectoryDiT | {n/1e6:.1f}M params | "
              f"depth={depth} hidden={hidden_size} heads={num_heads} "
              f"T={trajectory_length} RoPE=True FlashAttn=auto")

    def forward(
        self,
        x_tau:     torch.Tensor,                    # (B, T, D)
        tau:       torch.Tensor,                    # (B,)
        cond:      torch.Tensor,                    # (B, cond_dim)
        drop_mask: Optional[torch.Tensor] = None,   # (B,) bool
    ):
        """
        Returns: (velocity, obs_vel, action_vel, reward_vel)
        All shapes (B, T, D_modality).
        """
        # Embed trajectory tokens (multi-modal — no positional enc here, RoPE handles it)
        x = self.x_emb(x_tau)                                  # (B, T, H)

        # Combined conditioning vector
        c = self.tau_emb(tau) + self.cond_emb(cond, drop_mask) # (B, H)

        # Transformer
        for block in self.blocks:
            x = block(x, c)

        # Per-modality velocity output
        return self.head(x, c)

    def config_dict(self) -> dict:
        """Return constructor kwargs for checkpoint reconstruction."""
        return dict(
            obs_dim           = self.obs_dim,
            action_dim        = self.action_dim,
            reward_dim        = self.reward_dim,
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


# ──────────────────────────────────────────────────────────────────────────────
# Smoke test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TrajectoryDiT(
        obs_dim=17, action_dim=6, reward_dim=1,
        trajectory_length=100, hidden_size=512, depth=8, num_heads=8,
    ).to(device)

    B, T, D = 4, 100, 24
    x_tau = torch.randn(B, T, D, device=device)
    tau   = torch.rand(B, device=device)
    cond  = torch.randn(B, 18, device=device)

    vel, v_obs, v_act, v_rew = model(x_tau, tau, cond)
    assert vel.shape   == (B, T, D),  f"velocity shape {vel.shape}"
    assert v_obs.shape == (B, T, 17), f"obs vel shape {v_obs.shape}"
    assert v_act.shape == (B, T, 6),  f"action vel shape {v_act.shape}"
    assert v_rew.shape == (B, T, 1),  f"reward vel shape {v_rew.shape}"

    # CFG drop mask
    drop = torch.tensor([True, False, True, False], device=device)
    vel2, _, _, _ = model(x_tau, tau, cond, drop)
    assert vel2.shape == (B, T, D)

    # Config round-trip
    cfg    = model.config_dict()
    model2 = TrajectoryDiT(**cfg).to(device)
    print(f"\nAll assertions passed.")
    print(f"Config keys: {list(cfg.keys())}")
