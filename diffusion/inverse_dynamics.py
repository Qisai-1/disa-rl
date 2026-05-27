"""
Inverse Dynamics Model (IDM) for DiSA-RL pillar 1.

Given two consecutive (real or generated) states (s_t, s_{t+1}), predict the
action a_t that produced the transition. Trained on the offline dataset, then
used at *generation time* to fill in actions for state-only diffusion outputs.

Architecture: 4-layer MLP with LayerNorm + ReLU + tanh-squashed action head.
Output is a Gaussian over actions (mean, log_std) so we can either sample or
use the mean.

Loss: negative log-likelihood + small action-magnitude regularizer.

Why this exists:
  Joint (state, action) diffusion has compounding distribution shift — the
  action subspace and state subspace pull against each other under CFG. Even
  in-support state generation can produce actions that are inconsistent with
  the implied dynamics. Decoupling the two (states from diffusion, actions
  from a deterministic-ish IDM) makes the synthetic action *by construction*
  consistent with the synthetic state transition.

Reference: Pathak et al. 2017 (curiosity-driven exploration uses IDM as a
self-supervised signal); also used by LMPC and PETS in model-based RL.
"""

from __future__ import annotations
import math
import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, TensorDataset


# ──────────────────────────────────────────────────────────────────────────────
# Network
# ──────────────────────────────────────────────────────────────────────────────

class InverseDynamicsNet(nn.Module):
    """f_φ(s_t, s_{t+1}) → Gaussian(action). tanh-squashed for [-1, 1] outputs."""

    def __init__(
        self,
        obs_dim:     int,
        action_dim:  int,
        hidden_dims: Tuple[int, ...] = (256, 256, 256, 256),
    ):
        super().__init__()
        self.action_dim = action_dim
        dims = (2 * obs_dim,) + hidden_dims
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers.extend([nn.Linear(a, b), nn.LayerNorm(b), nn.ReLU()])
        self.trunk = nn.Sequential(*layers)
        self.mean_head    = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_head = nn.Linear(hidden_dims[-1], action_dim)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)
        # smaller init on heads for stable starting actions
        nn.init.orthogonal_(self.mean_head.weight,    gain=0.01)
        nn.init.orthogonal_(self.log_std_head.weight, gain=0.01)

    def forward(self, s_t: Tensor, s_tp1: Tensor) -> Tuple[Tensor, Tensor]:
        x = torch.cat([s_t, s_tp1], dim=-1)
        h = self.trunk(x)
        mean    = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(-5.0, 2.0)
        return mean, log_std

    @torch.no_grad()
    def predict(
        self,
        s_t:          Tensor,
        s_tp1:        Tensor,
        deterministic: bool = True,
    ) -> Tensor:
        """Return action in [-1, 1]. Default: deterministic (use mean)."""
        mean, log_std = self.forward(s_t, s_tp1)
        if deterministic:
            return torch.tanh(mean)
        eps = torch.randn_like(mean)
        return torch.tanh(mean + log_std.exp() * eps)

    def log_prob(self, s_t: Tensor, s_tp1: Tensor, action: Tensor) -> Tensor:
        """Log-probability of `action` (in [-1, 1]) under the IDM."""
        mean, log_std = self.forward(s_t, s_tp1)
        # inverse tanh
        a_clamped = action.clamp(-1 + 1e-6, 1 - 1e-6)
        x_t = torch.atanh(a_clamped)
        std = log_std.exp()
        lp = (
            -0.5 * ((x_t - mean) / std).pow(2)
            - log_std
            - 0.5 * math.log(2 * math.pi)
        ).sum(dim=-1)
        # tanh correction
        lp -= (2.0 * (math.log(2) - x_t - F.softplus(-2 * x_t))).sum(dim=-1)
        return lp


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def _load_pairs(data_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (s_t, s_{t+1}, a_t) arrays. Skips done-step pairs."""
    d = np.load(data_path, allow_pickle=True)
    obs  = d["observations"].astype(np.float32)
    act  = d["actions"].astype(np.float32)
    term = d["terminals"].astype(bool)
    if "next_observations" in d.files:
        s_tp1 = d["next_observations"].astype(np.float32)
    else:
        # construct from shifted obs
        s_tp1 = np.concatenate([obs[1:], obs[-1:]], axis=0)
    mask = ~term  # drop transitions where the trajectory ended
    return obs[mask], s_tp1[mask], act[mask]


def train_idm(
    env:          str,
    data_dir:     str   = "./data",
    output_dir:   str   = "./checkpoints",
    hidden_dims:  Tuple[int, ...] = (256, 256, 256, 256),
    lr:           float = 3e-4,
    batch_size:   int   = 1024,
    num_epochs:   int   = 50,
    val_fraction: float = 0.05,
    weight_decay: float = 1e-4,
    device:       Optional[torch.device] = None,
    seed:         int   = 0,
) -> str:
    """
    Train an IDM on the real D4RL dataset. Save to
    ./checkpoints/<env>/idm/idm.pt.
    """
    torch.manual_seed(seed); np.random.seed(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = os.path.join(data_dir, f"{env}.npz")
    s, s_next, a = _load_pairs(data_path)
    n = len(s)
    obs_dim = s.shape[1]; act_dim = a.shape[1]

    # Normalize states (z-score) — IDM is sensitive to scale
    s_mean = s.mean(0); s_std = s.std(0) + 1e-6
    s_norm     = (s      - s_mean) / s_std
    s_next_norm = (s_next - s_mean) / s_std

    # Train / val split
    rng = np.random.default_rng(seed)
    perm  = rng.permutation(n)
    n_val = int(n * val_fraction)
    val_i, train_i = perm[:n_val], perm[n_val:]

    train_ds = TensorDataset(
        torch.from_numpy(s_norm[train_i]),
        torch.from_numpy(s_next_norm[train_i]),
        torch.from_numpy(a[train_i]),
    )
    val_ds = TensorDataset(
        torch.from_numpy(s_norm[val_i]),
        torch.from_numpy(s_next_norm[val_i]),
        torch.from_numpy(a[val_i]),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=4096, shuffle=False)

    net = InverseDynamicsNet(obs_dim, act_dim, hidden_dims).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = float("inf")
    ckpt_dir = os.path.join(output_dir, env, "idm")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "idm.pt")

    print(f"IDM training: env={env}  obs_dim={obs_dim}  act_dim={act_dim}  "
          f"n_train={len(train_i):,}  n_val={len(val_i):,}")

    for epoch in range(1, num_epochs + 1):
        net.train()
        tot, count = 0.0, 0
        for s_b, sp_b, a_b in train_loader:
            s_b, sp_b, a_b = s_b.to(device), sp_b.to(device), a_b.to(device)
            mean, log_std = net(s_b, sp_b)
            std = log_std.exp()
            # NLL of action under tanh-Gaussian
            a_clamped = a_b.clamp(-1 + 1e-6, 1 - 1e-6)
            x_t = torch.atanh(a_clamped)
            nll = (
                0.5 * ((x_t - mean) / std).pow(2)
                + log_std
                + 0.5 * math.log(2 * math.pi)
            ).sum(dim=-1)
            nll += (2.0 * (math.log(2) - x_t - F.softplus(-2 * x_t))).sum(dim=-1)
            loss = nll.mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot += loss.item() * s_b.size(0); count += s_b.size(0)

        # Validation
        net.eval()
        with torch.no_grad():
            val_tot, val_count = 0.0, 0
            for s_b, sp_b, a_b in val_loader:
                s_b, sp_b, a_b = s_b.to(device), sp_b.to(device), a_b.to(device)
                lp = net.log_prob(s_b, sp_b, a_b)
                val_tot += (-lp).sum().item(); val_count += s_b.size(0)
            val_loss = val_tot / val_count
        train_loss = tot / count
        print(f"  epoch {epoch:>3d}/{num_epochs}  train_nll={train_loss:.3f}  val_nll={val_loss:.3f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "state_dict":  net.state_dict(),
                "obs_dim":     obs_dim,
                "action_dim":  act_dim,
                "hidden_dims": list(hidden_dims),
                "s_mean":      s_mean,
                "s_std":       s_std,
                "epoch":       epoch,
                "val_nll":     val_loss,
                "env":         env,
            }, ckpt_path)
            print(f"    saved → {ckpt_path}  (best val_nll={val_loss:.3f})")
    return ckpt_path


def load_idm(ckpt_path: str, device: torch.device) -> Tuple[InverseDynamicsNet, np.ndarray, np.ndarray]:
    """Load IDM + state normalization. Returns (net, s_mean, s_std)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    net = InverseDynamicsNet(
        obs_dim=ckpt["obs_dim"], action_dim=ckpt["action_dim"],
        hidden_dims=tuple(ckpt["hidden_dims"]),
    ).to(device)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net, ckpt["s_mean"], ckpt["s_std"]


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", type=str, required=True)
    ap.add_argument("--num_epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden_dims", type=int, nargs="*",
                    default=[256, 256, 256, 256])
    args = ap.parse_args()
    train_idm(
        env=args.env,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dims=tuple(args.hidden_dims),
    )
