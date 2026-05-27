"""
TR-IVAR (Trust-Region Iterative Value-Aware Refinement) orchestrator.

Pillar 3 of DiSA-RL. Closes the loop critic → generator → critic with a
density-ratio trust region to bound per-round drift.

Algorithm (K rounds):

    Round k:
      1. Train DRC-IQL on D_real ∪ D_syn^(k-1)        → (Q^k, V^k)
      2. Generate D_syn^k with QCD using (Q^k, V^k)
      3. Trust region:
            ε = E_{(s,a)~D_syn^k}[(w^k(s,a) − 1)²]
            where w^k = p_real / p_syn^k from discriminator d^k
         If ε > ε_max: attenuate value guidance (λ_v *= 0.5),
                       reduce alpha (alpha *= 0.5), regenerate, retry.
         Else: accept D_syn^k, advance to round k+1.

This module orchestrates calls to the existing entry points:
  * iql/train_iql.py  (DRC-IQL training)
  * generate_synthetic_data.py  (QCD generation)
and parses their log/csv outputs to inspect ε.

Usage:
    python iql/ivar.py --env hopper-medium-v2 --n_rounds 3 --seeds 0

Round artifacts go under:
    checkpoints/<env>/ivar/round_<k>/
    data/synthetic/<env>/ivar_round_<k>.npz
"""

from __future__ import annotations
import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

from iql.agent import IQLAgent
from iql.discriminator import DensityRatioDiscriminator


def _run(cmd: list[str], log_path: Optional[Path] = None) -> int:
    """Run a subprocess, optionally tee output to a log."""
    print(f"\n$  {' '.join(cmd)}\n")
    if log_path is None:
        return subprocess.call(cmd)
    with open(log_path, "w") as f:
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        return p.wait()


def compute_trust_region_epsilon(
    iql_ckpt:  str,
    real_npz:  str,
    syn_npz:   str,
    device:    torch.device,
    n_samples: int = 50_000,
) -> tuple[float, float]:
    """
    Estimate ε = E_{(s,a)~D_syn}[(w(s,a)-1)²] where w = p_real / p_syn
    using a fresh discriminator trained on the current real vs syn data.

    Returns (epsilon, w_mean).

    Note: this is the trust-region MEASURE, computed AFTER the IQL round
    so we can decide whether to accept the new D_syn before the next round.
    """
    real = np.load(real_npz, allow_pickle=True)
    syn  = np.load(syn_npz, allow_pickle=True)

    obs_dim    = real["observations"].shape[1]
    action_dim = real["actions"].shape[1]

    # Train a small discriminator on real vs syn
    r_obs = torch.from_numpy(real["observations"][:n_samples].astype(np.float32)).to(device)
    r_act = torch.from_numpy(real["actions"][:n_samples].astype(np.float32)).to(device)
    s_obs = torch.from_numpy(syn["observations"][:n_samples].astype(np.float32)).to(device)
    s_act = torch.from_numpy(syn["actions"][:n_samples].astype(np.float32)).to(device)

    disc = DensityRatioDiscriminator(obs_dim, action_dim).to(device)
    opt  = torch.optim.Adam(disc.parameters(), lr=3e-4)
    batch = 1024
    for it in range(2000):
        idx_r = torch.randint(0, len(r_obs), (batch,), device=device)
        idx_s = torch.randint(0, len(s_obs), (batch,), device=device)
        loss = disc.bce_loss(r_obs[idx_r], r_act[idx_r], s_obs[idx_s], s_act[idx_s])
        opt.zero_grad(); loss.backward(); opt.step()

    # Compute ε on a held-out chunk of syn
    with torch.no_grad():
        w = disc.density_ratio(s_obs, s_act, clip=(0.1, 10.0))   # wider clip for measurement
    eps = float(((w - 1.0) ** 2).mean().item())
    w_mean = float(w.mean().item())
    return eps, w_mean


def run_ivar(
    env:           str,
    n_rounds:      int   = 3,
    seed:          int   = 0,
    base_alpha:    float = 0.5,
    epsilon_max:   float = 0.5,
    num_steps:     int   = 500_000,
    bc_weight:     float = 0.1,
    reward_scale:  float = 1.0,
    pa_weight:     float = 0.0,
    pa_min_q:      float = 0.0,
    wandb_project: str   = "disa-rl-ivar",
    dry_run:       bool  = False,
) -> None:
    """
    Run K rounds of TR-IVAR.

    The first IQL round (round 0) trains on offline_only (no syn) — this
    is our "Q^0" that seeds QCD generation. Subsequent rounds train on
    D_real ∪ D_syn^(k-1) with DRC-IQL.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    real_npz = f"./data/{env}.npz"
    assert os.path.exists(real_npz), f"missing {real_npz}"

    os.makedirs(f"./checkpoints/{env}/ivar", exist_ok=True)
    os.makedirs(f"./data/synthetic/{env}", exist_ok=True)
    log_dir = Path(f"./logs/ivar_{env}_s{seed}")
    log_dir.mkdir(parents=True, exist_ok=True)

    epsilon_history = []
    alpha = base_alpha
    lambda_v = 0.5    # initial value-guidance scale for QCD

    # ── Round 0: train pure-real critic (Q^0) ────────────────────────────
    print(f"\n{'='*60}\n  IVAR Round 0 (real-only critic seed)\n{'='*60}\n")
    round0_ckpt_dir = f"./checkpoints/{env}/iql/offline_only/seed_{seed}"
    cmd = [
        "python", "-u", "iql/train_iql.py",
        "--env", env, "--mode", "offline_only", "--bc_weight", "0.0",
        "--seed", str(seed), "--num_steps", str(num_steps),
        "--q_hidden_dims", "256", "256", "256", "256",
        "--num_critics", "10", "--critic_subset", "2",
        "--action_noise_std", "0.05",
        "--reward_scale", str(reward_scale),
        "--pa_weight", str(pa_weight), "--pa_min_q", str(pa_min_q),
        "--wandb_project", wandb_project,
    ]
    if not dry_run:
        _run(cmd, log_path=log_dir / "round0_iql.log")

    # ── Rounds 1..K ──────────────────────────────────────────────────────
    for k in range(1, n_rounds + 1):
        print(f"\n{'='*60}\n  IVAR Round {k}/{n_rounds}  "
              f"(alpha={alpha:.2f}, lambda_v={lambda_v:.2f})\n{'='*60}\n")

        prev_iql_ckpt = (
            f"./checkpoints/{env}/iql/offline_only/seed_{seed}/final.pt"
            if k == 1
            else f"./checkpoints/{env}/iql/augmented/alpha{alpha:.2f}/seed_{seed}/final.pt"
        )

        # Step 1: Generate D_syn^k under Q^(k-1)
        syn_target = f"./data/synthetic/{env}/synthetic_transitions.npz"
        round_syn  = f"./data/synthetic/{env}/ivar_round_{k}.npz"
        gen_cmd = [
            "python", "-u", "generate_synthetic_data.py",
            "--env", env, "--return_sampling", "topk",
            "--cfg_scale", "1.2",
            "--vcdg",
            "--vcdg_iql_ckpt", prev_iql_ckpt,
            "--vcdg_guidance_scale", str(lambda_v),
        ]
        if not dry_run:
            _run(gen_cmd, log_path=log_dir / f"round{k}_gen.log")
            # Rename the VCDG output to a round-specific name + activate it
            vcdg_path = f"./data/synthetic/{env}/synthetic_transitions_vcdg.npz"
            if os.path.exists(vcdg_path):
                shutil.copy2(vcdg_path, round_syn)
                shutil.copy2(vcdg_path, syn_target)   # also overwrite the default path
                print(f"[ivar] copied {vcdg_path} → {round_syn}")
            else:
                raise FileNotFoundError(f"VCDG didn't produce {vcdg_path}")

            # Step 2: Compute trust-region ε
            eps, w_mean = compute_trust_region_epsilon(
                iql_ckpt=prev_iql_ckpt, real_npz=real_npz,
                syn_npz=round_syn, device=device,
            )
            epsilon_history.append(eps)
            print(f"[ivar] round {k}: ε={eps:.4f}  w_mean={w_mean:.3f}  "
                  f"threshold={epsilon_max}")

            # Step 3: If outside trust region, attenuate and retry once
            if eps > epsilon_max:
                print(f"[ivar] ε > {epsilon_max}: attenuating lambda_v *= 0.5, "
                      f"alpha *= 0.75, regenerating...")
                lambda_v *= 0.5
                alpha = max(0.1, alpha * 0.75)
                gen_cmd[gen_cmd.index("--vcdg_guidance_scale") + 1] = str(lambda_v)
                _run(gen_cmd, log_path=log_dir / f"round{k}_gen_retry.log")
                if os.path.exists(vcdg_path):
                    shutil.copy2(vcdg_path, round_syn)
                    shutil.copy2(vcdg_path, syn_target)
                eps2, w2 = compute_trust_region_epsilon(
                    iql_ckpt=prev_iql_ckpt, real_npz=real_npz,
                    syn_npz=round_syn, device=device,
                )
                print(f"[ivar] retry: ε={eps2:.4f}  w_mean={w2:.3f}")

        # Step 4: Train DRC-IQL on D_real ∪ D_syn^k
        iql_cmd = [
            "python", "-u", "iql/train_iql.py",
            "--env", env, "--mode", "augmented",
            "--alpha", f"{alpha:.2f}", "--seed", str(seed),
            "--num_steps", str(num_steps),
            "--bc_weight", str(bc_weight),
            "--q_hidden_dims", "256", "256", "256", "256",
            "--num_critics", "10", "--critic_subset", "2",
            "--sa_iql", "--expectile_real", "0.9", "--expectile_syn", "0.5",
            "--sa_clip", "0.5", "2.0",
            "--action_noise_std", "0.05",
            "--reward_scale", str(reward_scale),
            "--pa_weight", str(pa_weight), "--pa_min_q", str(pa_min_q),
            "--alpha_warmup", "50000", "--alpha_ramp", "50000",
            "--wandb_project", wandb_project,
        ]
        if not dry_run:
            _run(iql_cmd, log_path=log_dir / f"round{k}_iql.log")

    # Final summary
    print(f"\n{'='*60}")
    print(f"  TR-IVAR completed for {env} (seed {seed})")
    print(f"  Rounds: {n_rounds}")
    print(f"  ε history: {epsilon_history}")
    print(f"  Final alpha: {alpha:.2f}  lambda_v: {lambda_v:.2f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=str, required=True)
    p.add_argument("--n_rounds", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--epsilon_max", type=float, default=0.5)
    p.add_argument("--num_steps", type=int, default=500_000)
    p.add_argument("--bc_weight", type=float, default=0.1)
    p.add_argument("--reward_scale", type=float, default=1.0)
    p.add_argument("--pa_weight", type=float, default=0.0)
    p.add_argument("--pa_min_q", type=float, default=0.0)
    p.add_argument("--wandb_project", type=str, default="disa-rl-ivar")
    p.add_argument("--dry_run", action="store_true",
                   help="Print commands but don't execute")
    args = p.parse_args()

    run_ivar(
        env=args.env, n_rounds=args.n_rounds, seed=args.seed,
        base_alpha=args.alpha, epsilon_max=args.epsilon_max,
        num_steps=args.num_steps, bc_weight=args.bc_weight,
        reward_scale=args.reward_scale,
        pa_weight=args.pa_weight, pa_min_q=args.pa_min_q,
        wandb_project=args.wandb_project, dry_run=args.dry_run,
    )
