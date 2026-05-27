"""
Unified DiSA-RL pipeline entry point.

Runs all three phases sequentially or from a specific phase:

  Phase 0: Offline diffusion training
  Phase 1: Offline IQL training (with augmentation)
  Phase 2: Online SAC fine-tuning

Usage:
    # Full pipeline from scratch
    python run_disa_rl.py --env halfcheetah-medium-v2

    # Start from online phase (offline already done)
    python run_disa_rl.py --env halfcheetah-medium-v2 --start_phase online

    # Skip diffusion, run IQL + online
    python run_disa_rl.py --env halfcheetah-medium-v2 --start_phase iql

    # Just evaluate existing checkpoints
    python run_disa_rl.py --env halfcheetah-medium-v2 --eval_only
"""

from __future__ import annotations
import argparse
import os
import sys
import subprocess
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ──────────────────────────────────────────────────────────────────────────────
# Path helpers
# ──────────────────────────────────────────────────────────────────────────────

def diffusion_ckpt(env: str) -> str:
    return f"./checkpoints/{env}/diffusion/offline_final.pt"

def iql_ckpt(env: str, mode: str = "augmented") -> str:
    return f"./checkpoints/{env}/iql/{mode}/best.pt"

def synthetic_data(env: str) -> str:
    return f"./data/synthetic/{env}/synthetic_transitions.npz"

def online_ckpt(env: str, seed: int = 0) -> str:
    return f"./checkpoints/{env}/online/seed_{seed}/best.pt"


# ──────────────────────────────────────────────────────────────────────────────
# Phase runners
# ──────────────────────────────────────────────────────────────────────────────

def run_diffusion(args) -> bool:
    """Phase 0: Train diffusion model."""
    ckpt = diffusion_ckpt(args.env)
    if os.path.exists(ckpt) and not args.force_retrain:
        print(f"✓ Diffusion model exists: {ckpt}")
        return True

    print(f"\n{'='*60}")
    print(f"  Phase 0: Diffusion training — {args.env}")
    print(f"{'='*60}")

    cmd = [
        "python", "diffusion/train.py",
        "--env",        args.env,
        "--batch_size", str(args.diffusion_batch_size),
        "--lr",         str(args.diffusion_lr),
        "--num_steps",  str(args.diffusion_steps),
    ]
    ret = subprocess.run(cmd).returncode
    if ret != 0:
        print(f"Diffusion training failed (exit code {ret})")
        return False

    print(f"✓ Diffusion training complete: {ckpt}")
    return True


def run_generate(args) -> bool:
    """Phase 0b: Generate synthetic data."""
    syn = synthetic_data(args.env)
    if os.path.exists(syn) and not args.force_retrain:
        print(f"✓ Synthetic data exists: {syn}")
        return True

    print(f"\n{'='*60}")
    print(f"  Phase 0b: Generating synthetic data — {args.env}")
    print(f"{'='*60}")

    cmd = [
        "python", "generate_synthetic_data.py",
        "--env",          args.env,
        "--n_transitions", str(args.synthetic_n),
        "--batch_size",   "64",
    ]
    ret = subprocess.run(cmd).returncode
    if ret != 0:
        print(f"Synthetic data generation failed (exit code {ret})")
        return False

    print(f"✓ Synthetic data generated: {syn}")
    return True


def run_iql(args) -> bool:
    """Phase 1: Offline IQL training with augmentation."""
    best = iql_ckpt(args.env, "augmented")

    if os.path.exists(best) and not args.force_retrain:
        print(f"✓ IQL checkpoint exists: {best}")
        return True

    print(f"\n{'='*60}")
    print(f"  Phase 1: Offline IQL training — {args.env}")
    print(f"  Seeds: {list(range(args.num_seeds))}")
    print(f"{'='*60}")

    procs = []
    for seed in range(args.num_seeds):
        cmd = [
            "python", "iql/train_iql.py",
            "--env",       args.env,
            "--mode",      "augmented",
            "--seed",      str(seed),
            "--num_steps", str(args.iql_steps),
        ]
        env_vars = {**os.environ, "WANDB_MODE": "offline"}
        p = subprocess.Popen(cmd, env=env_vars)
        procs.append(p)

    # Wait for all seeds
    for p in procs:
        p.wait()

    print(f"✓ IQL training complete")
    return True


def run_online(args) -> bool:
    """Phase 2: Online SAC fine-tuning."""
    print(f"\n{'='*60}")
    print(f"  Phase 2: Online RL training — {args.env}")
    print(f"{'='*60}")

    # Check prerequisites
    iql_path = iql_ckpt(args.env, "augmented")
    diff_path = diffusion_ckpt(args.env)
    syn_path  = synthetic_data(args.env)

    if not os.path.exists(iql_path):
        print(f"ERROR: IQL checkpoint not found: {iql_path}")
        return False
    if not os.path.exists(syn_path):
        print(f"ERROR: Synthetic data not found: {syn_path}")
        return False

    cmd = [
        "python", "online_rl/train_online.py",
        "--env",               args.env,
        "--iql_ckpt",          iql_path,
        "--synthetic_data",    syn_path,
        "--num_steps",         str(args.online_steps),
        "--seed",              str(args.seed),
        "--critic_warmup_steps", str(args.critic_warmup),
        "--transition_steps",  str(args.transition_steps),
        "--min_real_steps",    str(args.min_real_steps),
        "--finetune_every",    str(args.finetune_every),
    ]
    if os.path.exists(diff_path):
        cmd += ["--diffusion_ckpt", diff_path]

    ret = subprocess.run(cmd).returncode
    return ret == 0


def run_eval(args) -> None:
    """Evaluate all available checkpoints and print results table."""
    import json

    print(f"\n{'='*60}")
    print(f"  Evaluation: {args.env}")
    print(f"{'='*60}")

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from iql.evaluator import make_evaluator
    evaluator = make_evaluator(args.env, device, n_episodes=10)

    results = {}

    # Evaluate IQL augmented (best seed)
    for seed in range(args.num_seeds):
        ckpt = f"./checkpoints/{args.env}/iql/augmented/best.pt"
        if os.path.exists(ckpt):
            from iql.agent import IQLAgent
            from iql.train_iql import ENV_REGISTRY
            obs_dim, action_dim, _ = ENV_REGISTRY[args.env]
            agent = IQLAgent(obs_dim, action_dim, device=device)
            agent.load(ckpt)
            metrics = evaluator.evaluate(agent.actor)
            if metrics:
                results[f"IQL_augmented"] = metrics["eval/normalized"]
                print(f"  IQL augmented:  {metrics['eval/normalized']:.1f}")
            break

    # Evaluate online SAC
    for seed in range(args.num_seeds):
        ckpt = online_ckpt(args.env, seed)
        if os.path.exists(ckpt):
            from online_rl.sac import SACAgent
            obs_dim, action_dim = ENV_REGISTRY_ONLINE[args.env]
            agent = SACAgent(obs_dim, action_dim, device=device)
            agent.load(ckpt)
            metrics = evaluator.evaluate(agent.actor)
            if metrics:
                results[f"DiSA_RL_online_s{seed}"] = metrics["eval/normalized"]
                print(f"  DiSA-RL online seed {seed}: {metrics['eval/normalized']:.1f}")

    evaluator.close()
    print(f"\nResults: {json.dumps(results, indent=2)}")


ENV_REGISTRY_ONLINE = {
    "halfcheetah-medium-v2":        (17,  6),
    "hopper-medium-v2":             (11,  3),
    "walker2d-medium-v2":           (17,  6),
    "ant-medium-v2":                (111, 8),
    "halfcheetah-medium-replay-v2": (17,  6),
    "hopper-medium-replay-v2":      (11,  3),
    "walker2d-medium-replay-v2":    (17,  6),
    "ant-medium-replay-v2":         (111, 8),
}


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DiSA-RL unified pipeline")
    parser.add_argument("--env",    type=str, required=True)
    parser.add_argument("--seed",   type=int, default=0)
    parser.add_argument("--start_phase", type=str, default="diffusion",
                        choices=["diffusion", "iql", "online", "eval"])
    parser.add_argument("--eval_only",   action="store_true")
    parser.add_argument("--force_retrain", action="store_true")
    parser.add_argument("--num_seeds",   type=int, default=5)

    # Diffusion
    parser.add_argument("--diffusion_steps",      type=int,   default=300_000)
    parser.add_argument("--diffusion_batch_size", type=int,   default=256)
    parser.add_argument("--diffusion_lr",         type=float, default=1e-4)
    parser.add_argument("--synthetic_n",          type=int,   default=1_000_000)

    # IQL
    parser.add_argument("--iql_steps", type=int, default=1_000_000)

    # Online
    parser.add_argument("--online_steps",      type=int, default=500_000)
    parser.add_argument("--critic_warmup",     type=int, default=5_000)
    parser.add_argument("--transition_steps",  type=int, default=20_000)
    parser.add_argument("--min_real_steps",    type=int, default=10_000)
    parser.add_argument("--finetune_every",    type=int, default=10_000)

    args = parser.parse_args()

    if args.eval_only:
        run_eval(args)
        exit(0)

    PHASES = ["diffusion", "iql", "online"]
    start_idx = PHASES.index(args.start_phase)

    success = True
    for phase in PHASES[start_idx:]:
        if phase == "diffusion":
            success = run_diffusion(args) and run_generate(args)
        elif phase == "iql":
            success = run_iql(args)
        elif phase == "online":
            success = run_online(args)
        if not success:
            print(f"\nPipeline stopped at phase: {phase}")
            exit(1)

    print(f"\n{'='*60}")
    print(f"  DiSA-RL pipeline complete: {args.env}")
    print(f"{'='*60}")
