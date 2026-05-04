"""
IQL training script for DiSA-RL.

Two modes:
  offline_only  — train on raw D4RL data only (baseline)
  augmented     — train on D4RL + pre-generated synthetic data

Workflow for augmented:
  Step 1: python generate_synthetic_data.py --env <env>
  Step 2: python iql/train_iql.py --env <env> --mode augmented

Usage:
    # Baseline
    python iql/train_iql.py --env halfcheetah-medium-v2 --mode offline_only --seed 0

    # DiSA-RL augmented (50% real, 50% synthetic)
    python iql/train_iql.py --env halfcheetah-medium-v2 --mode augmented --alpha 0.5 --seed 0

    # DiSA-RL augmented (25% real, 75% synthetic)
    python iql/train_iql.py --env halfcheetah-medium-v2 --mode augmented --alpha 0.25 --seed 0

    # DiSA-RL augmented (100% synthetic)
    python iql/train_iql.py --env halfcheetah-medium-v2 --mode augmented --alpha 0.0 --seed 0
"""

from __future__ import annotations
import argparse
import os
import sys
import numpy as np
import torch
import wandb
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iql.agent import IQLAgent
from iql.buffer import ReplayBuffer, SyntheticBuffer, AugmentedReplayBuffer
from iql.evaluator import make_evaluator


# ──────────────────────────────────────────────────────────────────────────────
# Dynamic environment detection — no hardcoded dims
# ──────────────────────────────────────────────────────────────────────────────

def get_env_info(env_name: str, data_dir: str = "./data"):
    """
    Read obs_dim and action_dim directly from the .npz file.
    No hardcoded registry needed — works for any D4RL environment.
    """
    data_path = os.path.join(data_dir, f"{env_name}.npz")
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found: {data_path}\n"
            f"Run: python download_data.py --datasets {env_name}"
        )
    data    = np.load(data_path, allow_pickle=True)
    obs_dim = data["observations"].shape[1]
    act_dim = data["actions"].shape[1]
    return obs_dim, act_dim, data_path

def get_synthetic_path(env_name: str) -> str:
    return f"./data/synthetic/{env_name}/synthetic_transitions.npz"


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train_iql(args) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    obs_dim, action_dim, data_path = get_env_info(args.env)

    # ── Agent ─────────────────────────────────────────────────────────────
    # bc_weight: BC anchor on real data only, keeps policy from chasing
    # unreachable synthetic states. Disabled for offline_only mode.
    agent = IQLAgent(
        obs_dim     = obs_dim,
        action_dim  = action_dim,
        hidden_dims = (256, 256),
        expectile   = args.expectile,
        temperature = args.temperature,
        discount    = 0.99,
        tau         = 0.005,
        lr_q        = 3e-4,
        lr_v        = 3e-4,
        lr_pi       = 3e-4,
        bc_weight   = args.bc_weight,   # ← NEW: BC anchor weight
        device      = device,
    )

    # ── Buffers ───────────────────────────────────────────────────────────
    real_buffer = ReplayBuffer(data_path, device)

    synthetic_buffer = None
    alpha = 1.0  # default: pure real

    if args.mode == "augmented":
        syn_path = args.synthetic_data or get_synthetic_path(args.env)
        if os.path.exists(syn_path):
            synthetic_buffer = SyntheticBuffer(
                syn_path, device,
                real_reward_mean  = real_buffer.reward_mean,
                real_reward_std   = real_buffer.reward_std,
                normalize_rewards = True,
                filter_sigma      = 3.0,
                return_weighting  = True,
            )
            alpha = args.alpha
        else:
            print(f"\nWARNING: Synthetic data not found at {syn_path}")
            print(f"Generate it first:")
            print(f"  python generate_synthetic_data.py --env {args.env}\n")
            print("Falling back to offline_only mode.")
            args.mode = "offline_only"

    aug_buffer = AugmentedReplayBuffer(
        real_buffer      = real_buffer,
        synthetic_buffer = synthetic_buffer,
        alpha            = alpha,
    )

    # ── Evaluator ─────────────────────────────────────────────────────────
    evaluator = make_evaluator(args.env, device, n_episodes=10)

    # ── WandB ─────────────────────────────────────────────────────────────
    env_tag  = args.env.replace("-v2", "").replace("-", "_")
    run_name = f"iql_{env_tag}_{args.mode}_alpha{args.alpha}_s{args.seed}"
    output_dir = os.path.join("./checkpoints", args.env, "iql", args.mode,
                               f"alpha{args.alpha}", f"seed_{args.seed}")
    os.makedirs(output_dir, exist_ok=True)

    wandb.init(
        project = args.wandb_project,
        name    = run_name,
        mode    = os.environ.get("WANDB_MODE", "online"),
        config  = dict(
            env        = args.env,
            mode       = args.mode,
            seed       = args.seed,
            num_steps  = args.num_steps,
            batch_size = args.batch_size,
            alpha       = alpha,
            bc_weight   = args.bc_weight,
            expectile   = args.expectile,
            temperature = args.temperature,
        ),
    )

    print(f"\n{'='*55}")
    print(f"  IQL: {args.env}")
    print(f"  Mode      : {args.mode}")
    print(f"  Alpha     : {alpha:.2f}  ({'pure real' if alpha >= 1.0 else f'{(1-alpha)*100:.0f}% synthetic'})")
    print(f"  BC weight : {args.bc_weight}")
    print(f"  Steps     : {args.num_steps:,}")
    print(f"  Output    : {output_dir}/")
    print(f"{'='*55}\n")

    # ── Training loop ──────────────────────────────────────────────────────
    best_score = -float("inf")
    pbar = tqdm(total=args.num_steps, desc=f"IQL [{args.mode}] seed={args.seed}")

    for step in range(1, args.num_steps + 1):
        batch = aug_buffer.sample(args.batch_size)

        # BC anchor: pass real_batch so actor stays close to real distribution.
        # Only used in augmented mode with bc_weight > 0.
        # In offline_only mode real_batch=None → standard AWR only.
        real_batch = aug_buffer.sample_real(args.batch_size) \
                     if args.mode == "augmented" and args.bc_weight > 0 \
                     else None

        metrics = agent.update(batch, real_batch=real_batch)  # ← NEW

        if step % args.log_every == 0:
            metrics["step"] = step
            wandb.log(metrics, step=step)
            pbar.set_postfix(
                Q=f"{metrics['loss/q']:.3f}",
                V=f"{metrics['loss/v']:.3f}",
                pi=f"{metrics['loss/actor']:.3f}",
            )

        if step % args.eval_every == 0:
            eval_metrics = evaluator.evaluate(agent.actor)
            if eval_metrics:
                wandb.log(eval_metrics, step=step)
                score = eval_metrics["eval/normalized"]
                print(f"\n[{step:>7d}]  normalized={score:.1f}  "
                      f"return={eval_metrics['eval/return_mean']:.1f}")
                if score > best_score:
                    best_score = score
                    agent.save(os.path.join(output_dir, "best.pt"))

        if step % args.save_every == 0:
            agent.save(os.path.join(output_dir, f"step_{step:07d}.pt"))

        pbar.update(1)

    pbar.close()
    agent.save(os.path.join(output_dir, "final.pt"))
    print(f"\nDone. Best normalized score: {best_score:.1f}")
    wandb.finish()
    evaluator.close()


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IQL offline RL")
    parser.add_argument("--env",  type=str, required=True,
                        help="D4RL dataset name e.g. halfcheetah-medium-v2")
    parser.add_argument("--mode", type=str, default="augmented",
                        choices=["offline_only", "augmented"])
    parser.add_argument("--synthetic_data", type=str, default=None,
                        help="Path to synthetic .npz (auto-detected if None)")
    parser.add_argument("--alpha",      type=float, default=0.5,
                        help="Fraction of real data (0.5=50%% real, 0.0=100%% synthetic)")
    parser.add_argument("--expectile",   type=float, default=0.7,
                        help="IQL expectile for value function (0.7=standard, 0.9=optimistic)")
    parser.add_argument("--temperature", type=float, default=3.0,
                        help="AWR temperature for actor update (3.0=standard, 10.0=stronger)")
    parser.add_argument("--bc_weight",  type=float, default=0.1,
                        help="BC anchor weight on real data (0.0 = disabled)")
    parser.add_argument("--num_steps",  type=int,   default=1_000_000)
    parser.add_argument("--batch_size", type=int,   default=256)
    parser.add_argument("--eval_every", type=int,   default=10_000)
    parser.add_argument("--log_every",  type=int,   default=1_000)
    parser.add_argument("--save_every", type=int,   default=100_000)
    parser.add_argument("--seed",       type=int,   default=0)
    parser.add_argument("--wandb_project", type=str, default="disa-rl",
                        help="WandB project name")
    args = parser.parse_args()
    train_iql(args)