"""
Main IQL training script with diffusion augmentation.

Modes
-----
--mode offline_only
    Train IQL on raw D4RL data.  Baseline — no augmentation.

--mode augmented
    Train IQL with diffusion augmentation.  Requires a trained diffusion
    checkpoint (--diffusion_ckpt).  This is the main DiSA-RL experiment.

--mode ablation_fixed_alpha
    Augmented IQL with a fixed alpha (no adaptive εroll schedule).
    Used for the ablation study comparing fixed vs adaptive mixing.

Usage examples
--------------
    # Baseline (no augmentation)
    python iql/train_iql.py --env halfcheetah-medium-v2 --mode offline_only

    # DiSA-RL augmented
    python iql/train_iql.py --env halfcheetah-medium-v2 --mode augmented \\
        --diffusion_ckpt ./checkpoints/halfcheetah-medium-v2/offline_final.pt

    # Ablation: fixed alpha
    python iql/train_iql.py --env halfcheetah-medium-v2 --mode ablation_fixed_alpha \\
        --diffusion_ckpt ./checkpoints/halfcheetah-medium-v2/offline_final.pt \\
        --alpha 0.5
"""

from __future__ import annotations
import argparse
import os
import sys
import numpy as np
import torch
import wandb
from tqdm import tqdm

# Add parent directory to path so we can import from disa_rl root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iql.networks import GaussianActor, TwinQNetwork, ValueNetwork
from iql.agent import IQLAgent
from iql.buffer import ReplayBuffer, AugmentedReplayBuffer
from iql.evaluator import make_evaluator


# ──────────────────────────────────────────────────────────────────────────────
# Environment registry (same as train.py)
# ──────────────────────────────────────────────────────────────────────────────

ENV_REGISTRY = {
    "halfcheetah-medium-v2":        (17, 6,   "./data/halfcheetah-medium-v2.npz"),
    "halfcheetah-medium-replay-v2": (17, 6,   "./data/halfcheetah-medium-replay-v2.npz"),
    "halfcheetah-expert-v2":        (17, 6,   "./data/halfcheetah-expert-v2.npz"),
    "hopper-medium-v2":             (11, 3,   "./data/hopper-medium-v2.npz"),
    "hopper-medium-replay-v2":      (11, 3,   "./data/hopper-medium-replay-v2.npz"),
    "hopper-expert-v2":             (11, 3,   "./data/hopper-expert-v2.npz"),
    "walker2d-medium-v2":           (17, 6,   "./data/walker2d-medium-v2.npz"),
    "walker2d-medium-replay-v2":    (17, 6,   "./data/walker2d-medium-replay-v2.npz"),
    "walker2d-expert-v2":           (17, 6,   "./data/walker2d-expert-v2.npz"),
    "ant-medium-v2":                (111, 8,  "./data/ant-medium-v2.npz"),
    "ant-medium-replay-v2":         (111, 8,  "./data/ant-medium-replay-v2.npz"),
}

# p90 target returns for CFG conditioning (computed from dataset statistics)
# Higher = more optimistic generation bias
TARGET_RETURNS = {
    "halfcheetah-medium-v2":        5000.0,
    "halfcheetah-medium-replay-v2": 4000.0,
    "halfcheetah-expert-v2":        10000.0,
    "hopper-medium-v2":             2000.0,
    "hopper-medium-replay-v2":      1500.0,
    "hopper-expert-v2":             3000.0,
    "walker2d-medium-v2":           3000.0,
    "walker2d-medium-replay-v2":    2500.0,
    "walker2d-expert-v2":           4000.0,
    "ant-medium-v2":                3500.0,
    "ant-medium-replay-v2":         3000.0,
}


# ──────────────────────────────────────────────────────────────────────────────
# Adaptive alpha schedule
# ──────────────────────────────────────────────────────────────────────────────

class AdaptiveAlpha:
    """
    Controls the real/synthetic mixing ratio based on εroll.

    α = clip(1 - ρ_max, α_min, 1.0)
    where ρ_max = η · (1 - γ) / εroll

    When εroll is small (diffusion model is accurate), ρ_max is large
    so α is small (more synthetic data allowed).
    When εroll is large (model drifted), α approaches 1.0 (mostly real data).

    Parameters
    ----------
    eta         : controls sensitivity (η in the paper, default 0.25)
    gamma       : discount factor
    alpha_min   : minimum real data fraction (never go below this)
    update_freq : how often to re-estimate εroll (in training steps)
    """

    def __init__(
        self,
        eta:         float = 0.25,
        gamma:       float = 0.99,
        alpha_min:   float = 0.3,
        update_freq: int   = 2_000,
    ):
        self.eta         = eta
        self.gamma       = gamma
        self.alpha_min   = alpha_min
        self.update_freq = update_freq
        self.current     = 1.0      # start pure real
        self._eroll_history = []

    def update(self, eroll: float) -> float:
        """Compute new alpha from εroll estimate."""
        self._eroll_history.append(eroll)
        rho_max    = self.eta * (1.0 - self.gamma) / (eroll + 1e-8)
        new_alpha  = float(np.clip(1.0 / (1.0 + rho_max), self.alpha_min, 1.0))
        self.current = new_alpha
        return new_alpha

    @property
    def rho(self) -> float:
        """Current synthetic ratio ρ = (1-α)/α."""
        return (1.0 - self.current) / (self.current + 1e-8)


# ──────────────────────────────────────────────────────────────────────────────
# Main training function
# ──────────────────────────────────────────────────────────────────────────────

def train_iql(args) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    obs_dim, action_dim, data_path = ENV_REGISTRY[args.env]

    # ── IQL Agent ─────────────────────────────────────────────────────────
    agent = IQLAgent(
        obs_dim     = obs_dim,
        action_dim  = action_dim,
        hidden_dims = (256, 256),
        expectile   = 0.7,
        temperature = 3.0,
        discount    = 0.99,
        tau         = 0.005,
        lr_q        = 3e-4,
        lr_v        = 3e-4,
        lr_pi       = 3e-4,
        device      = device,
    )

    # ── Replay buffer ─────────────────────────────────────────────────────
    real_buffer = ReplayBuffer(data_path, device)

    generator  = None
    normalizer = None
    if args.mode != "offline_only" and args.diffusion_ckpt:
        print(f"\nLoading diffusion model from {args.diffusion_ckpt} ...")
        try:
            # Import here to avoid circular dependency if not needed
            from generate import TrajectoryGenerator
            from data import DataNormalizer
            generator  = TrajectoryGenerator.from_checkpoint(args.diffusion_ckpt, device)
            normalizer = generator.normalizer
            print("Diffusion model loaded.")
        except Exception as e:
            print(f"Warning: could not load diffusion model: {e}")
            print("Falling back to offline_only mode.")
            generator = None

    # Set up augmented buffer
    target_return = TARGET_RETURNS.get(args.env, 3000.0)
    aug_buffer = AugmentedReplayBuffer(
        real_buffer   = real_buffer,
        generator     = generator,
        normalizer    = normalizer,
        alpha         = 1.0,       # start pure real
        target_return = target_return,
    )

    # Alpha schedule
    fixed_alpha    = args.alpha if args.mode == "ablation_fixed_alpha" else None
    adaptive_alpha = AdaptiveAlpha(eta=0.25, gamma=0.99, alpha_min=0.3)

    if fixed_alpha is not None:
        aug_buffer.set_alpha(fixed_alpha)
        print(f"Fixed alpha = {fixed_alpha}")
    else:
        print("Using adaptive alpha schedule (εroll-based)")

    # ── Evaluator ─────────────────────────────────────────────────────────
    evaluator = make_evaluator(args.env, device, n_episodes=10)

    # ── WandB ─────────────────────────────────────────────────────────────
    env_tag   = args.env.replace("-v2", "").replace("-", "_")
    run_name  = f"iql_{env_tag}_{args.mode}_s{args.seed}"
    output_dir = os.path.join("./checkpoints", args.env, "iql", args.mode)
    os.makedirs(output_dir, exist_ok=True)

    wandb.init(
        project = "disa-rl",
        name    = run_name,
        mode    = os.environ.get("WANDB_MODE", "online"),
        config  = dict(
            env          = args.env,
            mode         = args.mode,
            seed         = args.seed,
            num_steps    = args.num_steps,
            batch_size   = args.batch_size,
            alpha        = fixed_alpha,
            target_return = target_return,
        ),
    )

    print(f"\n{'='*55}")
    print(f"  IQL training: {args.env}")
    print(f"  Mode        : {args.mode}")
    print(f"  Steps       : {args.num_steps:,}")
    print(f"  Output      : {output_dir}/")
    print(f"{'='*55}\n")

    # ── Training loop ──────────────────────────────────────────────────────
    best_score = -float("inf")
    pbar = tqdm(total=args.num_steps, desc=f"IQL [{args.mode}]")

    for step in range(1, args.num_steps + 1):
        # Sample batch (real or mixed)
        batch   = aug_buffer.sample(args.batch_size)
        metrics = agent.update(batch)

        # Adaptive alpha update
        if (generator is not None
                and fixed_alpha is None
                and step % adaptive_alpha.update_freq == 0):
            # Sample a small set of real transitions to estimate εroll
            real_batch = real_buffer.sample(64)
            real_trajs_np = np.stack([
                np.concatenate([
                    real_batch["obs"].cpu().numpy(),
                    real_batch["action"].cpu().numpy(),
                    real_batch["reward"].cpu().numpy()[:, None],
                ], axis=-1)
            ], axis=0)   # crude: treat batch as a single "trajectory"

            try:
                eroll = generator.estimate_eroll(real_trajs_np.squeeze(0)[None])
                new_alpha = adaptive_alpha.update(eroll)
                aug_buffer.set_alpha(new_alpha)
                metrics["schedule/alpha"]  = new_alpha
                metrics["schedule/eroll"]  = eroll
                metrics["schedule/rho"]    = adaptive_alpha.rho
            except Exception:
                pass  # εroll estimation can fail gracefully

        # Logging
        if step % args.log_every == 0:
            metrics["step"] = step
            wandb.log(metrics, step=step)
            pbar.set_postfix(
                Q=f"{metrics['loss/q']:.3f}",
                V=f"{metrics['loss/v']:.3f}",
                π=f"{metrics['loss/actor']:.3f}",
            )

        # Evaluation
        if step % args.eval_every == 0:
            eval_metrics = evaluator.evaluate(agent.actor)
            if eval_metrics:
                wandb.log(eval_metrics, step=step)
                score = eval_metrics["eval/normalized"]
                print(f"\n[{step:>7d}]  normalized={score:.1f}  "
                      f"return={eval_metrics['eval/return_mean']:.1f}  "
                      f"alpha={aug_buffer.alpha:.3f}")

                # Save best checkpoint
                if score > best_score:
                    best_score = score
                    agent.save(os.path.join(output_dir, "best.pt"))

        # Periodic checkpoint
        if step % args.save_every == 0:
            agent.save(os.path.join(output_dir, f"step_{step:07d}.pt"))

        pbar.update(1)

    pbar.close()

    # Final save
    agent.save(os.path.join(output_dir, "final.pt"))
    print(f"\nTraining complete.  Best normalized score: {best_score:.1f}")
    print(f"Checkpoints saved to: {output_dir}/")
    wandb.finish()
    evaluator.close()


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IQL offline RL training")

    parser.add_argument("--env",    type=str, required=True, choices=list(ENV_REGISTRY.keys()))
    parser.add_argument("--mode",   type=str, default="augmented",
                        choices=["offline_only", "augmented", "ablation_fixed_alpha"])
    parser.add_argument("--diffusion_ckpt", type=str, default=None,
                        help="Path to diffusion model checkpoint (required for augmented mode)")
    parser.add_argument("--alpha",       type=float, default=0.5,
                        help="Fixed alpha for ablation_fixed_alpha mode")
    parser.add_argument("--num_steps",   type=int,   default=1_000_000)
    parser.add_argument("--batch_size",  type=int,   default=256)
    parser.add_argument("--eval_every",  type=int,   default=10_000)
    parser.add_argument("--log_every",   type=int,   default=1_000)
    parser.add_argument("--save_every",  type=int,   default=100_000)
    parser.add_argument("--seed",        type=int,   default=0)
    parser.add_argument("--wandb_entity", type=str,  default="")

    args = parser.parse_args()

    # Validate
    if args.mode != "offline_only" and args.diffusion_ckpt is None:
        # Auto-detect checkpoint
        auto_path = f"./checkpoints/{args.env}/offline_final.pt"
        if os.path.exists(auto_path):
            args.diffusion_ckpt = auto_path
            print(f"Auto-detected diffusion checkpoint: {auto_path}")
        else:
            print(f"Warning: --diffusion_ckpt not set and {auto_path} not found.")
            print("Falling back to offline_only mode.")
            args.mode = "offline_only"

    train_iql(args)
