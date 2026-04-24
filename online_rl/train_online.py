"""
Online RL training loop for DiSA-RL.

Implements the three connection points between offline and online:

Connection 1 — Policy and value continuity
    SAC actor and critic initialized from offline IQL weights.
    5k critic-only recalibration phase at the start.
    Conservative blend decays from 1.0 → 0.0 over transition_steps.

Connection 2 — Data continuity
    OnlineBuffer pre-populated with offline D4RL + synthetic data.
    Real env data added as online training progresses.

Connection 3 — Generative model continuity
    Same pψ used throughout. Background generator provides fresh
    synthetic data. Fine-tuned on real data every finetune_every steps.
    εroll controls adaptive ρ (synthetic ratio).

Usage:
    python online_rl/train_online.py \
        --env halfcheetah-medium-v2 \
        --iql_ckpt ./checkpoints/halfcheetah-medium-v2/iql/augmented/best.pt \
        --diffusion_ckpt ./checkpoints/halfcheetah-medium-v2/diffusion/offline_final.pt \
        --synthetic_data ./data/synthetic/halfcheetah-medium-v2/synthetic_transitions.npz
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

from online_rl.sac import SACAgent
from online_rl.online_buffer import OnlineBuffer, RealEnvBuffer, OfflineSyntheticBuffer
from online_rl.async_generator import AsyncSyntheticGenerator
from iql.evaluator import make_evaluator


# ──────────────────────────────────────────────────────────────────────────────
# Environment registry
# ──────────────────────────────────────────────────────────────────────────────

def get_env_dims(env_name: str, data_dir: str = "./data"):
    """Read obs_dim and action_dim directly from dataset — no hardcoded registry."""
    import numpy as np
    path = os.path.join(data_dir, f"{env_name}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    data = np.load(path, allow_pickle=True)
    return int(data["observations"].shape[1]), int(data["actions"].shape[1])

def get_target_return(env_name: str, data_dir: str = "./data", percentile: int = 90) -> float:
    """Compute target return as p90 of episode returns from real dataset."""
    import numpy as np
    path = os.path.join(data_dir, f"{env_name}.npz")
    if not os.path.exists(path):
        return 3000.0  # fallback
    data      = np.load(path, allow_pickle=True)
    rewards   = data["rewards"]
    terminals = data.get("terminals", np.zeros(len(rewards)))
    timeouts  = data.get("timeouts",  np.zeros(len(rewards)))
    done      = (terminals + timeouts) > 0
    ep_returns, ep_ret = [], 0.0
    for r, d in zip(rewards, done):
        ep_ret += r
        if d:
            ep_returns.append(ep_ret)
            ep_ret = 0.0
    if ep_ret > 0:
        ep_returns.append(ep_ret)
    if not ep_returns:
        raise ValueError(
            f"Could not compute episode returns from {path}. "
            f"Check that terminals/timeouts mark episode boundaries."
        )
    return float(np.percentile(ep_returns, percentile))


# ──────────────────────────────────────────────────────────────────────────────
# εroll estimator (MMD-based)
# ──────────────────────────────────────────────────────────────────────────────

def estimate_eroll(real_obs: np.ndarray, syn_obs: np.ndarray) -> float:
    """
    Estimate εroll via Maximum Mean Discrepancy (MMD) with RBF kernel.

    MMD measures the distance between the real and synthetic data
    distributions. Low εroll = diffusion model is well-aligned with
    real data → can use more synthetic data.
    High εroll = model has drifted → rely more on real data.

    Uses a subsample for efficiency.
    """
    n = min(512, len(real_obs), len(syn_obs))
    r = real_obs[np.random.choice(len(real_obs), n, replace=False)]
    s = syn_obs[np.random.choice(len(syn_obs), n, replace=False)]

    r = torch.from_numpy(r).float()
    s = torch.from_numpy(s).float()

    # RBF kernel with median heuristic bandwidth
    with torch.no_grad():
        all_data = torch.cat([r, s], dim=0)
        dists    = torch.cdist(all_data, all_data)
        sigma    = dists.median().item() + 1e-8

        def rbf(x, y):
            d = torch.cdist(x, y)
            return torch.exp(-d ** 2 / (2 * sigma ** 2))

        kxx = rbf(r, r).mean()
        kyy = rbf(s, s).mean()
        kxy = rbf(r, s).mean()
        mmd = (kxx + kyy - 2 * kxy).item()

    return float(np.clip(mmd, 0.0, None))


# ──────────────────────────────────────────────────────────────────────────────
# Adaptive ρ schedule
# ──────────────────────────────────────────────────────────────────────────────

def compute_rho(eroll: float, eta: float = 0.25, gamma: float = 0.99) -> float:
    """
    ρ = η(1-γ) / εroll

    Higher eroll → lower ρ (less synthetic).
    Lower eroll  → higher ρ (more synthetic).
    Clipped to [0.1, 5.0].
    """
    rho = eta * (1.0 - gamma) / (eroll + 1e-8)
    return float(np.clip(rho, 0.1, 5.0))


# ──────────────────────────────────────────────────────────────────────────────
# Environment rollout
# ──────────────────────────────────────────────────────────────────────────────

def collect_real_data(
    env,
    agent:        SACAgent,
    real_buffer:  RealEnvBuffer,
    n_steps:      int,
    deterministic: bool = False,
) -> Dict:
    """
    Collect n_steps transitions from the environment.
    Returns stats for logging.
    """
    ep_returns = []
    ep_return  = 0.0
    obs, _     = env.reset()

    for _ in range(n_steps):
        action = agent.act(obs, deterministic=deterministic)
        action = np.clip(action, -1.0, 1.0)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        real_buffer.add(obs, action, reward, next_obs, done)
        ep_return += reward
        obs = next_obs

        if done:
            ep_returns.append(ep_return)
            ep_return = 0.0
            obs, _ = env.reset()

    return {
        "real/ep_returns": ep_returns,
        "real/mean_return": np.mean(ep_returns) if ep_returns else 0.0,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main training function
# ──────────────────────────────────────────────────────────────────────────────

def train_online(args) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_dim, action_dim = get_env_dims(args.env)

    # ── SAC Agent ─────────────────────────────────────────────────────────
    agent = SACAgent(
        obs_dim    = obs_dim,
        action_dim = action_dim,
        hidden_dims= (256, 256),
        lr_actor   = 3e-4,
        lr_critic  = 3e-4,
        lr_alpha   = 3e-4,
        gamma      = 0.99,
        tau        = 0.005,
        device     = device,
    )

    # Connection 1: Load offline IQL weights
    if args.iql_ckpt and os.path.exists(args.iql_ckpt):
        agent.load_from_iql(args.iql_ckpt)
    else:
        print("WARNING: No IQL checkpoint — starting from random init")

    # ── Environment ───────────────────────────────────────────────────────
    try:
        import gymnasium as gym
        from iql.evaluator import DATASET_TO_GYM
        gym_name = DATASET_TO_GYM.get(args.env, args.env)
        env      = gym.make(gym_name)
        print(f"Environment: {gym_name}")
    except Exception as e:
        print(f"Could not create environment: {e}")
        return

    # ── Buffers ───────────────────────────────────────────────────────────
    # Connection 2: Pre-populate with offline synthetic data
    real_buffer = RealEnvBuffer(obs_dim, action_dim,
                                max_size=args.real_buffer_size, device=device)
    offline_syn = OfflineSyntheticBuffer(args.synthetic_data, device)

    # Entropy alignment using offline synthetic data as obs sample
    sample_obs = torch.from_numpy(offline_syn.obs[:1024]).float()
    agent.align_entropy_with_iql(sample_obs)

    # ── Async generator ───────────────────────────────────────────────────
    # Connection 3: Same pψ generates fresh synthetic during online training
    async_gen = None
    if args.diffusion_ckpt and os.path.exists(args.diffusion_ckpt):
        async_gen = AsyncSyntheticGenerator(
            ckpt_path      = args.diffusion_ckpt,
            env_name       = args.env,
            target_return  = get_target_return(args.env),
            batch_size     = 32,
            nfe            = args.nfe,
            cfg_scale      = 1.5,
            finetune_steps = args.finetune_steps,
            device_str     = str(device),
        )
        async_gen.start()
        # Let generator warm up before training starts
        import time; time.sleep(5)

    online_buffer = OnlineBuffer(
        real_buffer   = real_buffer,
        offline_syn   = offline_syn,
        fresh_syn_queue = async_gen.data_queue if async_gen else None,
        rho           = 1.0,   # start 50/50
        device        = device,
    )

    # ── Evaluator ─────────────────────────────────────────────────────────
    evaluator = make_evaluator(args.env, device, n_episodes=10)

    # ── WandB ─────────────────────────────────────────────────────────────
    env_tag    = args.env.replace("-v2","").replace("-","_")
    run_name   = f"online_{env_tag}_s{args.seed}"
    output_dir = os.path.join("./checkpoints", args.env, "online", f"seed_{args.seed}")
    os.makedirs(output_dir, exist_ok=True)

    wandb.init(
        project = "disa-rl",
        name    = run_name,
        mode    = os.environ.get("WANDB_MODE", "online"),
        config  = vars(args),
    )

    print(f"\n{'='*60}")
    print(f"  Online RL: {args.env}  seed={args.seed}")
    print(f"  Steps       : {args.num_steps:,}")
    print(f"  Critic warmup: {args.critic_warmup_steps:,}")
    print(f"  Conservative transition: {args.transition_steps:,}")
    print(f"  Fine-tune every: {args.finetune_every:,} real steps")
    print(f"  Min real steps: {args.min_real_steps:,}")
    print(f"{'='*60}\n")

    # ── Training loop ──────────────────────────────────────────────────────
    # Track offline baseline for rollback protection
    offline_score     = None
    below_baseline    = 0
    best_online_score = -float("inf")
    last_finetune_at  = 0
    eroll             = 1.0
    rho               = 1.0

    pbar = tqdm(total=args.num_steps, desc=f"Online [{args.env}]")

    for step in range(1, args.num_steps + 1):

        # ── Collect real data (1 step per training step) ───────────────
        rollout = collect_real_data(env, agent, real_buffer, n_steps=1)

        # ── Sample mixed batch ─────────────────────────────────────────
        batch = online_buffer.sample(args.batch_size)

        # ── Determine training phase ───────────────────────────────────
        # Phase 1: critic-only recalibration
        critic_only = (step <= args.critic_warmup_steps)

        # Conservative blend: decays linearly from 1.0 → 0.0
        if step <= args.transition_steps:
            conservative_weight = 1.0 - (step / args.transition_steps)
        else:
            conservative_weight = 0.0

        # ── SAC update ─────────────────────────────────────────────────
        metrics = agent.update(
            batch,
            critic_only         = critic_only,
            conservative_weight = conservative_weight,
        )

        # ── εroll update and adaptive ρ ────────────────────────────────
        if step % args.eroll_every == 0 and len(real_buffer) >= 256:
            real_recent = real_buffer.recent(256)
            syn_sample  = offline_syn.obs[
                np.random.choice(len(offline_syn), 256, replace=False)
            ]
            eroll = estimate_eroll(real_recent["obs"], syn_sample)
            rho   = compute_rho(eroll, eta=0.25, gamma=0.99)
            online_buffer.set_rho(rho)
            metrics["online/eroll"] = eroll
            metrics["online/rho"]   = rho

        # ── Diffusion fine-tuning trigger ──────────────────────────────
        real_count = len(real_buffer)
        if (async_gen is not None
                and real_count >= args.min_real_steps
                and real_count - last_finetune_at >= args.finetune_every):
            # Save real data to temp file for generator process
            tmp_path = os.path.join(output_dir, "real_data_for_finetune.npz")
            recent   = real_buffer.recent(min(real_count, 50_000))
            np.savez(tmp_path, observations=recent["obs"], actions=recent["action"])
            async_gen.finetune(tmp_path)
            last_finetune_at = real_count
            metrics["online/finetune_triggered"] = 1.0

        # ── Logging ────────────────────────────────────────────────────
        if step % args.log_every == 0:
            metrics.update(online_buffer.stats())
            metrics["online/step"]          = step
            metrics["online/real_collected"] = len(real_buffer)
            metrics["online/critic_only"]   = float(critic_only)
            if async_gen:
                metrics["online/queue_size"] = async_gen.queue_size
            wandb.log(metrics, step=step)
            pbar.set_postfix(
                Q=f"{metrics.get('online/critic_loss', 0):.3f}",
                ρ=f"{rho:.2f}",
                real=f"{len(real_buffer):,}",
            )

        # ── Evaluation ─────────────────────────────────────────────────
        if step % args.eval_every == 0:
            eval_metrics = evaluator.evaluate(agent.actor)
            if eval_metrics:
                score = eval_metrics["eval/normalized"]
                wandb.log(eval_metrics, step=step)
                print(f"\n[{step:>7d}]  score={score:.1f}  "
                      f"real={len(real_buffer):,}  "
                      f"rho={rho:.2f}  eroll={eroll:.4f}")

                # Record offline baseline at first eval
                if offline_score is None:
                    offline_score = score
                    print(f"  Offline baseline set: {offline_score:.1f}")

                # Save best
                if score > best_online_score:
                    best_online_score = score
                    agent.save(os.path.join(output_dir, "best.pt"))

                # Rollback protection
                if offline_score and score < offline_score * 0.9:
                    below_baseline += 1
                    if below_baseline >= 3:
                        print(f"\nPerformance dropped below 90% of offline "
                              f"baseline ({score:.1f} < {offline_score*0.9:.1f})")
                        print("Rolling back to offline policy...")
                        if args.iql_ckpt:
                            agent.load_from_iql(args.iql_ckpt)
                            below_baseline = 0
                else:
                    below_baseline = 0

        pbar.update(1)

    pbar.close()

    # ── Cleanup ────────────────────────────────────────────────────────────
    agent.save(os.path.join(output_dir, "final.pt"))
    print(f"\nDone. Best online score: {best_online_score:.1f}")
    print(f"Checkpoints: {output_dir}/")

    if async_gen:
        async_gen.stop()
    env.close()
    evaluator.close()
    wandb.finish()


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from typing import Dict

    parser = argparse.ArgumentParser(description="DiSA-RL online training")
    parser.add_argument("--env",  type=str, required=True,
                        help="D4RL dataset name e.g. halfcheetah-medium-v2")
    parser.add_argument("--iql_ckpt", type=str, default=None,
                        help="Path to offline IQL checkpoint (best.pt)")
    parser.add_argument("--diffusion_ckpt", type=str, default=None,
                        help="Path to diffusion model checkpoint")
    parser.add_argument("--synthetic_data", type=str, default=None,
                        help="Path to pre-generated synthetic .npz")

    # Training
    parser.add_argument("--num_steps",          type=int,   default=500_000)
    parser.add_argument("--batch_size",          type=int,   default=256)
    parser.add_argument("--critic_warmup_steps", type=int,   default=5_000,
                        help="Steps to update critic only (value recalibration)")
    parser.add_argument("--transition_steps",    type=int,   default=20_000,
                        help="Steps over which conservative weight decays 1→0")
    parser.add_argument("--real_buffer_size",    type=int,   default=1_000_000)

    # Adaptive ρ
    parser.add_argument("--eroll_every",    type=int, default=2_000)

    # Diffusion fine-tuning
    parser.add_argument("--min_real_steps",  type=int,   default=10_000,
                        help="Collect this many real steps before first fine-tune")
    parser.add_argument("--finetune_every",  type=int,   default=10_000,
                        help="Fine-tune diffusion every N new real transitions")
    parser.add_argument("--finetune_steps",  type=int,   default=1_000)
    parser.add_argument("--nfe",             type=int,   default=20)

    # Logging
    parser.add_argument("--eval_every",     type=int, default=5_000)
    parser.add_argument("--log_every",      type=int, default=1_000)
    parser.add_argument("--seed",           type=int, default=0)

    args = parser.parse_args()

    # Auto-detect paths
    if args.iql_ckpt is None:
        auto = f"./checkpoints/{args.env}/iql/augmented/best.pt"
        if os.path.exists(auto):
            args.iql_ckpt = auto
            print(f"Auto-detected IQL ckpt: {auto}")

    if args.diffusion_ckpt is None:
        auto = f"./checkpoints/{args.env}/diffusion/offline_final.pt"
        if os.path.exists(auto):
            args.diffusion_ckpt = auto
            print(f"Auto-detected diffusion ckpt: {auto}")

    if args.synthetic_data is None:
        auto = f"./data/synthetic/{args.env}/synthetic_transitions.npz"
        if os.path.exists(auto):
            args.synthetic_data = auto
            print(f"Auto-detected synthetic data: {auto}")
        else:
            print(f"ERROR: synthetic data not found at {auto}")
            print(f"Run first: python generate_synthetic_data.py --env {args.env}")
            exit(1)

    train_online(args)