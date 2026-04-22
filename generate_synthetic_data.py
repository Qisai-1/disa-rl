"""
Pre-generate synthetic trajectories from a trained diffusion model
and save them to disk as a .npz file.

Run this ONCE per environment before IQL training.
IQL then loads the saved file — no diffusion model needed during training.

Usage:
    python generate_synthetic_data.py --env halfcheetah-medium-v2
    python generate_synthetic_data.py --env hopper-medium-v2
    python generate_synthetic_data.py --env walker2d-medium-v2
    python generate_synthetic_data.py --env ant-medium-v2

Output:
    ./data/synthetic/<env>/synthetic_transitions.npz
    Keys: observations, actions, rewards, next_observations, terminals
"""

import argparse
import os
import sys
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate import TrajectoryGenerator, GenerationConfig
from reward_computer import RewardComputer

ENV_REGISTRY = {
    "halfcheetah-medium-v2":        (17, 6,   "./checkpoints/halfcheetah-medium-v2/diffusion/offline_final.pt",  5000.0),
    "halfcheetah-medium-replay-v2": (17, 6,   "./checkpoints/halfcheetah-medium-replay-v2/diffusion/offline_final.pt", 4000.0),
    "hopper-medium-v2":             (11, 3,   "./checkpoints/hopper-medium-v2/diffusion/offline_final.pt",       2000.0),
    "hopper-medium-replay-v2":      (11, 3,   "./checkpoints/hopper-medium-replay-v2/diffusion/offline_final.pt", 1500.0),
    "walker2d-medium-v2":           (17, 6,   "./checkpoints/walker2d-medium-v2/diffusion/offline_final.pt",     3000.0),
    "walker2d-medium-replay-v2":    (17, 6,   "./checkpoints/walker2d-medium-replay-v2/diffusion/offline_final.pt", 2500.0),
    "ant-medium-v2":                (111, 8,  "./checkpoints/ant-medium-v2/diffusion/offline_final.pt",          3500.0),
    "ant-medium-replay-v2":         (111, 8,  "./checkpoints/ant-medium-replay-v2/diffusion/offline_final.pt",   3000.0),
}


def generate_synthetic_data(env, n_transitions=1_000_000, batch_size=64,
                             nfe=20, cfg_scale=1.5, output_dir="./data/synthetic",
                             device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    obs_dim, action_dim, ckpt_path, target_return = ENV_REGISTRY[env]

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Diffusion checkpoint not found: {ckpt_path}\n"
            f"Run: python train.py --env {env}"
        )

    print(f"\n{'='*55}")
    print(f"  Environment   : {env}")
    print(f"  Transitions   : {n_transitions:,}")
    print(f"  Batch size    : {batch_size} trajectories")
    print(f"  NFE           : {nfe}  |  CFG scale: {cfg_scale}")
    print(f"  Target return : {target_return}")
    print(f"{'='*55}\n")

    generator = TrajectoryGenerator.from_checkpoint(ckpt_path, device)
    traj_len  = generator.model.T

    transitions_per_batch = batch_size * traj_len
    n_batches = int(np.ceil(n_transitions / transitions_per_batch))

    print(f"  Trajectory length : {traj_len}")
    print(f"  Batches needed    : {n_batches}")
    print(f"  Total transitions : {n_batches * transitions_per_batch:,}\n")

    gen_cfg = GenerationConfig(nfe=nfe, cfg_scale=cfg_scale, clip_actions=True)

    # Reward computer — analytic for D4RL locomotion, no diffusion needed
    reward_computer = RewardComputer.make(env, device=device)

    all_obs, all_actions, all_next_obs, all_dones = [], [], [], []

    for i in tqdm(range(n_batches), desc=f"Generating {env}"):
        result = generator.generate(
            n_trajectories=batch_size,
            target_return=target_return,
            gen_cfg=gen_cfg,
        )
        # Store obs, action, next_obs — rewards computed analytically below
        obs_b     = result["observations"]          # (B, T, obs_dim)
        actions_b = result["actions"]               # (B, T, action_dim)

        B, T, _ = obs_b.shape
        for b in range(B):
            for t in range(T):
                done     = (t == T - 1)
                next_obs = obs_b[b, t] if done else obs_b[b, t + 1]
                all_obs.append(obs_b[b, t])
                all_actions.append(actions_b[b, t])
                all_next_obs.append(next_obs)
                all_dones.append(float(done))

    obs      = np.stack(all_obs)[:n_transitions].astype(np.float32)
    actions  = np.stack(all_actions)[:n_transitions].astype(np.float32)
    next_obs = np.stack(all_next_obs)[:n_transitions].astype(np.float32)
    dones    = np.array(all_dones, dtype=np.float32)[:n_transitions]

    # Compute rewards analytically from (obs, action) — no NaN, exact values
    print("Computing rewards analytically...")
    rewards  = reward_computer.compute(obs, actions)

    # Clean any NaN/inf in obs (from diffusion) — rewards are analytic so clean
    obs      = np.nan_to_num(obs,      nan=0.0, posinf=0.0, neginf=0.0)
    next_obs = np.nan_to_num(next_obs, nan=0.0, posinf=0.0, neginf=0.0)
    actions  = np.clip(actions, -1.0, 1.0)

    print(f"\nGeneration complete.")
    print(f"  obs shape    : {obs.shape}")
    print(f"  reward range : [{rewards.min():.3f}, {rewards.max():.3f}]  mean={rewards.mean():.3f}")
    print(f"  action range : [{actions.min():.3f}, {actions.max():.3f}]")
    print(f"  nan count    : obs={np.isnan(obs).sum()}  rew={np.isnan(rewards).sum()}  act={np.isnan(actions).sum()}")

    save_dir  = os.path.join(output_dir, env)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "synthetic_transitions.npz")

    np.savez(save_path,
             observations=obs, actions=actions, rewards=rewards,
             next_observations=next_obs, terminals=dones)

    size_mb = os.path.getsize(save_path) / 1e6
    print(f"Saved → {save_path}  ({size_mb:.0f} MB)")
    return save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",           type=str,   required=True, choices=list(ENV_REGISTRY.keys()))
    parser.add_argument("--n_transitions", type=int,   default=1_000_000)
    parser.add_argument("--batch_size",    type=int,   default=64)
    parser.add_argument("--nfe",           type=int,   default=20)
    parser.add_argument("--cfg_scale",     type=float, default=1.5)
    parser.add_argument("--output_dir",    type=str,   default="./data/synthetic")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    generate_synthetic_data(
        env=args.env, n_transitions=args.n_transitions,
        batch_size=args.batch_size, nfe=args.nfe,
        cfg_scale=args.cfg_scale, output_dir=args.output_dir, device=device,
    )