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

# Add diffusion/ to path so its internal imports resolve correctly
_root = os.path.dirname(os.path.abspath(__file__))
_diff = os.path.join(_root, "diffusion")
for _p in [_root, _diff]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from diffusion.generate import TrajectoryGenerator, GenerationConfig
from reward_computer import RewardComputer

def get_env_info_for_generation(env_name, data_dir="./data", ckpt_dir="./checkpoints"):
    """
    Auto-detect all environment info from dataset and checkpoint.
    No hardcoded dims or target returns.
    """
    data_path = os.path.join(data_dir, f"{env_name}.npz")
    ckpt_path = os.path.join(ckpt_dir, env_name, "diffusion", "offline_final.pt")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Diffusion checkpoint not found: {ckpt_path}")

    # Read dims from dataset
    data     = np.load(data_path, allow_pickle=True)
    obs_dim  = int(data["observations"].shape[1])
    act_dim  = int(data["actions"].shape[1])

    # Compute target return as 90th percentile of episode returns
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
        raise ValueError(f"No episode boundaries found in {data_path}. Check terminals/timeouts.")
    target_return = float(np.percentile(ep_returns, 90))
    print(f"  Auto-detected: obs={obs_dim}  act={act_dim}  "
          f"target_return={target_return:.1f} (p90 of {len(ep_returns)} episodes)")

    return obs_dim, act_dim, ckpt_path, target_return


def generate_synthetic_data(env, n_transitions=1_000_000, batch_size=64,
                             nfe=20, cfg_scale=1.5, output_dir="./data/synthetic",
                             device=None, force_env=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    obs_dim, action_dim, ckpt_path, target_return = get_env_info_for_generation(env)

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

    # Reward computer — use force_env if specified (overrides checkpoint detection)
    reward_env = force_env if force_env else env
    reward_computer = RewardComputer.make(reward_env, device=device)

    # Load real initial states for conditioning diversity
    real_data_path = f"./data/{env}.npz"
    real_obs_for_init = None
    if os.path.exists(real_data_path):
        _real = np.load(real_data_path, allow_pickle=True)
        real_obs_for_init = _real["observations"].astype(np.float32)
        print(f"Sampling initial states from real data ({len(real_obs_for_init):,} states)")

    all_obs, all_actions, all_next_obs, all_dones = [], [], [], []

    for i in tqdm(range(n_batches), desc=f"Generating {env}"):
        # Sample real initial states for CFG conditioning
        if real_obs_for_init is not None:
            idx = np.random.choice(len(real_obs_for_init), size=batch_size, replace=True)
            init_states = real_obs_for_init[idx]
        else:
            init_states = None

        result = generator.generate(
            n_trajectories=batch_size,
            initial_states=init_states,
            target_return=target_return,
            gen_cfg=gen_cfg,
            force_env=force_env,
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

    # Clean NaN/inf
    obs      = np.nan_to_num(obs,      nan=0.0, posinf=0.0, neginf=0.0)
    next_obs = np.nan_to_num(next_obs, nan=0.0, posinf=0.0, neginf=0.0)
    actions  = np.clip(actions, -1.0, 1.0)

    # OOD filter: remove transitions outside real data distribution
    real_data_path = f"./data/{env}.npz"
    if os.path.exists(real_data_path):
        real     = np.load(real_data_path, allow_pickle=True)
        real_obs = real["observations"].astype(np.float32)
        real_r   = real["rewards"].astype(np.float32)
        # Use per-dim percentile bounds (more robust than sigma for non-Gaussian obs)
        obs_lo = np.percentile(real_obs, 0.5, axis=0)
        obs_hi = np.percentile(real_obs, 99.5, axis=0)
        # Allow 10% of dims to be OOD (handles ant's 111-dim complexity)
        max_ood_frac = 0.1
        n_ood = ((obs < obs_lo) | (obs > obs_hi)).sum(axis=1)
        in_bounds = (n_ood / obs.shape[1]) <= max_ood_frac
        n_before  = len(obs)
        n_kept    = int(in_bounds.sum())
        print(f"\nOOD filter: kept {n_kept:,}/{n_before:,} ({100*n_kept/n_before:.1f}%)")
        if n_kept > 0:
            obs=obs[in_bounds]; actions=actions[in_bounds]
            rewards=rewards[in_bounds]; next_obs=next_obs[in_bounds]; dones=dones[in_bounds]
            rewards = np.clip(rewards, real_r.min()-real_r.std(), real_r.max()+real_r.std())
            if len(obs) < n_transitions:
                idx = np.random.choice(len(obs), size=n_transitions-len(obs), replace=True)
                obs=np.concatenate([obs,obs[idx]]); actions=np.concatenate([actions,actions[idx]])
                rewards=np.concatenate([rewards,rewards[idx]]); next_obs=np.concatenate([next_obs,next_obs[idx]])
                dones=np.concatenate([dones,dones[idx]])
                print(f"  Padded to {len(obs):,} by resampling.")
        else:
            print("  WARNING: all transitions OOD — check diffusion model")
    else:
        rewards = np.clip(rewards, -10.0, 15.0)

    print(f"\nGeneration complete.")
    print(f"  obs shape    : {obs.shape}")
    print(f"  reward range : [{rewards.min():.3f}, {rewards.max():.3f}]  mean={rewards.mean():.3f}  std={rewards.std():.3f}")
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
    parser.add_argument("--env",           type=str,   required=True,
                        help="D4RL dataset name e.g. halfcheetah-medium-v2")
    parser.add_argument("--n_transitions", type=int,   default=1_000_000)
    parser.add_argument("--batch_size",    type=int,   default=64)
    parser.add_argument("--nfe",           type=int,   default=20)
    parser.add_argument("--cfg_scale",     type=float, default=1.5)
    parser.add_argument("--output_dir",    type=str,   default="./data/synthetic")
    parser.add_argument("--force_env",     type=str,   default=None,
                        help="Override env name for reward computer (e.g. walker2d-medium-v2)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    generate_synthetic_data(
        env=args.env, n_transitions=args.n_transitions,
        batch_size=args.batch_size, nfe=args.nfe,
        cfg_scale=args.cfg_scale, output_dir=args.output_dir, device=device,
        force_env=args.force_env,
    )