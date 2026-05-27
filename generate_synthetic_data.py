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
from typing import Optional

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
from diffusion.value_guided import (
    value_guided_heun, td_relabel_rewards, q_anomaly_mask,
)
from diffusion.inverse_dynamics import load_idm
from reward_computer import RewardComputer

def get_env_info_for_generation(env_name, data_dir="./data", ckpt_dir="./checkpoints",
                                 traj_len=100, stride=50):
    """
    Auto-detect dims and a sane sub-trajectory return distribution.

    Bug fix: the diffusion model is conditioned on SUB-TRAJECTORY returns
    (T=100 steps), not full-episode returns. Conditioning on episode-level
    returns puts the signal ~64 sigma above the training distribution and
    causes mode-collapse to a single high-reward attractor.
    """
    data_path = os.path.join(data_dir, f"{env_name}.npz")
    ckpt_path = os.path.join(ckpt_dir, env_name, "diffusion", "offline_final.pt")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Diffusion checkpoint not found: {ckpt_path}")

    data     = np.load(data_path, allow_pickle=True)
    obs_dim  = int(data["observations"].shape[1])
    act_dim  = int(data["actions"].shape[1])

    rewards   = data["rewards"].astype(np.float32)
    terminals = data.get("terminals", np.zeros(len(rewards))).astype(bool)
    timeouts  = data.get("timeouts",  np.zeros(len(rewards))).astype(bool)
    done      = terminals | timeouts

    # Split into episodes, then take sliding-window sub-trajectory returns
    end_idxs = np.where(done)[0] + 1
    starts   = np.concatenate([[0], end_idxs[:-1]])

    sub_returns = []
    for s, e in zip(starts, end_idxs):
        ep_r = rewards[s:e]
        for st in range(0, len(ep_r) - traj_len + 1, stride):
            sub_returns.append(ep_r[st:st + traj_len].sum())
    if not sub_returns:
        raise ValueError(f"No sub-trajectories of length {traj_len} found in {data_path}.")
    sub_returns = np.asarray(sub_returns, dtype=np.float32)

    # Target = 90th percentile of SUB-TRAJ returns. This is in-distribution
    # for the conditioning signal the diffusion model was trained with.
    target_return = float(np.percentile(sub_returns, 90))
    print(f"  Auto-detected: obs={obs_dim}  act={act_dim}")
    print(f"  Sub-traj returns (T={traj_len}): "
          f"min={sub_returns.min():.1f}  p50={np.percentile(sub_returns,50):.1f}  "
          f"p90={target_return:.1f}  max={sub_returns.max():.1f}  "
          f"(n={len(sub_returns):,})")

    # Expose the full return distribution so callers can sample per-batch
    # for diversity rather than condition every trajectory on the same value.
    return obs_dim, act_dim, ckpt_path, target_return, sub_returns


def _load_iql_critic(env: str, obs_dim: int, action_dim: int, device,
                     iql_ckpt: Optional[str] = None):
    """Return (q_fn, v_fn, agent) where q_fn(s,a) and v_fn(s) are callables
    on torch tensors. Auto-discovers checkpoint at
    ./checkpoints/{env}/iql/offline_only/seed_0/final.pt unless overridden.
    """
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from iql.agent import IQLAgent

    if iql_ckpt is None:
        # Prefer offline_only baseline (real-data-only critic = in-distribution)
        for cand in [
            f"./checkpoints/{env}/iql/offline_only/seed_0/final.pt",
            f"./checkpoints/{env}/iql/offline_only/alpha1.0/seed_0/final.pt",
        ]:
            if os.path.exists(cand):
                iql_ckpt = cand; break
    if iql_ckpt is None or not os.path.exists(iql_ckpt):
        raise FileNotFoundError(
            f"No IQL critic found for VCDG. Train one first with "
            f"`python iql/train_iql.py --env {env} --mode offline_only --seed 0`."
        )
    agent = IQLAgent(obs_dim=obs_dim, action_dim=action_dim, device=device)
    agent.load(iql_ckpt)
    agent.q.eval(); agent.v.eval(); agent.actor.eval()

    def q_fn(s, a):
        q1, q2 = agent.q(s, a)
        return torch.min(q1, q2)
    def v_fn(s):
        return agent.v(s)
    print(f"  VCDG critic loaded from {iql_ckpt}")
    return q_fn, v_fn, agent


def generate_synthetic_data(env, n_transitions=1_000_000, batch_size=64,
                             nfe=20, cfg_scale=1.5, output_dir="./data/synthetic",
                             device=None, force_env=None,
                             return_sampling="topk",
                             use_vcdg=False, vcdg_iql_ckpt=None,
                             vcdg_guidance_scale=0.5,
                             vcdg_td_relabel=True,
                             vcdg_q_anomaly=True,
                             use_idm=False, idm_ckpt=None):
    """
    Parameters
    ----------
    return_sampling : how to choose target_return for each batch
        "topk"  : sample uniformly from the top-20% of real sub-traj returns
                  (default — produces high-but-in-distribution trajectories)
        "p90"   : fixed at 90th percentile (legacy behaviour)
        "real"  : sample from the empirical sub-traj return distribution
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    obs_dim, action_dim, ckpt_path, target_return, sub_returns = \
        get_env_info_for_generation(env)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Diffusion checkpoint not found: {ckpt_path}\n"
            f"Run: python train.py --env {env}"
        )

    topk_pool = sub_returns[sub_returns >= np.percentile(sub_returns, 80)]

    print(f"\n{'='*55}")
    print(f"  Environment      : {env}")
    print(f"  Transitions      : {n_transitions:,}")
    print(f"  Batch size       : {batch_size} trajectories")
    print(f"  NFE              : {nfe}  |  CFG scale: {cfg_scale}")
    print(f"  Return sampling  : {return_sampling}")
    print(f"  Target return    : p90={target_return:.1f}  "
          f"topk-pool=[{topk_pool.min():.1f},{topk_pool.max():.1f}]")
    print(f"{'='*55}\n")

    generator = TrajectoryGenerator.from_checkpoint(ckpt_path, device)
    traj_len  = generator.model.T

    # If the checkpoint was trained with QCD (Q-conditioning), we need
    # target_q sequence instead of target_return. Auto-detect.
    cond_kind = getattr(generator.normalizer, "cond_kind", "return")
    if cond_kind == "q":
        from diffusion.q_conditional import sample_q_target
        # Reconstruct training Q distribution from the normalizer stats:
        # samples ~ N(q_mean, q_std); we'll draw target Q from upper quantile.
        q_mean = generator.normalizer.q_mean
        q_std  = generator.normalizer.q_std
        # Use the upper-half of an approximate gaussian as the top-quantile pool
        approx_q_pool = q_mean + q_std * np.random.randn(10_000)
        approx_q_pool = approx_q_pool[approx_q_pool >= np.quantile(approx_q_pool, 0.8)]
        print(f"  cond_kind='q' — sampling target_q from top-quantile pool "
              f"(mean={q_mean:.2f}, std={q_std:.2f})")
    else:
        approx_q_pool = None

    # VCDG critic (only loaded if requested)
    q_fn = v_fn = agent_critic = None
    if use_vcdg:
        print("\n─── VCDG: loading IQL critic ──────────────────────────")
        q_fn, v_fn, agent_critic = _load_iql_critic(
            env, obs_dim, action_dim, device, iql_ckpt=vcdg_iql_ckpt,
        )
        print(f"  guidance_scale={vcdg_guidance_scale}  "
              f"td_relabel={vcdg_td_relabel}  q_anomaly={vcdg_q_anomaly}\n")

    # Inverse Dynamics Model (Pillar 1) — replaces diffusion action head
    idm_net = idm_s_mean = idm_s_std = None
    if use_idm:
        if idm_ckpt is None:
            idm_ckpt = f"./checkpoints/{env}/idm/idm.pt"
        if not os.path.exists(idm_ckpt):
            raise FileNotFoundError(
                f"IDM checkpoint not found: {idm_ckpt}\n"
                f"Train one with: python diffusion/inverse_dynamics.py --env {env}"
            )
        print(f"\n─── IDM: loading inverse-dynamics action head ──────────")
        idm_net, idm_s_mean, idm_s_std = load_idm(idm_ckpt, device)
        idm_s_mean_t = torch.from_numpy(np.asarray(idm_s_mean, dtype=np.float32)).to(device)
        idm_s_std_t  = torch.from_numpy(np.asarray(idm_s_std,  dtype=np.float32)).to(device)
        print(f"  loaded {idm_ckpt}\n")

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

        # Per-batch target_return — diversifies conditioning so the model
        # doesn't always sit at the same point on the return-conditioned manifold
        if return_sampling == "topk":
            tr_batch = float(np.random.choice(topk_pool))
        elif return_sampling == "real":
            tr_batch = float(np.random.choice(sub_returns))
        else:  # "p90"
            tr_batch = target_return

        # QCD path: sample a target_q from the top-quantile pool per batch.
        target_q_batch = None
        if approx_q_pool is not None:
            target_q_batch = float(np.random.choice(approx_q_pool))

        result = generator.generate(
            n_trajectories=batch_size,
            initial_states=init_states,
            target_return=tr_batch,
            target_q=target_q_batch,
            gen_cfg=gen_cfg,
            force_env=force_env,
            value_fn=v_fn if use_vcdg else None,
            vcdg_guidance_scale=vcdg_guidance_scale,
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

    # Compute rewards — either analytically (default) or via TD relabel (VCDG)
    if use_vcdg and vcdg_td_relabel:
        print("VCDG: relabelling rewards via r = V(s) - gamma * V(s')")
        rewards = td_relabel_rewards(
            observations=obs, next_observations=next_obs,
            value_fn=v_fn, device=device, discount=0.99,
        )
    else:
        print("Computing rewards analytically...")
        rewards = reward_computer.compute(obs, actions)

    # Clean NaN/inf
    obs      = np.nan_to_num(obs,      nan=0.0, posinf=0.0, neginf=0.0)
    next_obs = np.nan_to_num(next_obs, nan=0.0, posinf=0.0, neginf=0.0)
    actions  = np.clip(actions, -1.0, 1.0)

    # VCDG: Q-anomaly filter
    if use_vcdg and vcdg_q_anomaly:
        real = np.load(f"./data/{env}.npz", allow_pickle=True)
        real_obs_for_q = real["observations"].astype(np.float32)
        real_act_for_q = real["actions"].astype(np.float32)
        # Subsample for speed
        if len(real_obs_for_q) > 50_000:
            sub = np.random.choice(len(real_obs_for_q), 50_000, replace=False)
            real_obs_for_q = real_obs_for_q[sub]
            real_act_for_q = real_act_for_q[sub]
        keep = q_anomaly_mask(
            observations=obs, actions=actions,
            q_fn=q_fn, v_fn=v_fn, device=device,
            real_obs=real_obs_for_q, real_act=real_act_for_q,
            quantile=0.99,
        )
        if keep.sum() > 0:
            obs=obs[keep]; actions=actions[keep]; rewards=rewards[keep]
            next_obs=next_obs[keep]; dones=dones[keep]

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
    fname = "synthetic_transitions_vcdg.npz" if use_vcdg else "synthetic_transitions.npz"
    save_path = os.path.join(save_dir, fname)

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
    parser.add_argument("--cfg_scale",     type=float, default=1.5,
                        help="CFG scale. 1.0 = no guidance. 1.2-1.5 typical.")
    parser.add_argument("--output_dir",    type=str,   default="./data/synthetic")
    parser.add_argument("--force_env",     type=str,   default=None,
                        help="Override env name for reward computer (e.g. walker2d-medium-v2)")
    parser.add_argument("--return_sampling", type=str, default="real",
                        choices=["topk", "p90", "real"],
                        help="How to choose target_return per batch")
    # ── VCDG (Value-Calibrated Diffusion Generation) ─────────────────────
    parser.add_argument("--vcdg", action="store_true",
                        help="Enable VCDG (value-guided sampling + TD relabel + Q-anomaly filter)")
    parser.add_argument("--vcdg_iql_ckpt", type=str, default=None,
                        help="Path to offline_only IQL critic. Auto-detects if None.")
    parser.add_argument("--vcdg_guidance_scale", type=float, default=0.5,
                        help="lambda_v for value gradient guidance")
    parser.add_argument("--no_vcdg_td_relabel", action="store_true",
                        help="Disable TD reward relabeling under VCDG")
    parser.add_argument("--no_vcdg_q_anomaly", action="store_true",
                        help="Disable Q-anomaly filter under VCDG")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    generate_synthetic_data(
        env=args.env, n_transitions=args.n_transitions,
        batch_size=args.batch_size, nfe=args.nfe,
        cfg_scale=args.cfg_scale, output_dir=args.output_dir, device=device,
        force_env=args.force_env, return_sampling=args.return_sampling,
        use_vcdg=args.vcdg, vcdg_iql_ckpt=args.vcdg_iql_ckpt,
        vcdg_guidance_scale=args.vcdg_guidance_scale,
        vcdg_td_relabel=not args.no_vcdg_td_relabel,
        vcdg_q_anomaly=not args.no_vcdg_q_anomaly,
    )