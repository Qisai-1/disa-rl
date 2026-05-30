#!/usr/bin/env python3
"""
Regenerate synthetic data using a K-member diffusion ensemble + pairwise-
disagreement filter.

Algorithm:
  For each batch of conditions:
    1. Each ensemble member denoises the same condition (GTA partial-noising
       seed). Get K trajectories per condition.
    2. Compute disagreement: mean pairwise MSE across the K obs-trajectories
       (per condition).
    3. Keep conditions where disagreement < quantile threshold (default
       bottom 50%) — high agreement = ensemble trusts the generated traj.
    4. Save the first ensemble member's trajectory for each kept condition.

This is the "calibrated stitching" thesis's gen-time uncertainty filter,
the analog of CAPA's training-time gate.

Usage:
  python scripts/regen_with_ensemble_filter.py \
      --env hopper-medium-replay-v2 \
      --ensemble_dir_pattern 'checkpoints/{env}/diffusion_ens_s{seed}/offline_final.pt' \
      --K 3 \
      --keep_quantile 0.5 \
      --n_transitions 1000000 \
      --output_dir data/synthetic_ens_filtered

Output:
  data/synthetic_ens_filtered/<env>/synthetic_transitions.npz   (same schema as GTA syn)
  data/synthetic_ens_filtered/<env>/disagreement_stats.npz      (audit data)
"""
from __future__ import annotations
import argparse, os, sys, time
import numpy as np
import torch
from tqdm import tqdm

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_diff = os.path.join(_root, "diffusion")
for _p in (_root, _diff):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from diffusion.generate import TrajectoryGenerator


def build_real_subtrajs(env, traj_len, obs_dim, action_dim, data_dir="./data", stride=10):
    """Copy of generate_synthetic_data.build_real_subtrajs."""
    real = np.load(f"{data_dir}/{env}.npz")
    obs = real["observations"].astype(np.float32)
    act = real["actions"].astype(np.float32)
    rew = real["rewards"].astype(np.float32)
    term = real["terminals"].astype(np.float32)
    timeout = real.get("timeouts", np.zeros_like(term)).astype(np.float32)
    starts = np.concatenate([[0], np.where(term | timeout)[0] + 1])
    subs, rets = [], []
    for i in range(len(starts) - 1):
        s, e = starts[i], starts[i+1]
        if e - s < traj_len:
            continue
        for j in range(s, e - traj_len + 1, stride):
            window = np.concatenate([obs[j:j+traj_len], act[j:j+traj_len]], axis=1)
            subs.append(window)
            rets.append(rew[j:j+traj_len].sum())
    return np.asarray(subs, dtype=np.float32), np.asarray(rets, dtype=np.float32)


def load_ensemble(env: str, K: int, pattern: str, device) -> list:
    paths = [pattern.format(env=env, seed=k) for k in range(K)]
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing ensemble member: {p}")
    gens = [TrajectoryGenerator.from_checkpoint(p, device) for p in paths]
    print(f"Loaded ensemble of {K} members for {env}")
    return gens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True)
    ap.add_argument("--ensemble_dir_pattern", type=str,
                    default="checkpoints/{env}/diffusion_ens_s{seed}/offline_final.pt")
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--keep_quantile", type=float, default=0.5,
                    help="Keep trajectories with disagreement below this quantile.")
    ap.add_argument("--n_transitions", type=int, default=1_000_000)
    ap.add_argument("--traj_len", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--gta_noise_ratio", type=float, default=0.5)
    ap.add_argument("--gta_return_alpha", type=float, default=1.2)
    ap.add_argument("--nfe", type=int, default=20)
    ap.add_argument("--cfg_scale", type=float, default=1.5)
    ap.add_argument("--output_dir", type=str, default="data/synthetic_ens_filtered")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the K ensemble members
    gens = load_ensemble(args.env, args.K, args.ensemble_dir_pattern, device)
    nrm = gens[0].normalizer  # all members share the same data normalizer

    # Read dims from a real data file
    real = np.load(f"./data/{args.env}.npz")
    obs_dim = real["observations"].shape[1]
    action_dim = real["actions"].shape[1]

    # Pre-build real sub-trajectories for GTA seeding
    subs, rets = build_real_subtrajs(args.env, args.traj_len, obs_dim, action_dim)
    print(f"  {len(subs):,} real sub-trajectories available for GTA seeding")

    # We oversample to compensate for filter rejection
    n_batches = int(np.ceil(args.n_transitions / (args.batch_size * (args.traj_len - 1))
                            / args.keep_quantile))
    print(f"  Targeting {args.n_transitions:,} transitions; oversample by 1/{args.keep_quantile}"
          f" → {n_batches:,} batches of {args.batch_size}")

    all_obs, all_actions, all_next_obs = [], [], []
    all_disagree = []  # per-condition disagreement (for audit)

    pbar = tqdm(range(n_batches), desc=f"Ens-filter {args.env}")
    for _ in pbar:
        # GTA seed: pick batch_size sub-trajectories, partial-noise + amplified return
        bidx = np.random.choice(len(subs), size=args.batch_size, replace=True)
        x1_raw = subs[bidx]
        ret_batch = rets[bidx]
        x1_norm = nrm.normalize_batch(x1_raw, obs_dim, action_dim)
        x1_t = torch.from_numpy(x1_norm.astype(np.float32)).to(device)

        s0 = x1_raw[:, 0, :obs_dim]
        norm_s0 = nrm.obs.normalize(s0).astype(np.float32)
        amp_ret = args.gta_return_alpha * ret_batch
        norm_ret = ((amp_ret - nrm.return_mean) / nrm.return_std).astype(np.float32)
        cond = torch.from_numpy(np.concatenate([norm_s0, norm_ret[:, None]], axis=1)).to(device)

        # Each ensemble member generates a trajectory
        member_trajs = []
        for g in gens:
            with torch.no_grad():
                x_gen = g.cfm.heun_sample_partial(
                    x1_t, cond, noise_ratio=args.gta_noise_ratio,
                    nfe=args.nfe, cfg_scale=args.cfg_scale,
                )
            member_trajs.append(x_gen)                # (B, T, D)

        # Stack: (K, B, T, D)
        stacked = torch.stack(member_trajs, dim=0)

        # Disagreement: pairwise MSE on the OBS portion, per condition
        # shape (K, B, T, obs_dim)
        obs_only = stacked[..., :obs_dim]
        # Variance across K members, mean over T and obs_dim
        var_kbT = obs_only.var(dim=0, unbiased=False)        # (B, T, obs_dim)
        disagree = var_kbT.mean(dim=(1, 2))                  # (B,)
        all_disagree.append(disagree.cpu().numpy())

        # Compute keep threshold on THIS batch and keep below median
        # (final filter uses global quantile but per-batch is a decent approx
        # at large batch sizes; we re-filter globally at the end)
        # First, denormalize the member-0 trajectory and store it
        gen_oa = nrm.denormalize_batch(
            member_trajs[0].detach().cpu().numpy(), obs_dim, action_dim)
        obs_b = gen_oa[..., :obs_dim]
        actions_b = np.clip(gen_oa[..., obs_dim:obs_dim + action_dim], -1.0, 1.0)

        # Stash per-condition rows tagged with their disagreement
        dis_np = disagree.cpu().numpy()
        for b in range(args.batch_size):
            for t in range(args.traj_len - 1):
                all_obs.append((dis_np[b], obs_b[b, t]))
                all_actions.append(actions_b[b, t])
                all_next_obs.append(obs_b[b, t + 1])

        if len(all_obs) >= args.n_transitions / args.keep_quantile:
            break

    # Global filter by disagreement quantile
    print(f"\nGlobal filter: keeping bottom {args.keep_quantile:.0%} by disagreement...")
    obs_np = np.array([x[1] for x in all_obs], dtype=np.float32)
    dis_per_row = np.array([x[0] for x in all_obs], dtype=np.float32)
    actions_np = np.array(all_actions, dtype=np.float32)
    next_obs_np = np.array(all_next_obs, dtype=np.float32)

    thresh = np.quantile(dis_per_row, args.keep_quantile)
    keep = dis_per_row <= thresh
    print(f"  threshold={thresh:.4f}  kept {keep.sum():,}/{len(keep):,} ({keep.mean()*100:.1f}%)")

    # Truncate to target
    obs_np = obs_np[keep][:args.n_transitions]
    actions_np = actions_np[keep][:args.n_transitions]
    next_obs_np = next_obs_np[keep][:args.n_transitions]

    # Compute rewards via analytic RewardComputer + analytic terminals
    from reward_computer import RewardComputer
    from generate_synthetic_data import compute_terminals
    rc = RewardComputer.make(args.env, device=device)
    obs_t = torch.from_numpy(obs_np).to(device)
    act_t = torch.from_numpy(actions_np).to(device)
    nobs_t = torch.from_numpy(next_obs_np).to(device)
    with torch.no_grad():
        rewards = rc.compute(obs_t, act_t, nobs_t).cpu().numpy()
    terminals = compute_terminals(obs_np, next_obs_np, args.env)
    # Treat ALL window boundaries as truncations (drop self-loop terminals)
    # No-op here since we didn't tag window boundaries; analytic terminals only.

    # Save
    env_out_dir = os.path.join(args.output_dir, args.env)
    os.makedirs(env_out_dir, exist_ok=True)
    out_path = os.path.join(env_out_dir, "synthetic_transitions.npz")
    np.savez_compressed(out_path,
        observations=obs_np, actions=actions_np, rewards=rewards,
        next_observations=next_obs_np, terminals=terminals,
    )
    print(f"\nSaved {len(obs_np):,} transitions → {out_path}")
    print(f"  reward mean={rewards.mean():.3f} std={rewards.std():.3f}")
    print(f"  terminal rate: {terminals.mean()*100:.3f}%")

    # Save audit
    audit_path = os.path.join(env_out_dir, "disagreement_stats.npz")
    np.savez_compressed(audit_path,
        per_row_disagree=dis_per_row[keep][:args.n_transitions],
        threshold=thresh,
        keep_quantile=args.keep_quantile,
    )
    print(f"Saved audit → {audit_path}")


if __name__ == "__main__":
    main()
