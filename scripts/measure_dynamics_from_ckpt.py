"""
Measure the kinematic-dynamics residual of a diffusion checkpoint directly,
by generating a few hundred trajectories and checking Δpos ≈ dt·velocity
WITHIN each generated trajectory. Lightweight — no 1M-transition npz needed.

Usage:
    python scripts/measure_dynamics_from_ckpt.py \
        --ckpt checkpoints/hopper-medium-replay-v2/diffusion_kinval_w001/offline_final.pt \
        --env hopper-medium-replay-v2 --n_traj 256
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import torch

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (_root, os.path.join(_root, "diffusion")):
    if p not in sys.path:
        sys.path.insert(0, p)

from diffusion.generate import TrajectoryGenerator, GenerationConfig

# pos dims, velocity offset, dt — must match diffusion/train.py KINEMATICS
KIN = {
    "hopper":      (5, 6, 0.008),
    "walker2d":    (8, 9, 0.008),
    "halfcheetah": (8, 9, 0.05),
}


def real_scale_and_resid(env, npos, voff, dt, data_dir="./data"):
    d = np.load(os.path.join(data_dir, f"{env}.npz"), allow_pickle=True)
    obs = d["observations"].astype(np.float64)
    term = d.get("terminals", np.zeros(len(obs), bool)).astype(bool)
    tout = d.get("timeouts",  np.zeros(len(obs), bool)).astype(bool)
    valid = ~(term | tout)[:-1]
    o, no = obs[:-1][valid], obs[1:][valid]
    scale = np.abs(no[:, :npos] - o[:, :npos]).mean(0) + 1e-8
    resid = (np.abs((no[:, :npos] - o[:, :npos]) - dt * o[:, voff:voff+npos]) / scale).mean()
    sub = None
    # a sane target return = p90 of 100-step sub-traj returns
    rew = d["rewards"].astype(np.float64); done = term | tout
    ends = np.where(done)[0] + 1; starts = np.concatenate([[0], ends[:-1]])
    subs = [rew[s:s+100].sum() for s, e in zip(starts, ends) for s in [s] if e - s >= 100]
    target = float(np.percentile(subs, 90)) if subs else float(rew.sum() / max(len(ends), 1))
    return scale, resid, obs, target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--env", required=True)
    ap.add_argument("--n_traj", type=int, default=256)
    ap.add_argument("--nfe", type=int, default=20)
    a = ap.parse_args()

    base = a.env.split("-")[0]
    if base not in KIN:
        raise SystemExit(f"{base} not in KIN (only hopper/walker2d/halfcheetah)")
    npos, voff, dt = KIN[base]
    scale, r_resid, real_obs, target = real_scale_and_resid(a.env, npos, voff, dt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = TrajectoryGenerator.from_checkpoint(a.ckpt, device)

    init = real_obs[np.random.choice(len(real_obs), a.n_traj, replace=False)].astype(np.float32)
    out = gen.generate(n_trajectories=a.n_traj, initial_states=init,
                       target_return=target, gen_cfg=GenerationConfig(nfe=a.nfe))
    obs = np.asarray(out["observations"], dtype=np.float64)        # (B,T,obs_dim)

    dpos = obs[:, 1:, :npos] - obs[:, :-1, :npos]
    vel  = obs[:, :-1, voff:voff+npos]
    s_resid = (np.abs(dpos - dt * vel) / scale).mean()
    obs_std_ratio = (obs.reshape(-1, obs.shape[-1]).std(0) / (real_obs.std(0) + 1e-8)).mean()

    print(f"\n  ckpt: {a.ckpt}")
    print(f"  env={a.env}  n_traj={a.n_traj}  target_return={target:.1f}")
    print(f"  REAL dyn residual : {r_resid:.3f}")
    print(f"  SYN  dyn residual : {s_resid:.3f}   ratio = {s_resid/r_resid:.1f}x   "
          f"(was 22.0x for the un-fixed model)")
    print(f"  obs std ratio     : {obs_std_ratio:.2f}")


if __name__ == "__main__":
    np.random.seed(0)
    main()
