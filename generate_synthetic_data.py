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


def compute_terminals(env_base: str, next_obs: np.ndarray) -> np.ndarray:
    """Analytic MuJoCo termination on an (N, obs_dim) array of next-states.

    Mirrors the gym-v2 `done` condition exactly — termination is a known
    closed-form function of the state for locomotion (just like the reward).
    A `done=1` flag means next_obs is a genuine terminal state (the agent
    fell), which correctly CUTS the value bootstrap. Window boundaries are
    handled separately as TRUNCATIONS (their transition is dropped, not
    flagged terminal). halfcheetah never terminates; unknown envs → all 0.

    Obs layout: obs[0]=height (qpos[1] for hopper/walker, qpos[2]=torso-z for
    ant), obs[1]=torso angle (hopper/walker).
    """
    finite = np.isfinite(next_obs).all(axis=1)
    if env_base == "hopper":
        z, ang = next_obs[:, 0], next_obs[:, 1]
        bounded = (np.abs(next_obs[:, 1:]) < 100).all(axis=1)
        healthy = finite & bounded & (z > 0.7) & (np.abs(ang) < 0.2)
        return (~healthy).astype(np.float32)
    if env_base == "walker2d":
        z, ang = next_obs[:, 0], next_obs[:, 1]
        healthy = finite & (z > 0.8) & (z < 2.0) & (ang > -1.0) & (ang < 1.0)
        return (~healthy).astype(np.float32)
    if env_base == "ant":
        z = next_obs[:, 0]
        healthy = finite & (z >= 0.2) & (z <= 1.0)
        return (~healthy).astype(np.float32)
    # halfcheetah (never terminates) and any unknown env
    return np.zeros(len(next_obs), dtype=np.float32)


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
        out = agent.q(s, a)
        # Works for TwinQ (returns (q1, q2) tuple) AND QEnsemble (returns (M,B) tensor).
        if isinstance(out, tuple):
            return torch.min(out[0], out[1])
        return out.min(dim=0).values
    def v_fn(s):
        return agent.v(s)
    print(f"  VCDG/Q-guidance critic loaded from {iql_ckpt}  "
          f"(critic type: {type(agent.q).__name__})")
    return q_fn, v_fn, agent


def build_real_subtrajs(env, traj_len, obs_dim, action_dim,
                        data_dir="./data", stride=10):
    """Real (obs+action) sub-trajectories + their returns, for GTA-style
    partial-noising generation. Returns (N, T, obs_dim+action_dim) float32 and
    (N,) sub-traj returns."""
    d = np.load(f"{data_dir}/{env}.npz", allow_pickle=True)
    obs = d["observations"].astype(np.float32)
    act = d["actions"].astype(np.float32)
    rew = d["rewards"].astype(np.float32)
    term = d.get("terminals", np.zeros(len(obs), bool)).astype(bool)
    tout = d.get("timeouts",  np.zeros(len(obs), bool)).astype(bool)
    done = term | tout
    ends = np.where(done)[0] + 1
    starts = np.concatenate([[0], ends[:-1]])
    subs, rets = [], []
    for s, e in zip(starts, ends):
        for st in range(s, e - traj_len + 1, stride):
            subs.append(np.concatenate([obs[st:st+traj_len], act[st:st+traj_len]], axis=1))
            rets.append(float(rew[st:st+traj_len].sum()))
    if not subs:
        raise ValueError(f"No sub-trajectories of length {traj_len} in {env}.")
    return np.asarray(subs, dtype=np.float32), np.asarray(rets, dtype=np.float32)


def generate_synthetic_data(env, n_transitions=1_000_000, batch_size=64,
                             nfe=20, cfg_scale=1.5, output_dir="./data/synthetic",
                             device=None, force_env=None,
                             return_sampling="topk",
                             use_vcdg=False, vcdg_iql_ckpt=None,
                             vcdg_guidance_scale=0.5,
                             vcdg_td_relabel=True,
                             vcdg_q_anomaly=True,
                             use_idm=False, idm_ckpt=None,
                             integrate_velocity=False,
                             gta=False, gta_noise_ratio=0.5, gta_return_alpha=1.2,
                             gta_adaptive_alpha=False,
                             gta_alpha_min=1.0, gta_alpha_max=1.5,
                             gta_q_guidance=False, gta_q_guidance_scale=0.5,
                             gta_q_guidance_ckpt=None):
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

    # VCDG and/or GTA-Q-guided critic (loaded if either path requests it)
    q_fn = v_fn = agent_critic = None
    if use_vcdg or gta_q_guidance:
        print("\n─── Loading IQL critic for guidance ──────────────────────")
        # Prefer the explicit gta_q_guidance_ckpt if given; else fall back to vcdg path
        guide_ckpt = gta_q_guidance_ckpt if (gta_q_guidance and gta_q_guidance_ckpt) else vcdg_iql_ckpt
        q_fn, v_fn, agent_critic = _load_iql_critic(
            env, obs_dim, action_dim, device, iql_ckpt=guide_ckpt,
        )
        if use_vcdg:
            print(f"  VCDG: guidance_scale={vcdg_guidance_scale}  "
                  f"td_relabel={vcdg_td_relabel}  q_anomaly={vcdg_q_anomaly}")
        if gta_q_guidance:
            print(f"  GTA-Q-guidance: scale={gta_q_guidance_scale}  "
                  f"(applied per Heun step on (s,a) channels)\n")
            # NB: q_fn expects obs in the space the IQL agent was trained in.
            # If that ckpt used --obs_norm, you'll need to re-normalize before
            # calling q_fn — track via a follow-up (Stage-2.5+ ckpts).

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

    # GTA-style generation: pre-build real sub-trajectories to partial-noise.
    if gta:
        gta_subtrajs, gta_subrets = build_real_subtrajs(env, traj_len, obs_dim, action_dim)
        nrm = generator.normalizer
        # Precompute per-sub-traj alpha if adaptive mode is on:
        # low-return seeds get max alpha (more lift), high-return get min alpha (safer).
        # Linear in rank percentile of the full sub-traj return distribution.
        if gta_adaptive_alpha:
            ranks = gta_subrets.argsort().argsort().astype(np.float32)
            pct   = ranks / max(len(ranks) - 1, 1)                          # 0=lowest, 1=highest
            gta_alpha_per_traj = (gta_alpha_max
                                  - (gta_alpha_max - gta_alpha_min) * pct).astype(np.float32)
            print(f"  GTA mode (ADAPTIVE α): {len(gta_subtrajs):,} real sub-trajs  "
                  f"noise_ratio={gta_noise_ratio}  α∈[{gta_alpha_min},{gta_alpha_max}] "
                  f"by return-rank  (per-traj amplification + partial-noising)")
            print(f"    α stats: mean={gta_alpha_per_traj.mean():.3f}  "
                  f"min={gta_alpha_per_traj.min():.3f}  max={gta_alpha_per_traj.max():.3f}")
        else:
            gta_alpha_per_traj = None
            print(f"  GTA mode: {len(gta_subtrajs):,} real sub-trajs  "
                  f"noise_ratio={gta_noise_ratio}  return_alpha={gta_return_alpha}  "
                  f"(amplified-return + partial-noising)")

    all_obs, all_actions, all_next_obs = [], [], []

    for i in tqdm(range(n_batches), desc=f"Generating {env}"):
        # ── GTA path: partial-noise a real sub-traj, denoise w/ amplified return ──
        if gta:
            bidx   = np.random.choice(len(gta_subtrajs), size=batch_size, replace=True)
            x1_raw = gta_subtrajs[bidx]                       # (B, T, obs+act) raw
            rets   = gta_subrets[bidx]                        # (B,)
            x1_norm = nrm.normalize_batch(x1_raw, obs_dim, action_dim)  # (B,T,D)
            x1_t   = torch.from_numpy(x1_norm.astype(np.float32)).to(device)
            # cond = [normalized s0, normalized AMPLIFIED return]
            s0       = x1_raw[:, 0, :obs_dim]
            norm_s0  = nrm.obs.normalize(s0).astype(np.float32)
            if gta_alpha_per_traj is not None:
                amp_ret = gta_alpha_per_traj[bidx] * rets
            else:
                amp_ret = gta_return_alpha * rets
            norm_ret = ((amp_ret - nrm.return_mean) / nrm.return_std).astype(np.float32)
            cond     = torch.from_numpy(
                np.concatenate([norm_s0, norm_ret[:, None]], axis=1)).to(device)
            # Choose sampler: Q-guided if requested, else plain GTA partial-Heun.
            if gta_q_guidance and q_fn is not None:
                from diffusion.value_guided import q_guided_partial
                # Closure: take (obs, act) in DataNormalizer space → denormalize →
                # call q_fn (which expects raw obs/action). Returns (N,) Q scalar.
                obs_dn = nrm.obs.denormalize
                act_dn = nrm.action.denormalize if hasattr(nrm, "action") else (lambda a: a)
                def q_closure(o_norm, a_norm):
                    o_raw = torch.from_numpy(
                        obs_dn(o_norm.detach().cpu().numpy().astype(np.float32))
                    ).to(device, dtype=torch.float32) if not torch.is_tensor(obs_dn(o_norm[0:1].detach().cpu().numpy())) else None
                    # Faster: assume normalizers operate on numpy; do round-trip on GPU manually
                    return q_fn(o_norm, a_norm)  # placeholder — see note below
                # The closure above is a stub: a proper denormalize-on-GPU requires
                # the normalizer to expose torch ops. To avoid silent mis-norm,
                # we use the simpler q_fn directly when generator.normalizer.obs has
                # `.mean`, `.std` as numpy arrays (the standard case):
                _omean = torch.from_numpy(nrm.obs.mean.astype("float32")).to(device)
                _ostd  = torch.from_numpy(nrm.obs.std.astype("float32")).to(device)
                _amean = (torch.from_numpy(nrm.action.mean.astype("float32")).to(device)
                          if hasattr(nrm, "action") else None)
                _astd  = (torch.from_numpy(nrm.action.std.astype("float32")).to(device)
                          if hasattr(nrm, "action") else None)
                def q_closure(o_norm, a_norm):
                    o_raw = o_norm * _ostd + _omean
                    if _amean is not None:
                        a_raw = a_norm * _astd + _amean
                    else:
                        a_raw = a_norm
                    return q_fn(o_raw, a_raw)
                x_gen = q_guided_partial(
                    generator.cfm, x1_t, cond, q_closure,
                    obs_dim=obs_dim, action_dim=action_dim,
                    noise_ratio=gta_noise_ratio, nfe=nfe, cfg_scale=cfg_scale,
                    guidance_scale=gta_q_guidance_scale,
                )
            else:
                x_gen = generator.cfm.heun_sample_partial(
                    x1_t, cond, noise_ratio=gta_noise_ratio, nfe=nfe, cfg_scale=cfg_scale)
            gen_oa = nrm.denormalize_batch(
                x_gen.detach().cpu().numpy(), obs_dim, action_dim)         # (B,T,D)
            obs_b     = gen_oa[..., :obs_dim]
            actions_b = np.clip(gen_oa[..., obs_dim:obs_dim + action_dim], -1.0, 1.0)
            B, T, _ = obs_b.shape
            for b in range(B):
                for t in range(T - 1):
                    all_obs.append(obs_b[b, t]); all_actions.append(actions_b[b, t])
                    all_next_obs.append(obs_b[b, t + 1])
            continue

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
        # Emit transitions for t = 0 .. T-2 only. The final step (t=T-1) has no
        # real successor inside the window — its boundary is a TRUNCATION (the
        # behavior continues beyond the window), not a termination, so we drop
        # that degenerate self-loop entirely rather than fake a terminal there.
        for b in range(B):
            for t in range(T - 1):
                all_obs.append(obs_b[b, t])
                all_actions.append(actions_b[b, t])
                all_next_obs.append(obs_b[b, t + 1])

    obs      = np.stack(all_obs)[:n_transitions].astype(np.float32)
    actions  = np.stack(all_actions)[:n_transitions].astype(np.float32)
    next_obs = np.stack(all_next_obs)[:n_transitions].astype(np.float32)

    # Velocity-integration: enforce per-transition physics by overwriting the
    # next-state POSITIONS with s'_pos = s_pos + dt·s_vel (the generated next-state
    # velocities are kept). Makes Δpos = dt·vel exactly, so the (s,s') the critic
    # bootstraps on is a physical successor of s. Fixes the temporal-jitter defect
    # (see project-diffusion-dynamics-defect) without retraining. Only for envs
    # whose Euler kinematics hold (hopper/walker2d; not halfcheetah at dt=0.05).
    if integrate_velocity:
        # env_base -> (n_pos, vel_offset, dt)
        _KIN = {"hopper": (5, 6, 0.008), "walker2d": (8, 9, 0.008)}
        base = env.split("-")[0]
        if base not in _KIN:
            print(f"  integrate_velocity: '{base}' has no kinematic layout — SKIPPED")
        else:
            npos, voff, dtp = _KIN[base]
            before = float(np.abs((next_obs[:, :npos] - obs[:, :npos])
                                  - dtp * obs[:, voff:voff+npos]).mean())
            next_obs[:, :npos] = obs[:, :npos] + dtp * obs[:, voff:voff+npos]
            print(f"  integrate_velocity ON ({base}): next_obs[:{npos}] ← "
                  f"obs_pos + {dtp}·obs_vel  (pre-fix mean |Δpos−dt·v|={before:.4f} → 0)")

    # Analytic termination: done=1 iff next_obs is a genuine terminal state
    # (computed on the FINAL next_obs, after velocity-integration). Window
    # boundaries were already dropped above as truncations.
    env_base = env.split("-")[0]
    dones = compute_terminals(env_base, next_obs)
    print(f"  terminals: {int(dones.sum()):,}/{len(dones):,} "
          f"({100*dones.mean():.2f}%) analytic done=1 ({env_base})")

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
    parser.add_argument("--integrate_velocity", action="store_true",
                        help="Enforce per-transition physics: next_obs positions "
                             "← obs_pos + dt·obs_vel (hopper/walker2d). Fixes the "
                             "temporal-jitter defect at generation time.")
    # ── GTA-style generation (arXiv 2405.16907) — the proven high-return win ──
    parser.add_argument("--gta", action="store_true",
                        help="GTA-style generation: partial-noise REAL sub-trajs + "
                             "AMPLIFIED-return guidance (the technique that beats "
                             "baselines, esp. hopper-mr). Pairs with --integrate_velocity.")
    parser.add_argument("--gta_noise_ratio", type=float, default=0.5,
                        help="GTA noise ratio μ∈(0,1]: 0→keep real traj, 1→full gen.")
    parser.add_argument("--gta_adaptive_alpha", action="store_true",
                        help="ADAPTIVE α(τ): per-trajectory amplification based on the "
                             "seed sub-trajectory's return percentile. Low-return seeds get "
                             "α_max (more lift), high-return seeds get α_min (safer). "
                             "Calibrated extension of GTA's uniform amplification. "
                             "When set, overrides --gta_return_alpha.")
    parser.add_argument("--gta_alpha_min", type=float, default=1.0,
                        help="Minimum α applied to the HIGHEST-return real sub-trajectories.")
    parser.add_argument("--gta_alpha_max", type=float, default=1.5,
                        help="Maximum α applied to the LOWEST-return real sub-trajectories.")
    parser.add_argument("--gta_q_guidance", action="store_true",
                        help="NOVELTY — Q-guided GTA: at each Heun denoising step, "
                             "take a gradient of Q(s,a) wrt the trajectory and add "
                             "it to the velocity. Stronger than V-guidance because "
                             "Q sees BOTH obs and action channels. Requires --gta + "
                             "--gta_q_guidance_ckpt (or --vcdg_iql_ckpt fallback).")
    parser.add_argument("--gta_q_guidance_scale", type=float, default=0.5,
                        help="Strength of the Q-gradient guidance term.")
    parser.add_argument("--gta_q_guidance_ckpt", type=str, default=None,
                        help="Explicit IQL/CAPA ckpt for the Q-guidance critic. "
                             "Falls back to --vcdg_iql_ckpt or the auto path.")
    parser.add_argument("--gta_return_alpha", type=float, default=1.2,
                        help="GTA return amplification α>1: condition on α·return to "
                             "steer toward higher-return compositions.")
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
        integrate_velocity=args.integrate_velocity,
        gta=args.gta, gta_noise_ratio=args.gta_noise_ratio,
        gta_return_alpha=args.gta_return_alpha,
        gta_adaptive_alpha=args.gta_adaptive_alpha,
        gta_alpha_min=args.gta_alpha_min,
        gta_alpha_max=args.gta_alpha_max,
        gta_q_guidance=args.gta_q_guidance,
        gta_q_guidance_scale=args.gta_q_guidance_scale,
        gta_q_guidance_ckpt=args.gta_q_guidance_ckpt,
    )