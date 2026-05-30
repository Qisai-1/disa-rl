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
from iql.agent_capa import CAPAAgent
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

def get_synthetic_path(env_name: str, vcdg: bool = False) -> str:
    fname = "synthetic_transitions_vcdg.npz" if vcdg else "synthetic_transitions.npz"
    return f"./data/synthetic/{env_name}/{fname}"


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
    q_h = tuple(args.q_hidden_dims) if args.q_hidden_dims else None
    # CAPA constraints: requires augmented mode + must not combine with --sa_iql.
    if args.capa:
        if args.mode != "augmented":
            raise SystemExit("ERROR: --capa requires --mode augmented (CAPA only "
                             "makes sense when synthetic data is mixed in).")
        if args.sa_iql:
            raise SystemExit("ERROR: --capa is incompatible with --sa_iql. "
                             "CAPA replaces DRC's defensive mechanisms.")

    AgentClass = CAPAAgent if args.capa else IQLAgent
    agent_kwargs = dict(
        obs_dim     = obs_dim,
        action_dim  = action_dim,
        hidden_dims = (256, 256),
        q_hidden_dims = q_h,
        v_hidden_dims = q_h,
        expectile   = args.expectile,
        expectile_real = args.expectile_real,
        expectile_syn  = args.expectile_syn,
        temperature = args.temperature,
        discount    = 0.99,
        tau         = 0.005,
        lr_q        = 3e-4,
        lr_v        = 3e-4,
        lr_pi       = 3e-4,
        bc_weight   = args.bc_weight,
        adv_normalize = args.adv_normalize,
        num_critics  = args.num_critics,
        critic_subset_size = args.critic_subset,
        sa_iql       = args.sa_iql,
        sa_clip      = tuple(args.sa_clip),
        action_noise_std = args.action_noise_std,
        pa_weight    = args.pa_weight,
        pa_min_q     = args.pa_min_q,
        device      = device,
    )
    if args.capa:
        agent_kwargs["unc_beta"] = args.unc_beta
        agent_kwargs["critic_syn_gate"] = args.capa_plus
        agent_kwargs["critic_syn_coef"] = args.critic_syn_coef
    agent = AgentClass(**agent_kwargs)

    # ── Buffers ───────────────────────────────────────────────────────────
    real_buffer = ReplayBuffer(data_path, device,
                                reward_scale=args.reward_scale,
                                reward_norm=args.reward_norm,
                                obs_norm=args.obs_norm)

    synthetic_buffer = None
    alpha = 1.0  # default: pure real

    if args.mode == "augmented":
        syn_path = args.synthetic_data or get_synthetic_path(args.env, vcdg=args.use_vcdg_data)
        if os.path.exists(syn_path):
            # Pass the CORL norm factor through reward_scale so syn rewards
            # are pre-scaled into the same magnitude as real before the
            # ±3σ OOD filter runs (else the filter rejects all syn).
            syn_reward_scale = args.reward_scale * getattr(real_buffer, "_corl_factor", 1.0)
            # Obs normalization stats only when --obs_norm is on; else syn
            # stays in raw obs space matching the real buffer.
            syn_obs_mean = real_buffer.obs_mean if args.obs_norm else None
            syn_obs_std  = real_buffer.obs_std  if args.obs_norm else None
            synthetic_buffer = SyntheticBuffer(
                syn_path, device,
                real_reward_mean  = real_buffer.reward_mean,
                real_reward_std   = real_buffer.reward_std,
                normalize_rewards = True,
                filter_sigma      = 3.0,
                return_weighting  = True,
                reward_scale      = syn_reward_scale,
                obs_mean          = syn_obs_mean,
                obs_std           = syn_obs_std,
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
        alpha_warmup     = args.alpha_warmup,
        alpha_ramp       = args.alpha_ramp,
    )

    # ── Evaluator ─────────────────────────────────────────────────────────
    eval_obs_mean = real_buffer.obs_mean if args.obs_norm else None
    eval_obs_std  = real_buffer.obs_std  if args.obs_norm else None
    evaluator = make_evaluator(args.env, device, n_episodes=10,
                                obs_mean=eval_obs_mean, obs_std=eval_obs_std)

    # ── WandB ─────────────────────────────────────────────────────────────
    env_tag  = args.env.replace("-v2", "").replace("-", "_")
    # CAPA runs go to a separate output dir so they don't collide with
    # vanilla augmented (DRC) runs at the same alpha.
    mode_dir = "capa" if args.capa else args.mode
    run_name = f"iql_{env_tag}_{mode_dir}_alpha{args.alpha}_s{args.seed}"
    output_dir = os.path.join("./checkpoints", args.env, "iql", mode_dir,
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

    # ── Resume from checkpoint ─────────────────────────────────────────────
    start_step = 1
    if args.resume:
        import glob
        ckpts = glob.glob(os.path.join(output_dir, "step_*.pt"))
        if ckpts:
            # Pick the most recently written checkpoint, not the highest step
            # number — stale checkpoints from an earlier run with a different
            # architecture may share this dir and have larger step numbers.
            latest = max(ckpts, key=os.path.getmtime)
            agent.load(latest)
            start_step = agent.total_steps + 1
            if start_step > args.num_steps:
                raise SystemExit(
                    f"  Resume checkpoint {os.path.basename(latest)} is at step "
                    f"{agent.total_steps:,} >= --num_steps {args.num_steps:,}. "
                    f"Refusing to no-op — check for stale checkpoints in {output_dir}.")
            # Keep the alpha warmup/ramp schedule aligned with the global step.
            aug_buffer._step = agent.total_steps
            print(f"  Resuming from {os.path.basename(latest)} "
                  f"→ continuing at step {start_step:,}/{args.num_steps:,}")
        else:
            print(f"  --resume set but no step_*.pt in {output_dir} — "
                  f"starting fresh from step 1.")

    # ── Training loop ──────────────────────────────────────────────────────
    best_score = -float("inf")
    pbar = tqdm(total=args.num_steps, initial=start_step - 1,
                desc=f"IQL [{args.mode}] seed={args.seed}")

    need_real_batch = (
        (args.mode == "augmented" and args.bc_weight > 0)
        or args.capa
    )

    for step in range(start_step, args.num_steps + 1):
        aug_buffer.step()                       # advances alpha warmup/ramp

        # UTD>1 (RLPD/EDAC-style): k-1 critic-only updates with FRESH batches +
        # 1 full update. Sharper Q with the ensemble; ~k× compute.
        for _ in range(max(args.utd - 1, 0)):
            b  = aug_buffer.sample(args.batch_size)
            rb = aug_buffer.sample_real(args.batch_size) if need_real_batch else None
            agent.update(b, real_batch=rb, critic_only=True)

        batch = aug_buffer.sample(args.batch_size)
        real_batch = aug_buffer.sample_real(args.batch_size) if need_real_batch else None
        metrics = agent.update(batch, real_batch=real_batch)

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
    parser.add_argument("--resume", action="store_true",
                        help="Resume from the latest step_*.pt checkpoint in the "
                             "run's output dir (restores optimizer + step state).")
    parser.add_argument("--wandb_project", type=str, default="disa-rl",
                        help="WandB project name")
    parser.add_argument("--use_vcdg_data", action="store_true",
                        help="Use synthetic_transitions_vcdg.npz (VCDG-generated)")
    parser.add_argument("--q_hidden_dims", type=int, nargs="*", default=None,
                        help="Q/V hidden dims, e.g. --q_hidden_dims 256 256 256 256")
    parser.add_argument("--adv_normalize", action="store_true", default=True,
                        help="Normalize AWR advantages (default ON)")
    parser.add_argument("--no_adv_normalize", dest="adv_normalize",
                        action="store_false", help="Disable advantage normalization")

    # ── SA-IQL (A1) ──────────────────────────────────────────────────────
    parser.add_argument("--sa_iql", action="store_true",
                        help="Enable SA-IQL: mixture expectile + density-ratio TD weighting")
    parser.add_argument("--expectile_real", type=float, default=None,
                        help="SA-IQL expectile on real transitions (default: --expectile)")
    parser.add_argument("--expectile_syn",  type=float, default=None,
                        help="SA-IQL expectile on syn transitions (default: --expectile)")
    parser.add_argument("--sa_clip", type=float, nargs=2, default=[0.5, 2.0],
                        help="SA-IQL density-ratio clip range")

    # ── CAPA (Critic-Anchored Proposal Augmentation, headline method) ────
    parser.add_argument("--capa", action="store_true",
                        help="Enable CAPA: real-only critic + syn-actor + "
                             "ensemble-uncertainty gate. Requires --mode augmented. "
                             "Incompatible with --sa_iql. See METHOD_V2_PROPOSAL.md.")
    parser.add_argument("--unc_beta", type=float, default=1.0,
                        help="CAPA uncertainty-gate strength on syn AWR rows: "
                             "gate(syn) = exp(-unc_beta * q_ensemble_std). "
                             "1.0 = standard. 0.0 = disable gate (ablation).")
    parser.add_argument("--capa_plus", action="store_true",
                        help="CAPA+: also feed gated (low-uncertainty) synthetic "
                             "transitions into the V/Q updates, not just the actor. "
                             "Trades reward-immunity for critic coverage; gate→0 on "
                             "untrusted syn ⇒ degenerates to vanilla CAPA.")
    parser.add_argument("--critic_syn_coef", type=float, default=1.0,
                        help="CAPA+ coefficient scaling the gated-syn V/Q loss terms.")

    # ── Q-ensemble ───────────────────────────────────────────────────────
    parser.add_argument("--num_critics", type=int, default=2,
                        help="Number of Q-networks. 2 = Twin-Q, >=3 = QEnsemble (REDQ-style)")
    parser.add_argument("--utd", type=int, default=1,
                        help="Update-To-Data ratio (RLPD/EDAC). >1 = utd-1 extra "
                             "critic-only updates per actor update. Sharper Q with "
                             "the ensemble; ~utd× critic compute. Try 4 with LayerNorm "
                             "+ 10-critic ensemble (we have both).")
    parser.add_argument("--critic_subset", type=int, default=2,
                        help="Random subset size for min target (REDQ trick)")

    # ── Action-noise augmentation ─────────────────────────────────────────
    parser.add_argument("--action_noise_std", type=float, default=0.0,
                        help="Gaussian noise std added to dataset actions in AWR log_prob")

    # ── Alpha warmup (fixes augmented-slower-than-baseline) ──────────────
    parser.add_argument("--alpha_warmup", type=int, default=0,
                        help="Number of pure-real (alpha=1.0) steps before any syn mixing. "
                             "Default 0 (immediate mixing). 50000 = standard warmup.")
    parser.add_argument("--alpha_ramp", type=int, default=0,
                        help="Steps to linearly ramp alpha from 1.0 down to --alpha. "
                             "Default 0 (instant). 50000 = smooth schedule.")

    # ── PARS-style reward scaling + PA loss (ICML 2025) ────────────────
    parser.add_argument("--reward_scale", type=float, default=1.0,
                        help="Multiply rewards by this factor (PARS). "
                             "Recommended: halfcheetah 5; hopper/walker2d 10; ant 5-10.")
    parser.add_argument("--reward_norm", type=str, default="none",
                        choices=["none", "corl"],
                        help="Reward normalization. 'corl' = standard IQL "
                             "D4RL locomotion recipe: rewards × 1000/"
                             "(max_ep_return-min_ep_return). Default 'none'.")
    parser.add_argument("--obs_norm", action="store_true", default=False,
                        help="Standard CORL/IQL per-dim obs normalization "
                             "((obs - mean) / std) using real-dataset stats. "
                             "Applied identically to syn and at eval.")
    parser.add_argument("--pa_weight", type=float, default=0.0,
                        help="Weight of PARS Penalty-for-Infeasible-Actions loss term. "
                             "Typical: 0.0001 (MuJoCo) — 0.01 (sparse).")
    parser.add_argument("--pa_min_q", type=float, default=0.0,
                        help="Q lower bound used as target for PA loss "
                             "(target Q on OOD actions).")

    args = parser.parse_args()
    train_iql(args)