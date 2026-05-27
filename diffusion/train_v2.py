"""
v2 diffusion training script.

Architecture changes vs v1 (train.py):
  - Uses TrajectoryDiTV2 (3-modal embedding, output D = obs+action+1).
  - Uses build_datasets_v2 with train_noise data augmentation.
  - Uses ConditionalFlowMatchingV2 (reward loss term + 5-component metrics).
  - Default bigger model: hidden=512 depth=8 num_heads=8 mlp_dropout=0.15.
  - ENV_REGISTRY extended with the 4 medium-expert envs (downloaded 2026-05-26).

Run:
    python diffusion/train_v2.py --env <env> \
        --hidden_size 512 --depth 8 --num_heads 8 \
        --mlp_dropout 0.15 --train_noise 0.01 \
        --batch_size 256 --lr 1e-4 --patience 200 --num_steps 300000

Checkpoints go to checkpoints/<env>/diffusion_v2/ to keep v1 and v2 separate.
"""

from __future__ import annotations
import os, argparse, sys
import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional

# Reuse v1 helpers — these are pure utilities, no v1 classes hardcoded
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path: sys.path.insert(0, _here)
if os.path.dirname(_here) not in sys.path: sys.path.insert(0, os.path.dirname(_here))

from train import EMA, cosine_with_warmup
from data import DataNormalizer
from data_v2 import build_datasets_v2
from model_v2 import TrajectoryDiTV2
from flow_matching_v2 import ConditionalFlowMatchingV2


# ──────────────────────────────────────────────────────────────────────────────
# ENV_REGISTRY — extended for medium-replay + medium-expert (added 2026-05-26)
# ──────────────────────────────────────────────────────────────────────────────

ENV_REGISTRY = {
    # medium-v2
    "halfcheetah-medium-v2":        (17,  6,  "./data/halfcheetah-medium-v2.npz"),
    "hopper-medium-v2":             (11,  3,  "./data/hopper-medium-v2.npz"),
    "walker2d-medium-v2":           (17,  6,  "./data/walker2d-medium-v2.npz"),
    "ant-medium-v2":                (111, 8,  "./data/ant-medium-v2.npz"),
    # medium-replay-v2
    "halfcheetah-medium-replay-v2": (17,  6,  "./data/halfcheetah-medium-replay-v2.npz"),
    "hopper-medium-replay-v2":      (11,  3,  "./data/hopper-medium-replay-v2.npz"),
    "walker2d-medium-replay-v2":    (17,  6,  "./data/walker2d-medium-replay-v2.npz"),
    "ant-medium-replay-v2":         (111, 8,  "./data/ant-medium-replay-v2.npz"),
    # medium-expert-v2 (added 2026-05-26 for 12-env top-tier table)
    "halfcheetah-medium-expert-v2": (17,  6,  "./data/halfcheetah-medium-expert-v2.npz"),
    "hopper-medium-expert-v2":      (11,  3,  "./data/hopper-medium-expert-v2.npz"),
    "walker2d-medium-expert-v2":    (17,  6,  "./data/walker2d-medium-expert-v2.npz"),
    "ant-medium-expert-v2":         (111, 8,  "./data/ant-medium-expert-v2.npz"),
}


# ──────────────────────────────────────────────────────────────────────────────
# v2 save/load
# ──────────────────────────────────────────────────────────────────────────────

def save_checkpoint_v2(model, ema, optimizer, scheduler, normalizer,
                       step, path, extra=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "step":                 step,
        "env_name":             extra.get("env_name", "") if extra else "",
        "arch_version":         "v2",
        "model_config":         model.config_dict() if not hasattr(model, "_orig_mod")
                                else model._orig_mod.config_dict(),
        "model_state_dict":     model.state_dict(),
        "ema_state_dict":       ema.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "normalizer":           normalizer.as_dict(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    print(f"[step {step:>7d}]  Saved → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Validation — extended to report reward stats
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate_v2(cfm: ConditionalFlowMatchingV2, val_loader: DataLoader,
                device: torch.device) -> dict:
    cfm.model.eval()
    total_metrics = {}
    n = 0
    for batch in val_loader:
        x1   = batch["trajectory"].to(device)
        cond = batch["condition"].to(device)
        _, metrics = cfm.loss(x1, cond)
        for k, v in metrics.items():
            total_metrics[k] = total_metrics.get(k, 0.0) + v * x1.shape[0]
        n += x1.shape[0]
    val_metrics = {f"val/{k}": v / n for k, v in total_metrics.items()}

    # Generation quality check
    sample_batch = next(iter(val_loader))
    cond_sample  = sample_batch["condition"][:16].to(device)
    gen = cfm.heun_sample(16, cond_sample, nfe=20)   # (16, T, D_with_r)

    obs_dim    = cfm.model.obs_dim
    action_dim = cfm.model.action_dim
    oa         = obs_dim + action_dim

    val_metrics["val/gen_obs_std"]      = gen[:, :, :obs_dim].std().item()
    val_metrics["val/gen_action_range"] = gen[:, :, obs_dim:oa].abs().max().item()
    # NEW v2 metrics — reward channel stats (in normalized space ≈ N(0,1))
    val_metrics["val/gen_reward_mean"]  = gen[:, :, oa:oa+1].mean().item()
    val_metrics["val/gen_reward_std"]   = gen[:, :, oa:oa+1].std().item()
    delta_gen = (gen[:, 1:, :obs_dim] - gen[:, :-1, :obs_dim]).norm(dim=-1)
    val_metrics["val/gen_obs_delta_mean"] = delta_gen.mean().item()
    val_metrics["val/gen_nan_count"]      = float(torch.isnan(gen).sum().item())

    cfm.model.train()
    return val_metrics


# ──────────────────────────────────────────────────────────────────────────────
# Main training
# ──────────────────────────────────────────────────────────────────────────────

def train_v2_offline(args) -> str:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.env not in ENV_REGISTRY:
        raise ValueError(f"Unknown env '{args.env}'. Choose from: {list(ENV_REGISTRY.keys())}")
    obs_dim, action_dim, data_path = ENV_REGISTRY[args.env]

    # Output dir — separate from v1's diffusion/ to keep checkpoints distinct
    env_tag    = args.env.replace("-v2", "").replace("-", "_")
    output_dir = f"./checkpoints/{args.env}/diffusion_v2"
    os.makedirs(output_dir, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────
    print("\n─── Loading dataset (v2) ───")
    train_ds, val_ds, normalizer = build_datasets_v2(
        dataset_name      = args.env,
        data_path         = data_path,
        trajectory_length = args.trajectory_length,
        stride            = args.stride,
        val_fraction      = 0.05,
        obs_dim           = obs_dim,
        action_dim        = action_dim,
        use_return_cond   = True,
        seed              = args.seed,
        train_noise       = args.train_noise,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size,
                              shuffle=False, num_workers=0, pin_memory=False)

    # ── Model (v2) ────────────────────────────────────────────────────────
    print("\n─── Building model (v2) ───")
    model = TrajectoryDiTV2(
        obs_dim           = obs_dim,
        action_dim        = action_dim,
        trajectory_length = args.trajectory_length,
        hidden_size       = args.hidden_size,
        depth             = args.depth,
        num_heads         = args.num_heads,
        mlp_dropout       = args.mlp_dropout,
        cfg_dropout_prob  = 0.10,
        use_return_cond   = True,
    ).to(device)

    try:
        model = torch.compile(model)
        print("torch.compile ✓")
    except Exception as e:
        print(f"torch.compile failed: {e!s:.80s}  — continuing without compile")

    ema = EMA(model, decay=args.ema_decay)
    cfm = ConditionalFlowMatchingV2(
        model, device,
        lambda_obs      = 1.0,
        lambda_action   = 1.0,
        lambda_reward   = args.lambda_reward,
        lambda_temporal = 0.1,
        cfg_scale       = 1.5,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay,
                                  betas=(0.9, 0.999))
    scheduler = cosine_with_warmup(optimizer, args.warmup_steps, args.num_steps)

    # ── WandB ─────────────────────────────────────────────────────────────
    wandb.init(
        project = args.wandb_project,
        name    = f"{env_tag}_v2_offline",
        config  = vars(args),
    )

    # ── Early-stopping state ──────────────────────────────────────────────
    best_val_loss    = float("inf")
    patience_counter = 0

    print(f"\n─── Offline training (v2): {args.num_steps:,} steps ───")
    print(f"    Early stopping: patience={args.patience} val checks")
    print(f"    arch: hidden={args.hidden_size} depth={args.depth} heads={args.num_heads} "
          f"dropout={args.mlp_dropout} | train_noise={args.train_noise} | λ_reward={args.lambda_reward}")
    train_iter = iter(train_loader)
    pbar = tqdm(total=args.num_steps, initial=0, desc="Offline train (v2)")

    for step in range(args.num_steps):
        model.train()
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        x1, cond = batch["trajectory"].to(device), batch["condition"].to(device)

        optimizer.zero_grad(set_to_none=True)
        loss, metrics = cfm.loss(x1, cond)
        if not torch.isfinite(loss):
            print(f"\n[step {step}] NaN/Inf loss! Stopping.")
            break

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        ema.update(model)

        if step % args.log_every == 0:
            metrics["train/grad_norm"] = grad_norm.item()
            metrics["train/lr"]        = scheduler.get_last_lr()[0]
            wandb.log(metrics, step=step)
            pbar.set_postfix(loss=f"{metrics['loss/total']:.4f}")

        if step % args.val_every == 0 and step > 0:
            val_metrics = validate_v2(cfm, val_loader, device)
            wandb.log(val_metrics, step=step)
            val_loss = val_metrics["val/loss/total"]
            print(f"\n[{step:>7d}]  val_loss={val_loss:.4f}  "
                  f"obs_std={val_metrics['val/gen_obs_std']:.3f}  "
                  f"reward_std={val_metrics['val/gen_reward_std']:.3f}")

            if val_loss < best_val_loss - args.min_delta:
                best_val_loss    = val_loss
                patience_counter = 0
                save_checkpoint_v2(model, ema, optimizer, scheduler, normalizer,
                                   step, os.path.join(output_dir, "best.pt"),
                                   extra={"env_name": args.env})
            else:
                patience_counter += 1
                print(f"    No improvement ({patience_counter}/{args.patience})")
                if patience_counter >= args.patience:
                    print(f"\nEarly stopping at step {step} (best val_loss={best_val_loss:.4f})")
                    break

        if step % args.save_every == 0 and step > 0:
            save_checkpoint_v2(model, ema, optimizer, scheduler, normalizer,
                               step, os.path.join(output_dir, f"offline_step{step:07d}.pt"),
                               extra={"env_name": args.env})
        pbar.update(1)

    pbar.close()
    final_path = os.path.join(output_dir, "offline_final.pt")
    save_checkpoint_v2(model, ema, optimizer, scheduler, normalizer,
                       args.num_steps - 1, final_path, extra={"env_name": args.env})
    print(f"\noffline_final.pt ← best.pt  (val_loss={best_val_loss:.4f})")
    wandb.finish()
    return final_path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=str, required=True, choices=list(ENV_REGISTRY.keys()))
    p.add_argument("--num_steps",         type=int,   default=300_000)
    p.add_argument("--batch_size",        type=int,   default=256)
    p.add_argument("--lr",                type=float, default=1e-4)
    p.add_argument("--warmup_steps",      type=int,   default=2_000)
    p.add_argument("--weight_decay",      type=float, default=1e-3)
    p.add_argument("--grad_clip",         type=float, default=1.0)
    p.add_argument("--ema_decay",         type=float, default=0.9999)
    p.add_argument("--patience",          type=int,   default=200)
    p.add_argument("--min_delta",         type=float, default=1e-4)
    p.add_argument("--num_workers",       type=int,   default=8)
    p.add_argument("--log_every",         type=int,   default=100)
    p.add_argument("--val_every",         type=int,   default=2_000)
    p.add_argument("--save_every",        type=int,   default=20_000)
    p.add_argument("--seed",              type=int,   default=42)
    p.add_argument("--trajectory_length", type=int,   default=100)
    p.add_argument("--stride",            type=int,   default=50)
    # v2-specific
    p.add_argument("--hidden_size",  type=int,   default=512)
    p.add_argument("--depth",        type=int,   default=8)
    p.add_argument("--num_heads",    type=int,   default=8)
    p.add_argument("--mlp_dropout",  type=float, default=0.15)
    p.add_argument("--train_noise",  type=float, default=0.01)
    p.add_argument("--lambda_reward",type=float, default=1.0)
    p.add_argument("--wandb_project",type=str,   default="disa-rl-v2")
    args = p.parse_args()
    train_v2_offline(args)
