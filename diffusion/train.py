"""
Training script for TrajectoryDiT.

Two training modes
------------------
1. Offline pre-training (train_offline)
   Standard flow matching on the D4RL dataset.
   At the end, computes and saves the EWC Fisher information.

2. Online fine-tuning (finetune_online)
   Fine-tunes the model on newly collected real environment data,
   with EWC regularisation to prevent forgetting the offline distribution.
   Supports optional replay mixing (offline samples in every batch).
"""

from __future__ import annotations
import os
import copy
import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader, ConcatDataset
from torch.cuda.amp import autocast
from tqdm import tqdm
from typing import Optional

from config import Config, ModelConfig, TrainingConfig, DataConfig, LossConfig
from data import build_datasets, DataNormalizer, TrajectoryDataset
from model import TrajectoryDiT
from flow_matching import ConditionalFlowMatching
from ewc import EWC


# ──────────────────────────────────────────────────────────────────────────────
# EMA
# ──────────────────────────────────────────────────────────────────────────────

class EMA:
    """Exponential Moving Average of model weights for stable generation."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.decay  = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for s, m in zip(self.shadow.parameters(), model.parameters()):
            s.data.mul_(self.decay).add_(m.data, alpha=1.0 - self.decay)

    def state_dict(self):  return self.shadow.state_dict()
    def load_state_dict(self, sd): self.shadow.load_state_dict(sd)


# ──────────────────────────────────────────────────────────────────────────────
# LR schedule
# ──────────────────────────────────────────────────────────────────────────────

def cosine_with_warmup(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.05, 0.5 * (1.0 + np.cos(np.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ──────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    model:      TrajectoryDiT,
    ema:        EMA,
    optimizer:  torch.optim.Optimizer,
    scheduler,
    normalizer: DataNormalizer,
    step:       int,
    path:       str,
    extra:      dict = None,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "step":               step,
        "env_name":           extra.get("env_name", "") if extra else "",
        "model_config":       model.config_dict(),
        "model_state_dict":   model.state_dict(),
        "ema_state_dict":     ema.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "normalizer":         normalizer.as_dict(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    print(f"[step {step:>7d}]  Saved → {path}")


def load_checkpoint(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    # Handle older checkpoints that don't have mlp_dropout key
    model_cfg = dict(ckpt["model_config"])
    model_cfg.setdefault("mlp_dropout", 0.0)
    model = TrajectoryDiT(**model_cfg).to(device)
    # Strip _orig_mod. prefix added by torch.compile
    model_sd = ckpt["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in model_sd):
        model_sd = {k.replace("_orig_mod.", ""): v for k, v in model_sd.items()}
    model.load_state_dict(model_sd)
    ema   = EMA(model)
    ema_sd = ckpt["ema_state_dict"]
    if any(k.startswith("_orig_mod.") for k in ema_sd):
        ema_sd = {k.replace("_orig_mod.", ""): v for k, v in ema_sd.items()}
    ema.load_state_dict(ema_sd)
    normalizer = DataNormalizer.from_dict(ckpt["normalizer"])
    return model, ema, normalizer, ckpt


# ──────────────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(
    cfm:        ConditionalFlowMatching,
    val_loader: DataLoader,
    device:     torch.device,
) -> dict:
    """
    Compute validation loss (exact) and quick generation statistics.
    Uses Euler with nfe=10 for speed during training.
    """
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

    # Generation quality check — use Heun to match inference quality
    sample_batch = next(iter(val_loader))
    cond_sample  = sample_batch["condition"][:16].to(device)
    gen = cfm.heun_sample(16, cond_sample, nfe=20)   # (16, T, D)

    obs_dim    = cfm.model.obs_dim
    action_dim = cfm.model.action_dim

    # Gen obs std should be ~1.0 in normalised space (data is z-scored)
    val_metrics["val/gen_obs_std"]      = gen[:, :, :obs_dim].std().item()
    # Gen action range should be within [-3, 3] in normalised space
    val_metrics["val/gen_action_range"] = gen[:, :, obs_dim:obs_dim+action_dim].abs().max().item()
    # No reward dim — check obs temporal smoothness instead
    # Temporal smoothness of generated observations (lower = smoother trajectories)
    delta_gen = (gen[:, 1:, :obs_dim] - gen[:, :-1, :obs_dim]).norm(dim=-1)
    val_metrics["val/gen_obs_delta_mean"] = delta_gen.mean().item()
    # NaN check in generated samples
    val_metrics["val/gen_nan_count"] = float(torch.isnan(gen).sum().item())

    cfm.model.train()
    return val_metrics


# ──────────────────────────────────────────────────────────────────────────────
# Offline training
# ──────────────────────────────────────────────────────────────────────────────

def train_offline(
    cfg:          Config,
    resume_from:  Optional[str] = None,
) -> str:
    """
    Phase 0: Offline pre-training on D4RL dataset.

    Returns path to the final checkpoint (which includes the EWC Fisher).
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(cfg.device)

    # ── Data ──────────────────────────────────────────────────────────────
    print("\n─── Loading dataset ───")
    train_ds, val_ds, normalizer = build_datasets(
        dataset_name      = cfg.data.dataset_name,
        data_path         = cfg.data.data_path,
        trajectory_length = cfg.data.trajectory_length,
        stride            = cfg.data.stride,
        val_fraction      = cfg.data.val_fraction,
        obs_dim           = cfg.model.obs_dim,
        action_dim        = cfg.model.action_dim,
        use_return_cond   = cfg.model.use_return_cond,
        seed              = cfg.seed,
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg.training.batch_size,
        shuffle=True, num_workers=cfg.training.num_workers,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.training.batch_size,
        shuffle=False, num_workers=2, pin_memory=True,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    print("\n─── Building model ───")
    model = TrajectoryDiT(
        obs_dim           = cfg.model.obs_dim,
        action_dim        = cfg.model.action_dim,
        trajectory_length = cfg.model.trajectory_length,
        hidden_size       = cfg.model.hidden_size,
        depth             = cfg.model.depth,
        num_heads         = cfg.model.num_heads,
        mlp_ratio         = cfg.model.mlp_ratio,
        rope_theta        = cfg.model.rope_theta,
        max_seq_len       = cfg.model.max_seq_len,
        use_return_cond   = cfg.model.use_return_cond,
        cfg_dropout_prob  = cfg.model.cfg_dropout_prob,
        mlp_dropout       = cfg.model.mlp_dropout,
    ).to(device)

    try:
        model = torch.compile(model)
        print("torch.compile ✓")
    except Exception:
        pass

    ema = EMA(model, decay=cfg.training.ema_decay)
    cfm = ConditionalFlowMatching(model, device, loss_cfg=cfg.loss)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = cfg.training.lr,
        weight_decay = cfg.training.weight_decay,
        betas        = (0.9, 0.999),
    )
    scheduler = cosine_with_warmup(optimizer, cfg.training.warmup_steps, cfg.training.num_steps)

    start_step = 0
    if resume_from:
        _, ema, _, ckpt = load_checkpoint(resume_from, device)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_step = ckpt["step"]
        print(f"Resumed from step {start_step}")

    # ── WandB ─────────────────────────────────────────────────────────────
    wandb.init(
        project = cfg.wandb_project,
        entity  = cfg.wandb_entity or None,
        name    = f"{cfg.exp_name}_offline",
        config  = {
            "model":    cfg.model.__dict__,
            "loss":     cfg.loss.__dict__,
            "training": cfg.training.__dict__,
            "data":     cfg.data.__dict__,
        },
    )

    # ── Early stopping state ──────────────────────────────────────────────
    best_val_loss    = float("inf")
    patience_counter = 0

    # ── Training loop ──────────────────────────────────────────────────────
    print(f"\n─── Offline training: {cfg.training.num_steps:,} steps ───")
    print(f"    Early stopping: patience={cfg.training.patience} val checks")
    train_iter = iter(train_loader)
    pbar = tqdm(total=cfg.training.num_steps, initial=start_step, desc="Offline train")

    for step in range(start_step, cfg.training.num_steps):
        model.train()

        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        x1   = batch["trajectory"].to(device)
        cond = batch["condition"].to(device)

        optimizer.zero_grad(set_to_none=True)
        # Pure float32 — autocast disabled for stability with torch.compile
        loss, metrics = cfm.loss(x1, cond)

        # Early NaN detection — stop immediately if loss diverges
        if not torch.isfinite(loss):
            print(f"\n[step {step}] NaN/Inf loss detected! loss={loss.item()}")
            print("Stopping training — check data and learning rate.")
            break

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
        optimizer.step()
        scheduler.step()
        ema.update(model)

        if step % cfg.training.log_every == 0:
            metrics["train/grad_norm"] = grad_norm.item()
            metrics["train/lr"]        = scheduler.get_last_lr()[0]
            wandb.log(metrics, step=step)
            pbar.set_postfix(loss=f"{metrics['loss/total']:.4f}")

        if step % cfg.training.val_every == 0 and step > 0:
            val_metrics = validate(cfm, val_loader, device)
            wandb.log(val_metrics, step=step)
            val_loss = val_metrics["val/loss/total"]
            print(f"\n[{step:>7d}]  val_loss={val_loss:.4f}  "
                  f"obs_std={val_metrics['val/gen_obs_std']:.3f}")

            # Early stopping
            if val_loss < best_val_loss - cfg.training.min_delta:
                best_val_loss    = val_loss
                patience_counter = 0
                # Save best model
                save_checkpoint(model, ema, optimizer, scheduler, normalizer,
                               step, os.path.join(cfg.output_dir, "best.pt"))
            else:
                patience_counter += 1
                print(f"    No improvement ({patience_counter}/{cfg.training.patience})")
                if patience_counter >= cfg.training.patience:
                    print(f"\nEarly stopping at step {step} "
                          f"(best val_loss={best_val_loss:.4f})")
                    break

        if step % cfg.training.save_every == 0 and step > 0:
            save_checkpoint(model, ema, optimizer, scheduler, normalizer, step,
                           os.path.join(cfg.output_dir, f"offline_step{step:07d}.pt"),
                           extra={"env_name": cfg.data.dataset_name})
        pbar.update(1)

    pbar.close()

    # ── Compute and save EWC Fisher ────────────────────────────────────────
    print("\n─── Computing EWC Fisher information ───")
    ewc = EWC(
        model       = model,
        cfm         = cfm,
        data_loader = train_loader,
        device      = device,
        n_batches   = 100,
        lambda_ewc  = cfg.loss.lambda_ewc,
    )

    final_path = os.path.join(cfg.output_dir, "offline_final.pt")
    ewc_path   = os.path.join(cfg.output_dir, "ewc_state.pt")
    best_path  = os.path.join(cfg.output_dir, "best.pt")

    # Use best checkpoint weights if early stopping saved one
    # EMA weights from the best validation step are higher quality
    if os.path.exists(best_path):
        import shutil
        shutil.copy2(best_path, final_path)
        print(f"offline_final.pt ← best.pt  (val_loss={best_val_loss:.4f})")
    else:
        save_checkpoint(model, ema, optimizer, scheduler, normalizer,
                       cfg.training.num_steps, final_path,
                       extra={"env_name": cfg.data.dataset_name})
        print(f"offline_final.pt ← last step weights")
    ewc.save(ewc_path)
    wandb.finish()

    print(f"\nOffline training complete.  Final ckpt: {final_path}")
    return final_path


# ──────────────────────────────────────────────────────────────────────────────
# Online fine-tuning
# ──────────────────────────────────────────────────────────────────────────────

def finetune_online(
    offline_ckpt_path: str,
    new_trajs_dataset: TrajectoryDataset,
    cfg:               Config,
    ewc_path:          Optional[str] = None,
    offline_dataset:   Optional[TrajectoryDataset] = None,   # for replay mixing
    replay_fraction:   float = 0.3,    # fraction of batch from offline replay
    n_steps:           int   = 1000,
    lr:                float = 1e-5,
) -> None:
    """
    Online fine-tuning of the diffusion model on newly collected real data.

    Parameters
    ----------
    offline_ckpt_path : path to the offline-trained checkpoint
    new_trajs_dataset : dataset of newly collected real trajectories
    cfg               : experiment config (used for device, output_dir, etc.)
    ewc_path          : path to EWC Fisher state (saved during offline training)
    offline_dataset   : if provided, mix offline samples to prevent forgetting
    replay_fraction   : fraction of each batch drawn from offline_dataset
    n_steps           : number of fine-tuning gradient steps
    lr                : learning rate for fine-tuning (lower than pretraining)
    """
    device = torch.device(cfg.device)

    model, ema, normalizer, _ = load_checkpoint(offline_ckpt_path, device)
    cfm = ConditionalFlowMatching(model, device, loss_cfg=cfg.loss)

    # Load EWC if available
    ewc = EWC.load(ewc_path, device) if (ewc_path and os.path.exists(ewc_path)) else None
    if ewc is None:
        print("Warning: no EWC state found — fine-tuning without forgetting protection.")

    # Build data loader (optionally mixed with offline replay)
    if offline_dataset is not None and replay_fraction > 0:
        combined = ConcatDataset([new_trajs_dataset, offline_dataset])
        loader   = DataLoader(combined, batch_size=cfg.training.batch_size,
                              shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
        print(f"Fine-tuning with {replay_fraction:.0%} offline replay.")
    else:
        loader = DataLoader(new_trajs_dataset, batch_size=min(64, len(new_trajs_dataset)),
                           shuffle=True, num_workers=2, pin_memory=True, drop_last=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    model.train()
    for step in tqdm(range(n_steps), desc="Online fine-tune"):
        try:
            batch = next(loader_iter)
        except (NameError, StopIteration):
            loader_iter = iter(loader)
            batch = next(loader_iter)

        x1   = batch["trajectory"].to(device)
        cond = batch["condition"].to(device)

        optimizer.zero_grad(set_to_none=True)
        fm_loss, metrics = cfm.loss(x1, cond)

        # EWC penalty prevents forgetting the offline distribution
        total_loss = fm_loss
        if ewc is not None:
            ewc_loss   = ewc.penalty(model)
            total_loss = fm_loss + ewc_loss
            metrics["finetune/ewc_loss"] = ewc_loss.item()

        metrics["finetune/fm_loss"]    = fm_loss.item()
        metrics["finetune/total_loss"] = total_loss.item()

        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        ema.update(model)

        if step % 50 == 0:
            wandb.log(metrics, step=step)

    # Save fine-tuned checkpoint
    ft_path = os.path.join(cfg.output_dir, "online_finetuned.pt")
    save_checkpoint(model, ema, optimizer,
                   torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0),
                   normalizer, n_steps, ft_path)

    # Update EWC reference to accept the fine-tuned weights
    if ewc is not None:
        ewc.update_reference(model)
        ewc.save(os.path.join(cfg.output_dir, "ewc_state_updated.pt"))

    print(f"Fine-tuning complete → {ft_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Environment registry
# Each entry: dataset_name -> (obs_dim, action_dim, data_path)
# ──────────────────────────────────────────────────────────────────────────────

ENV_REGISTRY = {
    # HalfCheetah
    "halfcheetah-medium-v2":        (17, 6, "./data/halfcheetah-medium-v2.npz"),
    "halfcheetah-medium-replay-v2": (17, 6, "./data/halfcheetah-medium-replay-v2.npz"),
    "halfcheetah-expert-v2":        (17, 6, "./data/halfcheetah-expert-v2.npz"),
    # Hopper
    "hopper-medium-v2":             (11, 3, "./data/hopper-medium-v2.npz"),
    "hopper-medium-replay-v2":      (11, 3, "./data/hopper-medium-replay-v2.npz"),
    "hopper-expert-v2":             (11, 3, "./data/hopper-expert-v2.npz"),
    # Walker2d
    "walker2d-medium-v2":           (17, 6, "./data/walker2d-medium-v2.npz"),
    "walker2d-medium-replay-v2":    (17, 6, "./data/walker2d-medium-replay-v2.npz"),
    "walker2d-expert-v2":           (17, 6, "./data/walker2d-expert-v2.npz"),
    # Ant
    "ant-medium-v2":                (111, 8, "./data/ant-medium-v2.npz"),
    "ant-medium-replay-v2":         (111, 8, "./data/ant-medium-replay-v2.npz"),
}


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train DiSA-RL diffusion model")
    parser.add_argument(
        "--env", type=str, default="halfcheetah-medium-v2",
        choices=list(ENV_REGISTRY.keys()),
        help="D4RL dataset name.  Each env gets its own checkpoint dir and WandB run.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=128,
        help="Batch size (default 128 — safe for 2 parallel runs on a 4090).",
    )
    parser.add_argument(
        "--num_steps", type=int, default=300_000,
        help="Total gradient steps.",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume from.",
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Override learning rate (useful when resuming diverged training).",
    )
    parser.add_argument(
        "--wandb_entity", type=str, default="",
        help="WandB username/team.",
    )
    parser.add_argument(
        "--patience", type=int, default=None,
        help="Early stopping patience (overrides config default of 10).",
    )
    parser.add_argument(
        "--min_delta", type=float, default=None,
        help="Early stopping min improvement (overrides config default of 1e-4).",
    )
    args = parser.parse_args()

    if args.env not in ENV_REGISTRY:
        raise ValueError(f"Unknown env '{args.env}'.  Choose from: {list(ENV_REGISTRY.keys())}")

    obs_dim, action_dim, data_path = ENV_REGISTRY[args.env]

    # Derive a short env tag for naming  (e.g. "halfcheetah-medium-v2" -> "halfcheetah_medium")
    env_tag = args.env.replace("-v2", "").replace("-", "_")

    cfg = Config(
        model = ModelConfig(
            obs_dim           = obs_dim,
            action_dim        = action_dim,
            trajectory_length = 100,
            hidden_size       = 512,
            depth             = 8,
            num_heads         = 8,
            cfg_dropout_prob  = 0.10,
            use_return_cond   = True,
            mlp_dropout       = 0.1,
        ),
        loss = LossConfig(
            lambda_obs      = 1.0,
            lambda_action   = 1.0,
            lambda_temporal = 0.1,
            lambda_ewc      = 500.0,
        ),
        training = TrainingConfig(
            batch_size   = args.batch_size,
            patience     = args.patience  if args.patience  is not None else 10,
            min_delta    = args.min_delta if args.min_delta is not None else 1e-4,
            lr           = args.lr if args.lr is not None else 1e-4,
            weight_decay = 1e-4,
            num_steps    = args.num_steps,
            grad_clip    = 1.0,
            ema_decay    = 0.9999,
            warmup_steps = 5_000,
            save_every   = 10_000,
            log_every    = 100,
            val_every    = 2_000,
            num_workers  = 4,
        ),
        data = DataConfig(
            dataset_name      = args.env,
            data_path         = data_path,
            trajectory_length = 100,
            stride            = 50,
            val_fraction      = 0.05,
        ),
        device        = "cuda",
        seed          = 42,
        wandb_project = "disa-rl",
        wandb_entity  = args.wandb_entity,
        # Each env gets a unique run name on WandB
        exp_name      = f"{env_tag}_bs{args.batch_size}",
        # Each env gets its own checkpoint folder
        output_dir    = f"./checkpoints/{args.env}/diffusion",
    )

    print(f"\n{'='*55}")
    print(f"  Environment : {args.env}")
    print(f"  obs_dim     : {obs_dim}   action_dim: {action_dim}")
    print(f"  batch_size  : {args.batch_size}")
    print(f"  output_dir  : ./checkpoints/{args.env}/diffusion/")
    print(f"  WandB run   : {cfg.exp_name}")
    print(f"{'='*55}\n")

    train_offline(cfg, resume_from=args.resume)