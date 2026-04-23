"""
Async background diffusion generation for online RL.

Architecture:
    Main process (RL training) ←── Queue ←── Generator process (background)

The generator process runs continuously, generating synthetic trajectory
batches and pushing them into a shared multiprocessing Queue.

The RL training process calls queue.get_nowait() each step — if fresh
synthetic data is available it uses it, otherwise falls back to the
pre-generated offline synthetic buffer.

Fine-tuning protocol:
    When the RL training process collects enough real data (min_real_steps),
    it sends a FINETUNE signal via a command queue with the path to the
    latest real data. The generator process pauses generation, fine-tunes
    the diffusion model with EWC, then resumes generation with the updated
    model.

Signal protocol (command_queue):
    {"cmd": "finetune", "data_path": str}  → fine-tune on new real data
    {"cmd": "stop"}                         → terminate generator process
    {"cmd": "reload", "ckpt_path": str}     → reload model from checkpoint
"""

from __future__ import annotations
import os
import sys
import time
import numpy as np
import torch
import multiprocessing as mp
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _generator_worker(
    ckpt_path:      str,
    env_name:       str,
    target_return:  float,
    data_queue:     mp.Queue,
    cmd_queue:      mp.Queue,
    batch_size:     int   = 32,
    nfe:            int   = 20,
    cfg_scale:      float = 1.5,
    queue_maxsize:  int   = 20,
    finetune_steps: int   = 1000,
    ewc_lambda:     float = 500.0,
    device_str:     str   = "cuda",
):
    """
    Background worker process.
    Generates synthetic batches and pushes to data_queue.
    Responds to commands via cmd_queue.
    """
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    # Import inside worker to avoid issues with multiprocessing fork
    from generate import TrajectoryGenerator, GenerationConfig
    from reward_computer import RewardComputer

    print(f"[Generator] Starting on {device}  |  {env_name}")
    generator       = TrajectoryGenerator.from_checkpoint(ckpt_path, device)
    reward_computer = RewardComputer.make(env_name, device=device)
    gen_cfg         = GenerationConfig(nfe=nfe, cfg_scale=cfg_scale, clip_actions=True)

    obs_dim    = generator.model.obs_dim
    action_dim = generator.model.action_dim

    print(f"[Generator] Ready  |  "
          f"obs={obs_dim}  act={action_dim}  "
          f"target_return={target_return}")

    while True:
        # Check for commands (non-blocking)
        try:
            cmd = cmd_queue.get_nowait()
            if cmd["cmd"] == "stop":
                print("[Generator] Received stop signal. Exiting.")
                break

            elif cmd["cmd"] == "finetune":
                print(f"[Generator] Fine-tuning on {cmd['data_path']} ...")
                _finetune(generator, cmd["data_path"], device,
                         finetune_steps, ewc_lambda)
                print("[Generator] Fine-tuning complete. Resuming generation.")

            elif cmd["cmd"] == "reload":
                print(f"[Generator] Reloading from {cmd['ckpt_path']}")
                generator = TrajectoryGenerator.from_checkpoint(
                    cmd["ckpt_path"], device
                )

        except Exception:
            pass  # Queue empty — continue generating

        # Don't fill queue beyond maxsize — back-pressure
        if data_queue.qsize() >= queue_maxsize:
            time.sleep(0.1)
            continue

        # Generate a batch of trajectories
        try:
            result  = generator.generate(
                n_trajectories = batch_size,
                target_return  = target_return,
                gen_cfg        = gen_cfg,
            )
            obs_b     = result["observations"]   # (B, T, obs_dim)
            actions_b = result["actions"]        # (B, T, action_dim)
            B, T, _   = obs_b.shape

            # Flatten to transitions
            obs_flat  = obs_b.reshape(B*T, obs_dim)
            act_flat  = actions_b.reshape(B*T, action_dim)
            rew_flat  = reward_computer.compute(obs_flat, act_flat)

            # next_obs from sequence
            next_obs_b        = np.zeros_like(obs_b)
            next_obs_b[:, :-1] = obs_b[:, 1:]
            next_obs_b[:, -1]  = obs_b[:, -1]  # last step repeats
            next_obs_flat      = next_obs_b.reshape(B*T, obs_dim)

            # Done at end of each sub-trajectory
            done_flat       = np.zeros(B*T, dtype=np.float32)
            done_flat[T-1::T] = 1.0

            # Clean NaN
            obs_flat      = np.nan_to_num(obs_flat,      nan=0.0)
            act_flat      = np.clip(act_flat, -1.0, 1.0)
            rew_flat      = np.nan_to_num(rew_flat,      nan=0.0)
            next_obs_flat = np.nan_to_num(next_obs_flat, nan=0.0)

            batch = {
                "obs":      obs_flat.astype(np.float32),
                "action":   act_flat.astype(np.float32),
                "reward":   rew_flat.astype(np.float32),
                "next_obs": next_obs_flat.astype(np.float32),
                "done":     done_flat,
            }
            data_queue.put(batch, timeout=1.0)

        except Exception as e:
            print(f"[Generator] Generation error: {e}")
            time.sleep(0.5)


def _finetune(generator, data_path: str, device, steps: int, ewc_lambda: float):
    """Fine-tune the diffusion model on new real data with EWC regularization."""
    try:
        from ewc import EWC
        import torch.optim as optim

        data     = np.load(data_path, allow_pickle=True)
        obs      = torch.from_numpy(data["observations"].astype(np.float32)).to(device)
        actions  = torch.from_numpy(data["actions"].astype(np.float32)).to(device)

        # Build (obs, action) trajectory tensor for flow matching
        # Stack into (N, 1, D) single-step "trajectories"
        D       = generator.model.obs_dim + generator.model.action_dim
        x1      = torch.cat([obs, actions], dim=-1).unsqueeze(1)   # (N, 1, D)
        cond    = obs   # condition on initial obs

        # EWC: compute fisher on current model before fine-tuning
        ewc = EWC(generator.model, ewc_lambda)
        ewc.compute_fisher(x1[:512], cond[:512])

        optimizer = optim.Adam(generator.model.parameters(), lr=1e-5)
        generator.model.train()

        for step in range(steps):
            idx  = torch.randint(0, len(x1), (64,))
            loss, _ = generator.cfm.loss(x1[idx], cond[idx])

            # Add EWC penalty to prevent catastrophic forgetting
            ewc_loss = ewc.penalty(generator.model)
            total    = loss + ewc_loss

            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(generator.model.parameters(), 1.0)
            optimizer.step()

        generator.model.eval()
        print(f"[Generator] Fine-tuned {steps} steps  "
              f"loss={loss.item():.4f}  ewc={ewc_loss.item():.4f}")

    except Exception as e:
        print(f"[Generator] Fine-tune failed: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# Public interface
# ──────────────────────────────────────────────────────────────────────────────

class AsyncSyntheticGenerator:
    """
    Manages the background generator process and provides a clean interface
    to the main training loop.

    Usage:
        gen = AsyncSyntheticGenerator(ckpt_path, env_name, target_return)
        gen.start()
        ...
        # In training loop:
        queue = gen.data_queue  # pass to OnlineBuffer
        ...
        # When enough real data collected:
        gen.finetune(real_data_path)
        ...
        gen.stop()
    """

    def __init__(
        self,
        ckpt_path:      str,
        env_name:       str,
        target_return:  float,
        batch_size:     int   = 32,
        nfe:            int   = 20,
        cfg_scale:      float = 1.5,
        queue_maxsize:  int   = 20,
        finetune_steps: int   = 1000,
        ewc_lambda:     float = 500.0,
        device_str:     str   = "cuda",
    ):
        self.ckpt_path      = ckpt_path
        self.env_name       = env_name
        self.target_return  = target_return
        self.batch_size     = batch_size
        self.nfe            = nfe
        self.cfg_scale      = cfg_scale
        self.finetune_steps = finetune_steps
        self.ewc_lambda     = ewc_lambda
        self.device_str     = device_str

        # Shared queues
        self.data_queue = mp.Queue(maxsize=queue_maxsize)
        self.cmd_queue  = mp.Queue()
        self._process: Optional[mp.Process] = None

    def start(self) -> None:
        """Launch the background generator process."""
        self._process = mp.Process(
            target = _generator_worker,
            kwargs = dict(
                ckpt_path      = self.ckpt_path,
                env_name       = self.env_name,
                target_return  = self.target_return,
                data_queue     = self.data_queue,
                cmd_queue      = self.cmd_queue,
                batch_size     = self.batch_size,
                nfe            = self.nfe,
                cfg_scale      = self.cfg_scale,
                queue_maxsize  = self.data_queue.maxsize,
                finetune_steps = self.finetune_steps,
                ewc_lambda     = self.ewc_lambda,
                device_str     = self.device_str,
            ),
            daemon = True,  # dies when main process dies
        )
        self._process.start()
        print(f"AsyncSyntheticGenerator started  |  pid={self._process.pid}")

    def finetune(self, data_path: str) -> None:
        """Signal generator to fine-tune on new real data."""
        self.cmd_queue.put({"cmd": "finetune", "data_path": data_path})
        print(f"Fine-tune signal sent  |  data={data_path}")

    def reload(self, ckpt_path: str) -> None:
        """Signal generator to reload model weights."""
        self.cmd_queue.put({"cmd": "reload", "ckpt_path": ckpt_path})

    def stop(self) -> None:
        """Stop the background process cleanly."""
        if self._process and self._process.is_alive():
            self.cmd_queue.put({"cmd": "stop"})
            self._process.join(timeout=10)
            if self._process.is_alive():
                self._process.terminate()
            print("AsyncSyntheticGenerator stopped.")

    @property
    def queue_size(self) -> int:
        return self.data_queue.qsize()

    def is_alive(self) -> bool:
        return self._process is not None and self._process.is_alive()
