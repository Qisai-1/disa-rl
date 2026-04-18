# DiSA-RL — Diffusion Model Training

Trajectory flow matching model for the DiSA-RL framework.
Generates high-quality (obs, action, reward) trajectories from D4RL datasets
for offline RL augmentation and online sample efficiency.

---

## File overview

| File | Purpose |
|------|---------|
| `config.py` | All hyperparameters in one place |
| `data.py` | D4RL loading, normalisation, sliding-window datasets |
| `model.py` | TrajectoryDiT — RoPE + multi-modal heads |
| `flow_matching.py` | CFM loss + Heun sampler + CFG |
| `ewc.py` | Elastic Weight Consolidation for online fine-tuning |
| `train.py` | Offline pre-training + online fine-tuning loops |
| `generate.py` | Generation interface + εroll estimator |

---

## 1. Environment setup

```bash
# Create a clean conda environment (Python 3.10 recommended)
conda create -n disa python=3.10 -y
conda activate disa

# PyTorch — pick the right CUDA version for your GPU
# CUDA 12.1 (RTX 3090, 4090, A100):
pip install torch==2.2.0 torchvision --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8 (older GPUs):
pip install torch==2.2.0 torchvision --index-url https://download.pytorch.org/whl/cu118

# CPU only (slow, for testing):
pip install torch==2.2.0 torchvision --index-url https://download.pytorch.org/whl/cpu

# Other dependencies
pip install numpy einops wandb tqdm
```

---

## 2. Dataset setup

### Option A — D4RL (recommended, requires MuJoCo)

```bash
# Install MuJoCo
pip install mujoco
# Install d4rl
pip install git+https://github.com/Farama-Foundation/d4rl.git

# Test the install
python -c "import d4rl, gym; env = gym.make('halfcheetah-medium-v2'); print('D4RL OK')"
```

Leave `data_path = None` in `config.py` and set `dataset_name` to one of:
- `"halfcheetah-medium-v2"`
- `"hopper-medium-v2"`
- `"walker2d-medium-v2"`
- `"ant-medium-v2"`

### Option B — Pre-saved .npz (no MuJoCo needed)

If you already have a `.npz` with keys `observations`, `actions`, `rewards`,
`terminals`, set in `config.py`:

```python
data = DataConfig(
    data_path = "/path/to/your/dataset.npz",
    ...
)
```

---

## 3. WandB setup

```bash
wandb login
# Paste your API key from https://wandb.ai/authorize
```

Or set `wandb_entity = ""` and `WANDB_MODE=offline` to skip it:
```bash
WANDB_MODE=offline python train.py
```

---

## 4. Run offline pre-training

```bash
python train.py
```

This runs `train_offline()` with the default config at the bottom of
`train.py`.  Edit the `Config(...)` block there to change hyperparameters.

**Key settings to adjust for your GPU:**

| GPU VRAM | Recommended batch_size | depth | hidden_size |
|----------|----------------------|-------|-------------|
| 8 GB     | 64                   | 6     | 384         |
| 16 GB    | 128                  | 8     | 512         |
| 24 GB+   | 256                  | 8     | 512         |
| 40 GB+   | 512                  | 12    | 768         |

Change them in `train.py`:
```python
model = ModelConfig(
    hidden_size = 512,   # ← reduce to 384 for 8GB GPU
    depth       = 8,     # ← reduce to 6 for 8GB GPU
    ...
)
training = TrainingConfig(
    batch_size = 256,    # ← reduce for smaller GPU
    ...
)
```

**Expected training time:**
- 300,000 steps at batch_size=256 on an RTX 3090: ~8 hours
- 300,000 steps at batch_size=64 on an RTX 3090: ~24 hours

**Outputs** (saved to `./checkpoints/halfcheetah/`):
```
offline_step0010000.pt   ← checkpoint every 10k steps
offline_step0020000.pt
...
offline_final.pt         ← final model weights + normaliser
ewc_state.pt             ← Fisher information (for online fine-tuning)
```

---

## 5. Monitor training

Open WandB and watch these metrics:

| Metric | Healthy range | What it means |
|--------|--------------|---------------|
| `loss/total` | decreasing to ~0.3-0.8 | overall training progress |
| `loss/obs` | decreasing | obs reconstruction quality |
| `loss/action` | decreasing | action reconstruction quality |
| `loss/temporal` | decreasing | temporal smoothness |
| `val/gen_obs_std` | ~1.0 | normalised obs std (should stay ~1) |
| `val/gen_action_range` | ~1-3 | action magnitude (>5 = mode collapse) |
| `val/gen_obs_delta_mean` | low | trajectory smoothness |

---

## 6. Test generation after training

```bash
python generate.py
```

This loads `./checkpoints/halfcheetah/offline_final.pt` and generates
16 test trajectories, printing obs/action/reward statistics.

---

## 7. Verify your data pipeline (no GPU needed)

```bash
python data.py
```

Runs a synthetic data test — all normaliser and dataset tests should pass
without needing d4rl or a GPU.

---

## 8. Multi-GPU training (optional)

For multi-GPU, wrap the training loop with `torchrun`:

```bash
torchrun --nproc_per_node=4 train.py
```

You will need to add `DistributedDataParallel` wrapping — open an issue
or ask if you need this set up.

---

## Quick-start one-liner

```bash
conda create -n disa python=3.10 -y && conda activate disa && \
pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cu121 && \
pip install numpy einops wandb tqdm && \
WANDB_MODE=offline python train.py
```
