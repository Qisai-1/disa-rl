#!/usr/bin/env bash
# reorganize.sh
# Restructures disa_rl into a clean layout without breaking imports.
# Safe to run multiple times (idempotent).
# Run from: ~/disa_rl/
#   bash reorganize.sh

set -e
echo "======================================================="
echo "  DiSA-RL directory reorganization"
echo "======================================================="

ROOT=$(pwd)

# ── 1. Create folder structure ─────────────────────────────────────────────
echo ""
echo "Creating folders..."
mkdir -p diffusion
mkdir -p iql
mkdir -p scripts
mkdir -p results/tables
mkdir -p results/plots
mkdir -p results/logs

# Reorganise checkpoints into diffusion/ subfolder per env
for env in halfcheetah-medium-v2 hopper-medium-v2 walker2d-medium-v2 ant-medium-v2 \
           halfcheetah-medium-replay-v2 hopper-medium-replay-v2 walker2d-medium-replay-v2; do
    if [ -d "checkpoints/$env" ]; then
        mkdir -p "checkpoints/$env/diffusion"
        mkdir -p "checkpoints/$env/iql/baseline"
        mkdir -p "checkpoints/$env/iql/augmented"
        mkdir -p "checkpoints/$env/iql/ablations"

        # Move diffusion checkpoints into diffusion/ subfolder
        for f in checkpoints/$env/offline_*.pt checkpoints/$env/ewc_state*.pt; do
            [ -f "$f" ] && mv "$f" "checkpoints/$env/diffusion/" && \
                echo "  Moved $(basename $f) → checkpoints/$env/diffusion/"
        done
    fi
done

# ── 2. Move diffusion model source files ───────────────────────────────────
echo ""
echo "Moving diffusion source files..."
DIFFUSION_FILES="config.py data.py model.py flow_matching.py ewc.py generate.py train.py"
for f in $DIFFUSION_FILES; do
    if [ -f "$ROOT/$f" ] && [ ! -f "$ROOT/diffusion/$f" ]; then
        cp "$ROOT/$f" "$ROOT/diffusion/$f"
        echo "  Copied $f → diffusion/$f"
    fi
done

# Create diffusion/__init__.py
touch diffusion/__init__.py

# ── 3. Fix imports in diffusion/ copies ────────────────────────────────────
# The diffusion/ copies need to import from each other, not the root.
# We add sys.path manipulation to each file's top so both locations work.
python3 - <<'PYEOF'
import os, re

diffusion_dir = "diffusion"
files_to_patch = ["train.py", "generate.py", "ewc.py", "flow_matching.py"]

path_fix = (
    "import sys, os as _os\n"
    "_d = _os.path.dirname(_os.path.abspath(__file__))\n"
    "if _d not in sys.path: sys.path.insert(0, _d)\n"
    "if _os.path.dirname(_d) not in sys.path: sys.path.insert(0, _os.path.dirname(_d))\n"
)

for fname in files_to_patch:
    fpath = os.path.join(diffusion_dir, fname)
    if not os.path.exists(fpath):
        continue
    with open(fpath) as f:
        content = f.read()
    if "_d = _os.path.dirname" in content:
        continue  # already patched
    # Insert after the __future__ import or at the top
    if "from __future__" in content:
        content = content.replace(
            "from __future__ import annotations\n",
            f"from __future__ import annotations\n{path_fix}"
        )
    else:
        content = path_fix + content
    with open(fpath, "w") as f:
        f.write(content)
    print(f"  Patched imports: diffusion/{fname}")
PYEOF

# ── 4. Update checkpoint paths in diffusion/train.py ──────────────────────
# The output_dir in train.py now needs to point to checkpoints/<env>/diffusion/
python3 - <<'PYEOF'
import os, re

fpath = "diffusion/train.py"

try:
    with open(fpath) as f:
        content = f.read()
    # Update output_dir to use diffusion subfolder
    new_content = content.replace(
        'output_dir    = f"./checkpoints/{args.env}"',
        'output_dir    = f"./checkpoints/{args.env}/diffusion"'
    )
    if new_content != content:
        with open(fpath, "w") as f:
            f.write(new_content)
        print("  Updated output_dir in diffusion/train.py")
except Exception as e:
    print(f"  Note: could not patch train.py: {e}")
PYEOF

# ── 5. Create __init__.py for iql if not exists ───────────────────────────
touch iql/__init__.py

# ── 6. Clean up failed WandB runs (keep only runs with actual metrics) ─────
echo ""
echo "Cleaning up wandb runs..."
python3 - <<'PYEOF'
import os, glob

wandb_dir = "wandb"
if not os.path.exists(wandb_dir):
    print("  No wandb directory found, skipping.")
    import sys; sys.exit(0)

cleaned = 0
for run_dir in glob.glob(os.path.join(wandb_dir, "offline-run-*")) + \
               glob.glob(os.path.join(wandb_dir, "run-*")):
    if not os.path.isdir(run_dir):
        continue
    # Check if this run has any actual logged data
    wandb_file = glob.glob(os.path.join(run_dir, "*.wandb"))
    files_dir  = os.path.join(run_dir, "files")
    has_data   = False
    if os.path.exists(files_dir):
        data_files = [f for f in os.listdir(files_dir)
                      if f not in ("requirements.txt",)]
        has_data = len(data_files) > 0

    if not has_data and wandb_file:
        size = os.path.getsize(wandb_file[0])
        if size < 50_000:   # < 50KB = essentially empty run (crashed early)
            import shutil
            shutil.rmtree(run_dir)
            cleaned += 1
            print(f"  Removed empty run: {os.path.basename(run_dir)}")

print(f"  Cleaned {cleaned} empty wandb runs.")
PYEOF

# ── 7. Create .gitignore ───────────────────────────────────────────────────
echo ""
echo "Writing .gitignore..."
cat > .gitignore << 'EOF'
# Data (large binary files)
data/

# Trained model checkpoints
checkpoints/

# WandB logs
wandb/

# Results (generated files)
results/

# Python
__pycache__/
*.pyc
*.pyo
*.egg-info/
.eggs/
*.so

# Environment
.env
*.env

# OS
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints/
*.ipynb

# Logs
*.log
tea_debug.log

# Temporary
*.tmp
*.bak
EOF
echo "  .gitignore created."

# ── 8. Write launch scripts ────────────────────────────────────────────────
echo ""
echo "Writing launch scripts..."

cat > scripts/train_diffusion.sh << 'EOF'
#!/usr/bin/env bash
# Train diffusion models for all environments
# Usage: bash scripts/train_diffusion.sh
# Runs two environments in parallel (adjust for your GPU count)

cd "$(dirname "$0")/.."

ENVS=("halfcheetah-medium-v2" "hopper-medium-v2" "walker2d-medium-v2" "ant-medium-v2")
PIDS=()

for env in "${ENVS[@]}"; do
    echo "Starting diffusion training: $env"
    WANDB_MODE=offline python diffusion/train.py --env "$env" --batch_size 128 &
    PIDS+=($!)
    sleep 120   # Wait 2 min between launches to let VRAM stabilize
done

echo "All training jobs started. PIDs: ${PIDS[@]}"
wait
echo "All done."
EOF

cat > scripts/train_iql_baseline.sh << 'EOF'
#!/usr/bin/env bash
# Train IQL baselines (no augmentation) across all envs and seeds
# Run this while diffusion models are training

cd "$(dirname "$0")/.."

ENVS=("halfcheetah-medium-v2" "hopper-medium-v2" "walker2d-medium-v2" "ant-medium-v2")
SEEDS=(0 1 2 3 4)

for env in "${ENVS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        echo "IQL baseline: $env  seed=$seed"
        WANDB_MODE=offline python iql/train_iql.py \
            --env "$env" \
            --mode offline_only \
            --seed "$seed" \
            --num_steps 1000000
    done
done
EOF

cat > scripts/train_iql_augmented.sh << 'EOF'
#!/usr/bin/env bash
# Train DiSA-RL augmented IQL across all envs and seeds
# Run AFTER diffusion models are trained

cd "$(dirname "$0")/.."

ENVS=("halfcheetah-medium-v2" "hopper-medium-v2" "walker2d-medium-v2" "ant-medium-v2")
SEEDS=(0 1 2 3 4)

for env in "${ENVS[@]}"; do
    CKPT="./checkpoints/$env/diffusion/offline_final.pt"
    if [ ! -f "$CKPT" ]; then
        echo "WARNING: $CKPT not found, skipping $env"
        continue
    fi
    for seed in "${SEEDS[@]}"; do
        echo "DiSA-RL augmented: $env  seed=$seed"
        WANDB_MODE=offline python iql/train_iql.py \
            --env "$env" \
            --mode augmented \
            --diffusion_ckpt "$CKPT" \
            --seed "$seed" \
            --num_steps 1000000
    done
done
EOF

cat > scripts/run_ablations.sh << 'EOF'
#!/usr/bin/env bash
# Run ablation studies on halfcheetah (representative environment)
# Ablations: fixed_alpha_0.3, fixed_alpha_0.5, fixed_alpha_0.7, offline_only

cd "$(dirname "$0")/.."

ENV="halfcheetah-medium-v2"
CKPT="./checkpoints/$ENV/diffusion/offline_final.pt"
SEEDS=(0 1 2)

# Fixed alpha ablations
for alpha in 0.3 0.5 0.7; do
    for seed in "${SEEDS[@]}"; do
        WANDB_MODE=offline python iql/train_iql.py \
            --env "$ENV" --mode ablation_fixed_alpha \
            --diffusion_ckpt "$CKPT" \
            --alpha "$alpha" --seed "$seed"
    done
done

# No augmentation ablation
for seed in "${SEEDS[@]}"; do
    WANDB_MODE=offline python iql/train_iql.py \
        --env "$ENV" --mode offline_only --seed "$seed"
done
EOF

chmod +x scripts/*.sh
echo "  scripts/ created."

# ── 9. Convert ant hdf5 if still present ──────────────────────────────────
if [ -f "data/ant-medium-v2.hdf5" ] && [ ! -f "data/ant-medium-v2.npz" ]; then
    echo ""
    echo "Converting ant-medium-v2.hdf5 → .npz ..."
    python download_data.py --datasets ant-medium-v2
fi

# ── 10. Summary ───────────────────────────────────────────────────────────
echo ""
echo "======================================================="
echo "  Reorganization complete!"
echo "======================================================="
echo ""
echo "New structure:"
echo "  diffusion/     ← diffusion model source (copy of root files)"
echo "  iql/           ← IQL offline RL"
echo "  scripts/       ← experiment launch scripts"
echo "  checkpoints/   ← <env>/diffusion/  and  <env>/iql/"
echo "  results/       ← tables/ plots/ logs/"
echo ""
echo "Root files (original) still work as-is."
echo ""
echo "Next steps:"
echo "  1. git add -A && git commit -m 'Reorganize project structure'"
echo "  2. git push"
echo "  3. Start IQL baseline: bash scripts/train_iql_baseline.sh"
