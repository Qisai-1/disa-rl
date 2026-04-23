#!/usr/bin/env bash
# reorganize.sh — DiSA-RL full project restructure
# Run from repo root: bash reorganize.sh

set -e
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info() { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC}  $*"; }

echo "======================================================="
echo "  DiSA-RL Directory Reorganization"
echo "======================================================="
ROOT=$(pwd)

# ── 1. Create directory structure ─────────────────────────────────────────
echo && info "Creating directories..."
mkdir -p diffusion iql online_rl scripts logs
mkdir -p results/tables results/plots results/logs

for env in halfcheetah-medium-v2 hopper-medium-v2 walker2d-medium-v2 ant-medium-v2 \
           halfcheetah-medium-replay-v2 hopper-medium-replay-v2 walker2d-medium-replay-v2; do
    mkdir -p "checkpoints/$env/diffusion"
    mkdir -p "checkpoints/$env/iql/augmented"
    mkdir -p "checkpoints/$env/iql/offline_only"
    mkdir -p "checkpoints/$env/iql/ablations"
    mkdir -p "checkpoints/$env/online"
done
info "Done."

# ── 2. Move stale diffusion checkpoints ───────────────────────────────────
echo && info "Reorganising diffusion checkpoints..."
for env in halfcheetah-medium-v2 hopper-medium-v2 walker2d-medium-v2 ant-medium-v2; do
    for f in "checkpoints/$env"/offline_*.pt "checkpoints/$env"/ewc_state*.pt; do
        [ -f "$f" ] && mv "$f" "checkpoints/$env/diffusion/" && \
            info "  Moved $(basename $f) → checkpoints/$env/diffusion/"
    done
done

# ── 3. Sync diffusion source files ────────────────────────────────────────
echo && info "Syncing diffusion/ source files..."
for f in config.py data.py model.py flow_matching.py ewc.py generate.py train.py; do
    [ -f "$ROOT/$f" ] && cp "$ROOT/$f" "$ROOT/diffusion/$f" && info "  Synced $f"
done
touch diffusion/__init__.py

# ── 4. Sync iql source files ──────────────────────────────────────────────
echo && info "Syncing iql/ source files..."
for f in networks.py agent.py buffer.py evaluator.py train_iql.py; do
    [ -f "$ROOT/$f" ] && cp "$ROOT/$f" "$ROOT/iql/$f" && info "  Synced $f"
done
touch iql/__init__.py

# ── 5. Move stray root-level files into correct locations ────────────────────
echo && info "Moving stray files to correct locations..."

# online_rl/ files that ended up at root
for f in sac.py online_buffer.py async_generator.py train_online.py; do
    if [ -f "$ROOT/$f" ] && [ ! -f "$ROOT/online_rl/$f" ]; then
        mv "$ROOT/$f" "$ROOT/online_rl/$f"
        info "  Moved $f → online_rl/"
    elif [ -f "$ROOT/$f" ] && [ -f "$ROOT/online_rl/$f" ]; then
        rm "$ROOT/$f"
        info "  Removed duplicate root/$f"
    fi
done

# scripts that ended up at root
for f in run_experiments.sh check_health.sh collect_results.sh sync_wandb.sh; do
    if [ -f "$ROOT/$f" ]; then
        mv "$ROOT/$f" "$ROOT/scripts/$f"
        chmod +x "$ROOT/scripts/$f"
        info "  Moved $f → scripts/"
    fi
done

# Verify online_rl/ contents
echo && info "Checking online_rl/ source files..."
for f in sac.py online_buffer.py async_generator.py train_online.py; do
    [ -f "$ROOT/online_rl/$f" ] && info "  Found: $f" || warn "  Missing: $f"
done
touch online_rl/__init__.py

# ── 6. Remove stale root-level duplicates ─────────────────────────────────
echo && info "Removing root-level duplicates..."
for f in config.py data.py model.py flow_matching.py ewc.py generate.py train.py \
         networks.py agent.py buffer.py evaluator.py train_iql.py; do
    if [ -f "$ROOT/$f" ]; then
        subdir="diffusion"; [[ "$f" == networks.py || "$f" == agent.py || "$f" == buffer.py || "$f" == evaluator.py || "$f" == train_iql.py ]] && subdir="iql"
        if [ -f "$ROOT/$subdir/$f" ]; then
            rm "$ROOT/$f" && info "  Removed root/$f (lives in $subdir/)"
        fi
    fi
done

# ── 7. Patch imports in diffusion/ ────────────────────────────────────────
echo && info "Patching imports in diffusion/..."
python3 - <<'PYEOF'
import os
path_fix = (
    "import sys, os as _os\n"
    "_d = _os.path.dirname(_os.path.abspath(__file__))\n"
    "if _d not in sys.path: sys.path.insert(0, _d)\n"
    "if _os.path.dirname(_d) not in sys.path: sys.path.insert(0, _os.path.dirname(_d))\n"
)
for fname in ["train.py","generate.py","ewc.py","flow_matching.py"]:
    fpath = os.path.join("diffusion", fname)
    if not os.path.exists(fpath): continue
    with open(fpath) as f: content = f.read()
    if "_d = _os.path.dirname" in content: continue
    tag = "from __future__ import annotations\n"
    content = content.replace(tag, tag + path_fix) if tag in content else path_fix + content
    with open(fpath, "w") as f: f.write(content)
    print(f"  Patched: diffusion/{fname}")

# Fix output_dir
fpath = "diffusion/train.py"
if os.path.exists(fpath):
    with open(fpath) as f: c = f.read()
    new = c.replace('output_dir    = f"./checkpoints/{args.env}"',
                    'output_dir    = f"./checkpoints/{args.env}/diffusion"')
    if new != c:
        with open(fpath, "w") as f: f.write(new)
        print("  Fixed output_dir in diffusion/train.py")
PYEOF

# ── 8. Clean empty WandB runs ─────────────────────────────────────────────
echo && info "Cleaning empty WandB runs..."
python3 - <<'PYEOF'
import os, glob, shutil
wd = "wandb"
if not os.path.exists(wd): import sys; sys.exit(0)
cleaned = 0
for d in glob.glob(f"{wd}/offline-run-*") + glob.glob(f"{wd}/run-*"):
    if not os.path.isdir(d): continue
    wf = glob.glob(f"{d}/*.wandb")
    if not wf: continue
    extra = [f for f in os.listdir(f"{d}/files") if f != "requirements.txt"] \
            if os.path.exists(f"{d}/files") else []
    if os.path.getsize(wf[0]) < 50_000 and not extra:
        shutil.rmtree(d); cleaned += 1
        print(f"  Removed: {os.path.basename(d)}")
print(f"  Cleaned {cleaned} empty runs.")
PYEOF

# ── 9. Convert leftover .hdf5 files ──────────────────────────────────────
for env in ant-medium-v2; do
    [ -f "data/${env}.hdf5" ] && [ ! -f "data/${env}.npz" ] && \
        { echo && info "Converting data/${env}.hdf5..."; python download_data.py --datasets "$env"; }
done

# ── 10. Write .gitignore ──────────────────────────────────────────────────
echo && info "Writing .gitignore..."
cat > .gitignore << 'EOF'
data/
checkpoints/
wandb/
results/
logs/
__pycache__/
*.pyc
*.pyo
*.egg-info/
.eggs/
*.so
.env
*.env
.DS_Store
Thumbs.db
.ipynb_checkpoints/
*.ipynb
*.log
tea_debug.log
*.tmp
*.bak
EOF

# ── 11. Write all scripts ─────────────────────────────────────────────────
echo && info "Writing scripts/..."

# ─ run_experiments.sh ─
cat > scripts/run_experiments.sh << 'SCRIPT'
#!/usr/bin/env bash
# Usage: bash scripts/run_experiments.sh [machine1|machine2] [diffusion|synthetic|iql|online|ablations|status|all]
set -e; cd "$(dirname "$0")/.."
MACHINE=${1:-machine1}; PHASE=${2:-all}
G='\033[0;32m'; Y='\033[1;33m'; R='\033[0;31m'; N='\033[0m'
info()  { echo -e "${G}[INFO]${N}  $*"; }
warn()  { echo -e "${Y}[WARN]${N}  $*"; }
error() { echo -e "${R}[ERROR]${N} $*"; exit 1; }
[[ "$MACHINE" == "machine1" ]] && ENVS=("halfcheetah-medium-v2" "hopper-medium-v2") \
|| [[ "$MACHINE" == "machine2" ]] && ENVS=("walker2d-medium-v2" "ant-medium-v2") \
|| error "Unknown machine: $MACHINE"
SEEDS=(0 1 2 3 4); mkdir -p logs
wait_jobs() { local f=0; for p in "$@"; do wait "$p" || ((f++)); done; [[ $f -gt 0 ]] && warn "$f failed" || info "All done."; }

phase_status() {
    info "━━━ Status: $MACHINE ━━━"
    for env in "${ENVS[@]}"; do
        echo "  [$env]"
        [[ -f "checkpoints/$env/diffusion/offline_final.pt" ]] && echo "    ✓ Diffusion" || echo "    ✗ Diffusion"
        [[ -f "data/synthetic/$env/synthetic_transitions.npz" ]] && echo "    ✓ Synthetic" || echo "    ✗ Synthetic"
        local i=0 o=0
        for s in "${SEEDS[@]}"; do
            [[ -f "checkpoints/$env/iql/augmented/seed_${s}/final.pt" ]] && ((i++))
            [[ -f "checkpoints/$env/online/seed_${s}/final.pt" ]] && ((o++))
        done
        echo "    IQL augmented: $i/${#SEEDS[@]}  |  Online: $o/${#SEEDS[@]}"
    done
}

phase_diffusion() {
    info "━━━ Phase 0: Diffusion Training ━━━"; local pids=()
    for i in "${!ENVS[@]}"; do
        local env="${ENVS[$i]}"
        [[ -f "checkpoints/$env/diffusion/offline_final.pt" ]] && { info "Skip $env"; continue; }
        [[ $i -gt 0 ]] && sleep 120
        WANDB_MODE=offline python diffusion/train.py --env "$env" --batch_size 256 --lr 1e-4 --num_steps 300000 >> "logs/diffusion_${env}.log" 2>&1 &
        pids+=($!); info "$env → PID $! | logs/diffusion_${env}.log"
    done
    [[ ${#pids[@]} -gt 0 ]] && wait_jobs "${pids[@]}"
}

phase_synthetic() {
    info "━━━ Phase 1: Synthetic Data ━━━"
    for env in "${ENVS[@]}"; do
        [[ -f "data/synthetic/$env/synthetic_transitions.npz" ]] && { info "Skip $env"; continue; }
        [[ -f "checkpoints/$env/diffusion/offline_final.pt" ]] || error "No diffusion ckpt for $env"
        info "Generating: $env"
        python generate_synthetic_data.py --env "$env" --n_transitions 1000000 >> "logs/synthetic_${env}.log" 2>&1
        python reward_computer.py --env "$env" --test_analytic >> "logs/synthetic_${env}.log" 2>&1
        info "$env done"
    done
}

phase_iql() {
    info "━━━ Phase 2: Offline IQL ━━━"; local pids=()
    for env in "${ENVS[@]}"; do
        [[ -f "data/synthetic/$env/synthetic_transitions.npz" ]] || error "No synthetic data for $env"
        for seed in "${SEEDS[@]}"; do
            WANDB_MODE=offline python iql/train_iql.py --env "$env" --mode augmented --seed "$seed" --num_steps 1000000 >> "logs/iql_${env}_s${seed}.log" 2>&1 &
            pids+=($!)
        done
    done
    info "Launched ${#pids[@]} IQL jobs"; wait_jobs "${pids[@]}"
}

phase_online() {
    info "━━━ Phase 3: Online RL ━━━"; local pids=()
    for env in "${ENVS[@]}"; do
        local iq="checkpoints/$env/iql/augmented/best.pt"
        local df="checkpoints/$env/diffusion/offline_final.pt"
        local sy="data/synthetic/$env/synthetic_transitions.npz"
        [[ -f "$iq" ]] || error "No IQL ckpt: $iq"; [[ -f "$sy" ]] || error "No synthetic: $sy"
        for seed in "${SEEDS[@]}"; do
            WANDB_MODE=offline python online_rl/train_online.py --env "$env" --iql_ckpt "$iq" --diffusion_ckpt "$df" --synthetic_data "$sy" --num_steps 500000 --seed "$seed" >> "logs/online_${env}_s${seed}.log" 2>&1 &
            pids+=($!)
        done
    done
    info "Launched ${#pids[@]} online jobs"; wait_jobs "${pids[@]}"
}

phase_ablations() {
    [[ "$MACHINE" == "machine1" ]] || { info "Ablations: machine1 only"; return; }
    info "━━━ Phase 4: Ablations ━━━"; local pids=()
    local env="halfcheetah-medium-v2"
    for alpha in 0.3 0.5 0.7; do for seed in 0 1 2; do
        WANDB_MODE=offline python iql/train_iql.py --env "$env" --mode augmented --alpha "$alpha" --seed "$seed" >> "logs/ablation_alpha${alpha}_s${seed}.log" 2>&1 & pids+=($!)
    done; done
    for seed in 0 1 2; do
        WANDB_MODE=offline python iql/train_iql.py --env "$env" --mode offline_only --seed "$seed" >> "logs/ablation_offline_s${seed}.log" 2>&1 & pids+=($!)
    done
    wait_jobs "${pids[@]}"
}

case "$PHASE" in
    diffusion) phase_diffusion ;;  synthetic) phase_synthetic ;;
    iql)       phase_iql ;;        online)    phase_online ;;
    ablations) phase_ablations ;;  status)    phase_status ;;
    all) phase_status; phase_diffusion; phase_synthetic; phase_iql; phase_online; phase_ablations; phase_status ;;
    *) error "Unknown phase: $PHASE" ;;
esac
SCRIPT

# ─ check_health.sh ─
cat > scripts/check_health.sh << 'SCRIPT'
#!/usr/bin/env bash
cd "$(dirname "$0")/.."
python3 - << 'PYEOF'
import torch, glob, os
G='\033[0;32m'; R='\033[0;31m'; Y='\033[1;33m'; N='\033[0m'
def check_pt(p):
    try:
        c = torch.load(p, map_location='cpu', weights_only=False)
        step = c.get('step', c.get('total_steps','?'))
        nan = 0
        for k in ['model_state_dict','ema_state_dict','actor','critic','q','v']:
            if k not in c: continue
            sd = c[k]
            if any(x.startswith('_orig_mod.') for x in sd): sd={x.replace('_orig_mod.',''):v for x,v in sd.items()}
            nan += sum(torch.isnan(v).any().item() for v in sd.values())
        return nan==0, nan, step
    except Exception as e: return False,-1,str(e)

ENVS=["halfcheetah-medium-v2","hopper-medium-v2","walker2d-medium-v2","ant-medium-v2"]
print(); print("="*65); print("  DiSA-RL Checkpoint Health"); print("="*65)
ok = True
for env in ENVS:
    print(f"\n  [{env}]")
    for sub,lbl in [(f"checkpoints/{env}/diffusion","diffusion"),(f"checkpoints/{env}/iql/augmented","iql/augmented"),(f"checkpoints/{env}/online","online")]:
        if not os.path.exists(sub): continue
        pts = glob.glob(f"{sub}/offline_final.pt")+glob.glob(f"{sub}/best.pt")+glob.glob(f"{sub}/seed_*/best.pt")
        if not pts: print(f"    {lbl}: {Y}?{N} not found"); continue
        cl,n,s = check_pt(pts[0])
        print(f"    {lbl}: {G}✓{N} clean (step {s})" if cl else f"    {lbl}: {R}✗{N} NaN in {n} tensors")
        if not cl: ok = False
print(); print("="*65); print(f"  {G+'All clean'+N if ok else R+'Issues found'+N}"); print("="*65); print()
PYEOF
SCRIPT

# ─ collect_results.sh ─
cat > scripts/collect_results.sh << 'SCRIPT'
#!/usr/bin/env bash
cd "$(dirname "$0")/.."
python3 - << 'PYEOF'
import os, json, numpy as np
ENVS=["halfcheetah-medium-v2","hopper-medium-v2","walker2d-medium-v2","ant-medium-v2"]
SHORT={"halfcheetah-medium-v2":"HalfCheetah","hopper-medium-v2":"Hopper","walker2d-medium-v2":"Walker2d","ant-medium-v2":"Ant"}
BASELINES={"BC†":[42.6,29.0,75.0,85.5],"TD3+BC†":[48.3,59.3,83.7,90.7],"IQL†":[47.4,66.3,78.3,86.7]}
def load(env,mode):
    f=f"results/tables/{env}_{mode}_scores.json"
    return json.load(open(f))["scores"] if os.path.exists(f) else None
def fmt(v): return "---" if v is None else f"{np.mean(v):.1f}±{np.std(v):.1f}"
print(); print("="*70); print("  DiSA-RL Results Table"); print("="*70)
print(f"{'Method':<22}"+"".join(f"  {SHORT[e]:>12}" for e in ENVS)+f"  {'Avg':>7}")
print("-"*70)
for n,v in BASELINES.items():
    print(f"  {n:<20}"+"".join(f"  {x:>12.1f}" for x in v)+f"  {np.mean(v):>7.1f}")
print("-"*70)
for mode,lbl in [("augmented","DiSA-RL (offline)"),("online","DiSA-RL (online)")]:
    scores=[load(e,mode) for e in ENVS]
    row=f"  {lbl:<20}"+"".join(f"  {fmt(s):>12}" for s in scores)
    avgs=[np.mean(s) for s in scores if s]
    row+=f"  {np.mean(avgs):>7.1f}" if avgs else f"  {'---':>7}"
    print(row)
print("="*70); print("  † Kostrikov et al. 2022"); print()
print("Status:")
for env in ENVS:
    d="✓" if os.path.exists(f"checkpoints/{env}/diffusion/offline_final.pt") else "✗"
    s="✓" if os.path.exists(f"data/synthetic/{env}/synthetic_transitions.npz") else "✗"
    i=sum(1 for x in range(5) if os.path.exists(f"checkpoints/{env}/iql/augmented/seed_{x}/final.pt"))
    o=sum(1 for x in range(5) if os.path.exists(f"checkpoints/{env}/online/seed_{x}/final.pt"))
    print(f"  {env:<35} diff:{d} syn:{s} iql:{i}/5 online:{o}/5")
PYEOF
SCRIPT

# ─ sync_wandb.sh ─
cat > scripts/sync_wandb.sh << 'SCRIPT'
#!/usr/bin/env bash
cd "$(dirname "$0")/.."
[[ ! -d "wandb" ]] && echo "No wandb directory" && exit 0
n=$(find wandb -name "offline-run-*" -type d | wc -l)
echo "Syncing $n offline runs..."
wandb sync --sync-all wandb/
echo "Done — view at https://wandb.ai"
SCRIPT

chmod +x scripts/*.sh
info "All scripts written."

# ── 12. Summary ───────────────────────────────────────────────────────────
echo ""
echo "======================================================="
echo "  Reorganization Complete!"
echo "======================================================="
echo ""
echo "  Structure:"
echo "    diffusion/     ← model, train, generate, flow_matching"
echo "    iql/           ← networks, agent, buffer, evaluator, train_iql"
echo "    online_rl/     ← sac, online_buffer, async_generator, train_online"
echo "    scripts/       ← run_experiments, check_health, collect_results, sync_wandb"
echo "    logs/          ← one log file per training job"
echo "    checkpoints/   ← <env>/diffusion/ | iql/ | online/"
echo "    data/synthetic/← <env>/synthetic_transitions.npz"
echo ""
echo "  Usage:"
echo "    bash scripts/run_experiments.sh machine1 status"
echo "    bash scripts/run_experiments.sh machine1 all"
echo "    bash scripts/check_health.sh"
echo "    bash scripts/collect_results.sh"
echo ""
echo "  Git:"
echo "    git add -A && git commit -m 'Reorganize: full DiSA-RL structure'"
echo "    git push && (machine 2) git pull"
echo ""