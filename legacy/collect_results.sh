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
