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
