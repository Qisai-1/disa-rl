import sys, os as _os
_d = _os.path.dirname(_os.path.abspath(__file__))
if _d not in sys.path: sys.path.insert(0, _d)
sys.path.insert(0, _os.path.join(_d, "diffusion"))
import torch, glob, os

def check_env(ckpt_dir, env_name):
    ckpts = sorted(glob.glob(f"{ckpt_dir}/offline_step*.pt"))
    final = f"{ckpt_dir}/offline_final.pt"
    if os.path.exists(final):
        ckpts.append(final)

    if not ckpts:
        print(f"\n{env_name}: NO CHECKPOINTS FOUND")
        return

    print(f"\n{env_name} ({len(ckpts)} checkpoints):")
    clean_count = 0
    for path in ckpts:
        try:
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            sd   = ckpt.get('model_state_dict', ckpt.get('ema_state_dict', {}))
            if any(k.startswith('_orig_mod.') for k in sd):
                sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
            nan_count = sum(torch.isnan(v).any().item() for v in sd.values())
            step      = ckpt.get('step', '?')
            status    = "✓ CLEAN" if nan_count == 0 else f"✗ NaN in {nan_count} tensors"
            if nan_count == 0:
                clean_count += 1
            print(f"  step {str(step):>7}  {os.path.basename(path):<35} {status}")
        except Exception as e:
            print(f"  {os.path.basename(path):<35} ERROR: {e}")

    print(f"  Summary: {clean_count}/{len(ckpts)} clean")

# Check all environments — adjust base path as needed
base = "./checkpoints"
for env in ["halfcheetah-medium-v2", "hopper-medium-v2",
            "walker2d-medium-v2", "ant-medium-v2"]:
    ckpt_dir = f"{base}/{env}/diffusion"
    if os.path.exists(ckpt_dir):
        check_env(ckpt_dir, env)
    else:
        print(f"\n{env}: directory not found ({ckpt_dir})")
