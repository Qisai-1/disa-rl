"""
Download D4RL datasets directly from the source without mujoco_py or d4rl.

D4RL hosts all datasets publicly as .hdf5 files.
This script downloads them and converts to our .npz format.

Usage:
    python download_data.py                          # downloads halfcheetah-medium-v2
    python download_data.py --datasets all           # downloads all 4 locomotion datasets
    python download_data.py --datasets hopper walker # specific datasets
"""

import argparse
import os
import urllib.request
import numpy as np

try:
    import h5py
except ImportError:
    print("Installing h5py...")
    os.system("pip install h5py")
    import h5py


# ──────────────────────────────────────────────────────────────────────────────
# Dataset URLs
# Direct download from the official D4RL host — no mujoco_py needed
# ──────────────────────────────────────────────────────────────────────────────

DATASETS = {
    # HalfCheetah
    "halfcheetah-medium-v2":        "https://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/halfcheetah_medium-v2.hdf5",
    "halfcheetah-medium-replay-v2": "https://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/halfcheetah_medium_replay-v2.hdf5",
    "halfcheetah-expert-v2":        "https://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/halfcheetah_expert-v2.hdf5",
    "halfcheetah-random-v2":        "https://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/halfcheetah_random-v2.hdf5",

    # Hopper
    "hopper-medium-v2":             "https://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/hopper_medium-v2.hdf5",
    "hopper-medium-replay-v2":      "https://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/hopper_medium_replay-v2.hdf5",
    "hopper-expert-v2":             "https://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/hopper_expert-v2.hdf5",

    # Walker2d
    "walker2d-medium-v2":           "https://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/walker2d_medium-v2.hdf5",
    "walker2d-medium-replay-v2":    "https://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/walker2d_medium_replay-v2.hdf5",
    "walker2d-expert-v2":           "https://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/walker2d_expert-v2.hdf5",

    # Ant
    "ant-medium-v2":                "https://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/ant_medium-v2.hdf5",
    "ant-medium-replay-v2":         "https://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/ant_medium_replay-v2.hdf5",
}

# Observation/action dims for each environment
ENV_DIMS = {
    "halfcheetah": (17, 6),
    "hopper":      (11, 3),
    "walker2d":    (17, 6),
    "ant":         (27, 8),
}


def get_env_dims(dataset_name: str):
    for env, dims in ENV_DIMS.items():
        if env in dataset_name:
            return dims
    raise ValueError(f"Unknown environment in dataset name: {dataset_name}")


def download_hdf5(url: str, save_path: str) -> None:
    """Download with a progress bar."""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    print(f"Downloading {os.path.basename(save_path)} ...")

    def progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(downloaded / total_size * 100, 100)
            bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
            mb  = downloaded / 1e6
            print(f"\r  [{bar}] {pct:5.1f}%  {mb:.0f} MB", end="", flush=True)

    urllib.request.urlretrieve(url, save_path, reporthook=progress)
    print(f"\n  Saved → {save_path}")


def hdf5_to_npz(hdf5_path: str, npz_path: str, dataset_name: str) -> None:
    """
    Convert D4RL .hdf5 to our flat-transitions .npz format.

    Output keys: observations, actions, rewards, terminals, timeouts
    """
    obs_dim, action_dim = get_env_dims(dataset_name)

    with h5py.File(hdf5_path, "r") as f:
        print(f"  HDF5 keys: {list(f.keys())}")

        observations = f["observations"][:]
        actions      = f["actions"][:]
        rewards      = f["rewards"][:]
        terminals    = f["terminals"][:]
        timeouts     = f.get("timeouts", np.zeros_like(terminals))[:]

    print(f"  observations : {observations.shape}  (expected obs_dim={obs_dim})")
    print(f"  actions      : {actions.shape}  (expected action_dim={action_dim})")
    print(f"  rewards      : {rewards.shape}")
    print(f"  terminals    : {terminals.sum()} episode ends")
    print(f"  timeouts     : {timeouts.sum()} timeouts")

    # Basic sanity checks
    assert observations.shape[1] == obs_dim, \
        f"obs_dim mismatch: got {observations.shape[1]}, expected {obs_dim}"
    assert actions.shape[1] == action_dim, \
        f"action_dim mismatch: got {actions.shape[1]}, expected {action_dim}"

    np.savez(
        npz_path,
        observations = observations.astype(np.float32),
        actions      = actions.astype(np.float32),
        rewards      = rewards.astype(np.float32),
        terminals    = terminals.astype(bool),
        timeouts     = timeouts.astype(bool),
    )
    size_mb = os.path.getsize(npz_path + ".npz") / 1e6
    print(f"  Converted → {npz_path}.npz  ({size_mb:.0f} MB)")


def download_dataset(name: str, data_dir: str = "./data") -> str:
    """
    Download and convert one dataset.
    Returns the path to the .npz file (without .npz extension).
    """
    if name not in DATASETS:
        raise ValueError(
            f"Unknown dataset '{name}'.\nAvailable:\n  " +
            "\n  ".join(DATASETS.keys())
        )

    url       = DATASETS[name]
    hdf5_path = os.path.join(data_dir, f"{name}.hdf5")
    npz_path  = os.path.join(data_dir, name)   # np.savez adds .npz

    # Skip download if already done
    if os.path.exists(npz_path + ".npz"):
        print(f"Already exists: {npz_path}.npz  (skipping download)")
        return npz_path + ".npz"

    if not os.path.exists(hdf5_path):
        download_hdf5(url, hdf5_path)
    else:
        print(f"HDF5 already downloaded: {hdf5_path}")

    print(f"Converting to .npz ...")
    hdf5_to_npz(hdf5_path, npz_path, name)

    # Remove the .hdf5 to save disk space (optional)
    os.remove(hdf5_path)
    print(f"Removed {hdf5_path}")

    return npz_path + ".npz"


def print_dataset_info(npz_path: str) -> None:
    """Print a summary of the saved dataset."""
    data = np.load(npz_path)
    obs  = data["observations"]
    n    = len(obs)

    terminals = data["terminals"]
    timeouts  = data["timeouts"]
    n_episodes = (terminals | timeouts).sum()

    rewards = data["rewards"]
    print(f"\n── Dataset summary: {os.path.basename(npz_path)} ──")
    print(f"  Transitions : {n:,}")
    print(f"  Episodes    : {n_episodes:,}")
    print(f"  obs shape   : {obs.shape}")
    print(f"  action shape: {data['actions'].shape}")
    print(f"  reward range: [{rewards.min():.2f}, {rewards.max():.2f}]")
    print(f"  mean ep return (approx): "
          f"{rewards.sum() / max(n_episodes, 1):.1f}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download D4RL datasets without mujoco_py"
    )
    parser.add_argument(
        "--datasets", nargs="+",
        default=["halfcheetah-medium-v2"],
        help=(
            "Dataset names to download, or 'all' for all locomotion datasets.\n"
            "Examples:\n"
            "  python download_data.py\n"
            "  python download_data.py --datasets halfcheetah-medium-v2 hopper-medium-v2\n"
            "  python download_data.py --datasets all"
        ),
    )
    parser.add_argument(
        "--data_dir", default="./data",
        help="Directory to save datasets (default: ./data)"
    )
    args = parser.parse_args()

    # Expand 'all'
    if args.datasets == ["all"]:
        datasets = list(DATASETS.keys())
    else:
        datasets = args.datasets

    print(f"Will download: {datasets}")
    print(f"Save to: {os.path.abspath(args.data_dir)}\n")

    downloaded_paths = []
    for name in datasets:
        print(f"\n{'='*55}")
        print(f"  {name}")
        print(f"{'='*55}")
        try:
            path = download_dataset(name, args.data_dir)
            print_dataset_info(path)
            downloaded_paths.append((name, path))
        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\n{'='*55}")
    print("Done. Set data_path in train.py:")
    for name, path in downloaded_paths:
        print(f'  # {name}')
        print(f'  data_path = "{path}"')
