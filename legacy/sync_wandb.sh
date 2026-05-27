#!/usr/bin/env bash
cd "$(dirname "$0")/.."
[[ ! -d "wandb" ]] && echo "No wandb directory" && exit 0
n=$(find wandb -name "offline-run-*" -type d | wc -l)
echo "Syncing $n offline runs..."
wandb sync --sync-all wandb/
echo "Done — view at https://wandb.ai"
