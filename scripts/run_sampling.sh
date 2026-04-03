#!/bin/bash
set -euo pipefail

# Usage:
#   ./scripts/run_sampling.sh <task_config_name_without_suffix> <gpu_id> <save_dir> [c_rate] [particle_size]
# Example:
#   ./scripts/run_sampling.sh inpainting 0 results/manual_run 0.95 5

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <task> <gpu> <save_dir> [c_rate] [particle_size]"
    exit 1
fi

TASK="$1"
GPU_ID="$2"
SAVE_DIR="$3"
C_RATE="${4:-0.95}"
PARTICLE_SIZE="${5:-5}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

python3 sample_condition.py \
    --model_config configs/model_config.yaml \
    --diffusion_config configs/diffusion_config.yaml \
    --task_config "configs/${TASK}_config.yaml" \
    --gpu "${GPU_ID}" \
    --save_dir "${SAVE_DIR}" \
    --c_rate "${C_RATE}" \
    --particle_size "${PARTICLE_SIZE}"
