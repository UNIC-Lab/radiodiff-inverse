#!/bin/bash
set -euo pipefail

# Sensor-rectangle sampling + unconditional mode
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"

TASK_CONFIG="configs/inpainting_config_sensor_uncond.yaml"
OUTPUT_DIR="${1:-results/sensor_uncond}"
GPU_ID="${GPU_ID:-0}"

python3 sample_condition_uncond.py \
    --save_dir "${OUTPUT_DIR}" \
    --task_config "${TASK_CONFIG}" \
    --version v1 \
    --gpu "${GPU_ID}" \
    --num_images 50 \
    --mask_ratios 0.5 0.6 0.7 0.8 0.85 0.9 0.95 \
    --noise_levels 0.01 0.03 0.05 0.07 0.09
