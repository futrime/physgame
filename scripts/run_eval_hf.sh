#!/usr/bin/env bash

set -e

EVALS=(
    "physgame_video"
    "physgame"
)

source scripts/prepare_env.sh

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $(($(nvidia-smi --list-gpus | wc -l)-1)))}

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Get N_GPUS from CUDA_VISIBLE_DEVICES if it's set, otherwise from nvidia-smi
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    # Count the number of GPUs in CUDA_VISIBLE_DEVICES (comma-separated list)
    N_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
else
    # If CUDA_VISIBLE_DEVICES is not set, use all available GPUs
    N_GPUS=$(nvidia-smi --list-gpus | wc -l)
fi

export CUDA_VISIBLE_DEVICES

for eval in "${EVALS[@]}"; do
    file="physgame/eval/${eval}_hf.py"

    echo "Running $file"

    torchrun \
        --nproc_per_node $N_GPUS \
        $file \
        --output-base-dir ./runs/eval/ \
        $@
done
