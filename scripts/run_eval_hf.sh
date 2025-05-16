#!/usr/bin/env bash

set -e

EVALS=(
    "physgame"
)

source scripts/prepare_env.sh

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $(($(nvidia-smi --list-gpus | wc -l)-1)))}

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

N_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

echo "Number of GPUs: $N_GPUS"

export CUDA_VISIBLE_DEVICES

for eval in "${EVALS[@]}"; do
    file="physgame/eval/${eval}_hf.py"

    echo "Running $file"

    torchrun \
        --nproc_per_node $N_GPUS \
        $file \
        --output-base-dir ./runs/eval/ \
        $@ || true
done
