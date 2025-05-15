#!/usr/bin/env bash

set -e

PERFS=(
    "sharegpt_mm"
    "physgame_eager"
    "physgame_fa2"
    "physgame"
    "sharegpt"
)

source scripts/prepare_env.sh

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

export CUDA_VISIBLE_DEVICES

for perf in "${PERFS[@]}"; do
    file="physgame/perf/${perf}_hf.py"

    echo "Running $file"

    python $file \
        --output-base-dir ./runs/perf/ \
        $@ || true
done
