#!/usr/bin/env bash

set -e

source scripts/prepare_env.sh

N_GPUS=$(nvidia-smi --list-gpus | wc -l)

for file in physgame/eval/*.py; do
    # Check if the file is __init__.py
    if [ "$(basename "$file")" = "__init__.py" ]; then
        continue
    fi

    EVAL_NAME=$(basename "$file" .py)

    echo "Running $file"

    torchrun \
        --nproc_per_node $N_GPUS \
        $file \
        --output-base-dir ./runs/eval/ \
        $@
done
