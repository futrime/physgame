#!/usr/bin/env bash

set -e

EVALS=(
    "physgame"
)

source scripts/prepare_env.sh

for eval in $EVALS; do
    file="physgame/eval/${eval}_openai.py"

    echo "Running $file"

    python $file \
        --output-base-dir ./runs/eval/ \
        $@ || true
done
