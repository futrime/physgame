#!/usr/bin/env bash

set -e

PRESET=$1
shift

echo "PRESET: ${PRESET}"

# Check if the preset file exists
if [ ! -f "physgame/datasets/build_${PRESET}.py" ]; then
    echo "Error: Preset '${PRESET}' not found at physgame/datasets/build_${PRESET}.py"
    exit 1
fi

source scripts/prepare_env.sh

python physgame/datasets/build_${PRESET}.py \
    --output-base-dir ./runs/datasets/ \
    $@
