#!/usr/bin/env bash

set -e

PRESET=$1
shift

echo "PRESET: ${PRESET}"

# Check if the preset file exists
if [ ! -f "physgame/train/${PRESET}.py" ]; then
    echo "Error: Preset '${PRESET}' not found at physgame/train/${PRESET}.py"
    exit 1
fi

source scripts/prepare_env.sh

accelerate launch physgame/train/${PRESET}.py \
    --output-base-dir runs/train \
    $@
