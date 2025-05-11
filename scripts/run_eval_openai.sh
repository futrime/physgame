#!/usr/bin/env bash

set -e

source scripts/prepare_env.sh

for file in physgame/eval_openai/*.py; do
    # Check if the file is __init__.py
    if [ "$(basename "$file")" = "__init__.py" ]; then
        continue
    fi

    EVAL_NAME=$(basename "$file" .py)

    echo "Running $file"

    python $file \
        --output-base-dir ./runs/eval/ \
        $@
done
