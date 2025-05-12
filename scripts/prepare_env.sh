#!/usr/bin/env bash

set -e

CONDA_ENV=${CONDA_ENV:-"physgame"}

echo "CONDA_ENV: $CONDA_ENV"

eval "$(conda shell.bash hook)"

if conda info --envs | grep -q "^$CONDA_ENV "; then
    conda activate $CONDA_ENV
else
    echo "Conda environment '$CONDA_ENV' not found. Please create it first."
    exit 1
fi
