#!/usr/bin/env bash

set -e

eval "$(conda shell.bash hook)"

if conda info --envs | grep -q '^physgame '; then
    conda activate physgame
else
    echo "Conda environment 'physgame' not found. Please create it first."
    exit 1
fi
