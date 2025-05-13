#!/usr/bin/env bash

set -e

source scripts/prepare_env.sh

script_name=$1
shift

python physgame/datasets/build_${script_name}.py \
    --output-base-dir ./runs/datasets/ \
    $@
