#!/usr/bin/env bash

set -e

source scripts/prepare_env.sh

script_name=$1
shift

accelerate launch physgame/train/${script_name}.py "$@"
