#!/usr/bin/env bash

# PPO-Lagrangian 

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export ALGORITHM="${ALGORITHM:-ppolag}"

exec "${SCRIPT_DIR}/run_example.sh" "$@"

