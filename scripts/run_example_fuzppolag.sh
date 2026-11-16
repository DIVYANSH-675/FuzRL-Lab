#!/usr/bin/env bash

# Fuz-PPO-Lagrangian

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export ALGORITHM="${ALGORITHM:-fuzppolag}"

exec "${SCRIPT_DIR}/run_example.sh" "$@"

