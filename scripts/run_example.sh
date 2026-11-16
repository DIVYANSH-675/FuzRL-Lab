#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
args=(--algorithm "${ALGORITHM:-fuzppolag}" --env "${ENVIRONMENT:-Pendulum-v1}"
      --epochs "${EPOCHS:-100}" --seed "${SEED:-42}"
      --train-eps "${TRAIN_EPS:-0}" --disturb-part "${DISTURB_PART:-observation}"
      --disturb-type "${DISTURB_TYPE:-white_noise}")
if [[ -n "${GPU:-}" ]]; then args+=(--gpu "$GPU"); fi
exec "${PYTHON:-python}" "$SCRIPT_DIR/run_training.py" "${args[@]}" "$@"
