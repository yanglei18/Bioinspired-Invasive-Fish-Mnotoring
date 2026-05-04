#!/usr/bin/env bash

set -euo pipefail

CONFIG="${1:-coarse_classifier/configs/default.yaml}"
N_GPUS="${2:-1}"

echo "Config: ${CONFIG}"
echo "GPUs: ${N_GPUS}"

if (( N_GPUS > 1 )); then
    command -v torchrun >/dev/null 2>&1 || { echo "torchrun not found"; exit 1; }
    torchrun --nproc_per_node="${N_GPUS}" \
        coarse_classifier/train.py \
        --config "${CONFIG}"
else
    python3 coarse_classifier/train.py --config "${CONFIG}"
fi
