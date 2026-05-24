#!/usr/bin/env bash
set -euo pipefail
GPUS=${GPUS:-0}
MAX_PARALLEL=${MAX_PARALLEL:-12}
python ./scripts/run_alv_ad_asd_aggregate.py \
  --kind label \
  --gpus "${GPUS}" \
  --max-parallel "${MAX_PARALLEL}"
