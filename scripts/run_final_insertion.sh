#!/bin/sh
set -eu

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <seed: 0|1|2> <phase: generate|metrics|all>" >&2
    exit 2
fi

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
exec python "$repo_root/ConStruct/analysis/run_final_constraint_suite.py" \
    --modality insertion \
    --seed "$1" \
    --phase "$2" \
    --samples "${FINAL_SAMPLES:-10000}" \
    --initial-generation-batch-size "${FINAL_BATCH_SIZE:-10000}" \
    --device "${FINAL_DEVICE:-cuda:0}" \
    --metrics-workers "${FINAL_METRICS_WORKERS:-1}" \
    --stuck-max-search-states "${FINAL_STUCK_MAX_SEARCH_STATES:-100000}" \
    --profile-start-index "${FINAL_PROFILE_START:-0}" \
    --profile-end-index "${FINAL_PROFILE_END:-30}"
