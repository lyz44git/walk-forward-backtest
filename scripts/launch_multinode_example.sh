#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Run on each participating machine from the same experiment layout and case cache.
# Give each machine a unique NODE_RANK, and point MASTER_ADDR to rank zero.
# The total process count must divide eight.
: "${MASTER_ADDR:?Set MASTER_ADDR to the reachable hostname or IP of node rank zero}"
: "${NODE_RANK:?Set NODE_RANK to a unique rank from zero to NNODES minus one}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
mkdir -p tmp/rebalance/process_tmp
export TMPDIR="$PWD/tmp/rebalance/process_tmp"
exec "${TORCHRUN:-torchrun}" --nnodes="${NNODES:-2}" --nproc_per_node="${NPROC_PER_NODE:-4}" \
  --node_rank="$NODE_RANK" --master_addr="$MASTER_ADDR" --master_port="${MASTER_PORT:-29500}" \
  scripts/08_train_ppo.py "$@"
