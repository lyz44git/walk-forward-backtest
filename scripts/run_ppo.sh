#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p tmp/rebalance/process_tmp tmp/rebalance/matplotlib
export TMPDIR="$PWD/tmp/rebalance/process_tmp" MPLCONFIGDIR="$PWD/tmp/rebalance/matplotlib" MPLBACKEND=Agg
PYTHON=${PYTHON:-python}
TORCHRUN=${TORCHRUN:-torchrun}
# Contain local Gloo communication on loopback. Override on other operating systems.
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-lo0}
launch=("$TORCHRUN" --nnodes=1 --node_rank=0 --master_addr=127.0.0.1
        --master_port="${MASTER_PORT:-29632}" --nproc_per_node=8)
case "$1" in
  single)
    { time "$PYTHON" -u scripts/08_train_ppo.py --cases etf:ridge --seeds 0 --tag matched_single --resume; } ;;
  matched)
    { time "${launch[@]}" scripts/08_train_ppo.py --cases etf:ridge --seeds 0 --tag main --resume; } ;;
  main)
    { time "${launch[@]}" scripts/08_train_ppo.py --tag main --resume; } ;;
  case)
    shift
    { time "${launch[@]}" scripts/08_train_ppo.py --cases "$@" --tag main --resume; } ;;
  *)
    echo "usage: $0 {single|matched|main|case [dataset:signal ...]}" >&2
    exit 2 ;;
esac
