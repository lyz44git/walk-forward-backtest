#!/usr/bin/env bash
# Every fit of Part 4, then the summary. Ridge fits take seconds; LightGBM fits run one at a time.
set -euo pipefail
cd "$(dirname "$0")/.."
run() { python scripts/13_nonprice_features.py fit "$@"; }
python scripts/13_nonprice_features.py coverage
for ds in etf french; do
  for model in ridge lgbm; do
    for fs in P PN N; do run --dataset $ds --signal $model --set $fs; done
    if [ $model = ridge ]; then seeds=$(seq 0 19); else seeds=$(seq 0 2); fi
    for s in $seeds; do run --dataset $ds --signal $model --set PN --variant placebo --seed $s; done
  done
done
for model in ridge lgbm; do run --dataset french --signal $model --set PN --variant lagged; done
python scripts/13_nonprice_features.py summarize
