# Deviations and implementation notes

## 2026-11-14, before new test evaluation

- Preserved the handoff preregistration and recorded an execution amendment before running any new test or evaluation. It fixes comparable random streams across one and eight physical processes, exact Beta sampling through the inverse CDF, double-precision training, the timing case and tolerances, and deterministic baseline ties.
- French LightGBM is built alone, reusing the source implementation. This avoids overwriting the existing ETF LightGBM cache.
- Clarified the boundary initialization. All rules inherit the full-rebalance holdings from the calendar week before the test. That week is excluded from train rewards and baseline tuning.
- Removed the inherited ex-post test-period fixed-eta reference under the instruction to avoid test-period parameter selection. All parameter selection is confined to the training span.
- The macOS `torchrun --standalone` launcher repeatedly attempted to resolve the reverse-DNS name for 127.0.0.1. The local launcher uses static loopback rendezvous and the same Gloo backend, worker count, code and training settings.
- Before any test evaluation, the first topology benchmark showed training histories drifting late in optimization. CPU batch shapes and floating-point reduction differed by topology. This was corrected by forwarding the same partition shapes in each topology and using a fixed-order reduction of partition rows after the Gloo all-reduce. Both topologies were rerun from initialization. The rolled-out samples, PPO settings, test criteria, budgets and baselines are unchanged. The first benchmark logs and completed training artifacts are retained under `tmp/` as exploratory implementation checks.
- The fixed eight-process PPO jobs were scheduled across two local Gloo groups to finish inside the compute budget. Each group handled different cases. Checkpoints were reused only when their process count and hyperparameters matched. The benchmark finished before the second training group started.

## 2026-11-14, after evaluating frozen policies

- Some bootstrap draws for the ETF LightGBM no-trade band have zero return variance. The preregistered qmb CI and paired decision interval remain blank when any draw is undefined. Supplemental columns show the 95% quantiles of finite draws and the number of valid draws. The figure marks the supplemental paired interval separately. The decision still uses the preregistered interval. Models, baselines and grids were frozen throughout.
