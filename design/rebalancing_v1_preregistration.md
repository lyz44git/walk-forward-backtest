# Pre-registration (first design): PPO partial rebalancing on the Part 1 targets

Superseded as the headline analysis by `rebalancing_v2_plan.md`; kept unchanged in substance as the record of the first design.

Written 2026-11-14, after building the target portfolios and checking that full rebalancing reproduces the
project's `reports/ls_weekly.csv` (max abs error 1e-16), and BEFORE any test-period number of any new rule
(fixed rate, band, RL) was computed. The full-rebalancing test-period numbers are already public in Part 1
results; nothing below was chosen by looking at them. Later changes go to `design/rebalancing_v1_deviations.md`.

## Question
Can a PPO policy that chooses, each week, how far to trade toward the model's target long-short portfolio
improve net-of-cost performance over (a) full weekly rebalancing, (b) the best fixed partial-adjustment rate
chosen on the training span, (c) a per-asset no-trade band tuned on the training span?

## Data, targets, accounting (unchanged from Part 1)
- Targets: qmb walk-forward predictions with the project's main settings (expanding window, refit every 4 weeks,
  purge + 1-week embargo); long top 20% / short bottom 20%, equal weight, 100% per leg.
- Signals: `mom_12_1` and `ridge` on both datasets; `lgbm` too if its cached targets build within the compute
  budget (decided before any test number; it does not depend on results).
- Timing: ETFs enter at next open, industries at next close; weights drift between rebalances.
- Costs: 5 bps per unit of traded notional + 50 bps/yr borrow on short notional (`qmb.engine.net_returns`).

## Splits (over the project's OOS weeks)
| Dataset | Train (tuning + RL training) | Test (frozen evaluation) |
|---|---|---|
| ETFs | signals 2011-04-21 .. 2018-12 (OOS weeks 0-401) | first signal on/after 2019-01-01 .. 2026-10-23 (weeks 402-809, 408 weeks) |
| Industries | 1975-07-11 .. 2000-12 (weeks 0-1329) | first signal on/after 2001-01-01 .. 2026-09-18 (weeks 1330-2671, 1342 weeks) |

- The last training week is dropped from training/tuning (its holding period ends after the first test signal).
- Training/tuning window: from the first week at which every state feature is defined (week 53) to split-1.
- Every rule starts the test span from the same state: holdings = full target of the last training week,
  drifted. No rule pays the initial entry cost. With eta=1 the test-span returns equal the project's.

## MDP
- Action: scalar partial-adjustment rate eta_t in [0,1]; h_t = d_t + eta_t (W_t - d_t) on assets present this
  week (absent assets are closed), d_t = drifted holdings. eta=1 is the original strategy. At eta=0, the strategy only closes assets that leave the current universe.
- State (9 numbers, all known at signal date s_t):
  1. required turnover |W_t - h_{t-1}|_1 / 2 (h_{t-1} set at entry_{t-1} <= s_t; the drift after s_t is not used)
  2. gross exposure |h_{t-1}|_1 - 2
  3. signal gap: sum((W_t - h_{t-1}) z_t) / sum(W_t z_t), z_t = cross-sectional z-score of the prediction
  4. target change |W_t - W_{t-1}|_1
  5. cross-sectional std of raw 21-day returns at s_t
  6-7. std of the full-rebalance gross return over the last 4 / 26 completed holding weeks (exit <= s_t)
  8-9. mean rank IC over the last 13 / 52 completed holding weeks
  Features 4-9 are z-scored with training-span mean/std.
- Reward (percent): 100 [ (net_t - net_t^full) - (gamma/2)(net_t^2 - net_t^full^2) ], gamma = 2.
  net_t^full is full rebalancing's net return, which does not depend on the action; subtracting it only removes
  common market noise. So the policy maximises mean-variance utility of its own net return.

## PPO (implemented from scratch in PyTorch; no tuning on the test span)
- Actor: MLP 9-64-64 (tanh) -> softplus+1 -> Beta(alpha, beta). Critic: separate MLP 9-64-64-1.
  Test action = Beta mean (deterministic).
- Episodes: 52 weeks, start uniform in the training window. 64 episodes per update (split across workers),
  250 updates, discount 0.95, GAE lambda 0.95, clip 0.2, 8 epochs x 4 minibatches, Adam lr 3e-4,
  value coef 0.5, entropy coef 0.001, grad-norm clip 0.5, advantages normalised over the global batch.
- Distributed: torch.distributed (gloo), launched with torchrun; each worker collects its own episodes with its
  own RNG stream, gradients are all-reduced (averaged) every minibatch. Main results: 8 worker processes on one
  machine. Seeds 0-4, final policy after the last update (no checkpoint selection).

## Baselines (tuned on the training window only, criterion = training net Sharpe)
- (a) Full rebalancing (eta = 1).
- (b) Fixed eta from {0.05, 0.10, ..., 1.00}.
- (c) Per-asset no-trade band: h_i = W_i + clip(d_i - W_i, -b, b), b = m / k for m in {0, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2}
  (k = names per leg).
- An earlier draft proposed an ex-post test-span fixed-eta reference. It is omitted to avoid any parameter selection on the test period.

## Metrics (test span)
Net Sharpe (52 weeks) with stationary block-bootstrap 95% CI (`qmb.stats.stationary_bootstrap_sharpe`, block 8,
2000 draws, seed 0), annual return, turnover/yr, max drawdown, mean eta. Sharpe difference RL - baseline with a
paired stationary bootstrap (same resampled weeks for both series; block 8, 2000 draws, seed 0).
RL is reported per seed and as the seed-average policy (weekly net return averaged over the 5 seeds).

## Decision rule (fixed now)
For each case (dataset x signal) and baseline, RL "improves" only if (i) all 5 seeds have a higher test net
Sharpe than the baseline and (ii) the paired 95% CI of the Sharpe difference for the seed-average policy is above
zero. Otherwise: "no evidence of improvement". No multiple-testing correction is applied; the number of
comparisons is reported.

## Execution amendment before any new test evaluation

The existing full-rebalance check in the opening paragraph predates the execution amendment and is an accounting parity check against the published source series. The new pytest checks and all test-policy metrics follow this amendment.

- Keep all original cost, split, architecture, optimization, baseline and decision settings. The French LightGBM build uses the original walk-forward engine for at most 40 minutes, with the same settings as the source. Include all six cases if this finishes and reproduces the source series. Otherwise use the five existing cases and log the build status.
- Use eight logical rollout partitions in any process count. Rollout start weeks, independent uniform inputs for exact inverse-CDF Beta sampling, and per-partition epoch permutations are derived from the seed and update. A process computes its assigned partitions, and averages gradients over the same global minibatches through Gloo when distributed. Use float64 for the actor/critic and the optimizer; normalize advantages over the global batch. The topology and sampling implementation are fixed here so the matched single- and eight-process runs can be directly compared.
- Time the complete 250-update training job for ETF ridge, seed 0, on a single machine under each topology with one Torch CPU thread per process. Use the matched eight-process policy as the corresponding main-run policy. Check maximum absolute differences in parameters and weekly actions (tolerance 1e-6), weekly net returns (tolerance 1e-8), and test Sharpe, turnover/year and drawdown (tolerance 1e-6). Report all differences and timing even if the check fails. Compare the same training configuration under each topology.
- Fixed-eta and band candidates use ascending grid order as the tie-breaker. Band size uses the number of names per leg for that week, with the source rule for rounding. Annual return uses arithmetic annualization, and turnover is annual traded notional relative to capital. A seed-average row averages weekly returns, turnover and eta over all five seeds before applying summary statistics.
- Preserve signal-date availability and the original train-only scaling. The final observation of an episode may bootstrap from the first skipped week, which precedes the first test signal. Both training and tuning exclude the skipped week's return. At the test boundary "last training week" for initialization means the calendar week immediately before test; all policies inherit its full-rebalance holding at entry and drift it to the first test execution as a common initial condition, with no initial opening charge.
