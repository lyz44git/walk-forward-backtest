# Part 5, portfolio construction with a risk model and a turnover penalty: plan written 2026-11-19, before any optimizer return was computed

## Question

Parts 1 and 2 trade the predictions with an equal-weight top-and-bottom-20% book and simple rules for trading
less. Does a mean-variance optimizer, with a shrunk covariance matrix, a penalty on turnover, dollar
neutrality and a cap on gross exposure, earn a higher net Sharpe ratio on the same predictions than full
rebalancing to the quintile book?

**What is known when this is written.** The test spans were used by Parts 1 to 4, so this is a further
analysis of spans already looked at, not an untouched confirmation. The models are known to be mostly
momentum (Part 3), the non-price features added nothing detectable (Part 4), and no rebalancing rule differed
detectably from full rebalancing at 5 bps (Part 2). No optimizer has been run on any case, on either span.
This part changes portfolio construction only: the predictions, signals, universes, dates and cost model are
those of Part 2, and no feature is added (the Part 4 plan rules out a second feature round).

## Cases, spans, costs (unchanged from Part 2)

- Six cases: {30 ETFs, 49 industry portfolios} × {12-1 momentum, ridge, LightGBM}, from the case files of
  `scripts/07_build_targets.py`.
- Training span `[lo, hi)` and test span `[t0, t1)` from `qmb.rebalance.splits.split_indices`. Every run starts
  from the full quintile target of the week before its first week.
- 5 bps per unit of one-way turnover and 50 bps a year on short notional; sensitivity grid 0, 2, 5 and 10 bps,
  with the parameters re-selected on the training span at each cost.

## Risk model and expected returns

- **Covariance at week t:** Ledoit and Wolf (2004) shrinkage towards a scaled identity of the weekly returns of
  the 52 most recent weeks whose holding period ended on or before the signal date of week t (the weeks used by
  the `ic52` state variable of Part 2). All assets are present in every week of all six cases.
- **Expected return at week t:** alpha_i = IC × sigma_i × z_i (Grinold and Kahn), where z is the case's
  standardized prediction, sigma_i the square root of the diagonal of the covariance, and IC the mean weekly
  rank information coefficient of the case over its training span.

## Optimizer

Each week, from the drifted holdings d (Part 2 accounting), choose holdings h to

    maximize   alpha'h − (gamma / 2) h' Sigma h − kappa ||h − d||_1
    subject to sum(h) = 0,  ||h||_1 ≤ 2,

with kappa = m × cost / 10,000 (m = 1 charges the one-way cost once; m > 1 amortizes it over a shorter
expected holding period). Solved with CVXPY and the Clarabel interior-point solver. Accounting, costs and
borrow fees are those of `qmb.rebalance.env`.

| Parameter | Grid |
|---|---|
| gamma (risk aversion, weekly returns) | 1, 3, 10, 30, 100 |
| m (turnover penalty, in units of the cost) | 0, 1, 2, 4 |

**Selection (as in Part 2).** On the training span, at each cost level, the grid point with the highest net
Sharpe ratio among the admissible ones, where admissible means a mean short notional between 0.90 and 1.10.
Ties go to the smaller m, then the smaller gamma. If no grid point is admissible, the one whose mean short
notional is closest to 1.00 is used, and this is reported.

**Variants.** (1) The optimizer, gamma and m selected. (2) The optimizer without a turnover penalty (m = 0,
gamma selected), to separate the risk model from the penalty.

## Estimands and inference

- **Primary, per dataset:** the mean over the three signals of [net Sharpe(optimizer) − net Sharpe(full
  rebalancing)] on the test span at 5 bps. 95% interval from a stationary bootstrap (mean block 8 weeks, 2,000
  draws, seed 0) that resamples the same weeks for every signal and both strategies: the procedure and index
  paths of Part 2.
- **Secondary:** the same difference per case; the cost grid; blocks of 4 and 16 weeks; variant (2) against full
  rebalancing; the optimizer against each Part 2 rule, using the published weekly series in
  `reports/rebalancing/rules_weekly.csv` (5 bps).
- **Reported for every run:** selected parameters, training Sharpe ratio, test Sharpe ratio, annual net and gross
  return, cost drag, volatility, annual turnover, mean short notional, maximum drawdown, the training Sharpe ratio
  of every grid point, and the solver's largest constraint violation and non-optimal solves.

## Reading rule (fixed in advance)

The optimizer "improves on full rebalancing" only if its primary interval is above zero on both datasets. An
interval above zero on one dataset is reported as dataset-specific. Anything else is reported as no detectable
difference, with the change in turnover and volatility stated separately. No per-case or per-cost result is
promoted to a headline, and the result is reported whatever it is.
