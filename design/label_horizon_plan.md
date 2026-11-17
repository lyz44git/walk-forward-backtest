# Label horizon (Part 3): plan written 2026-11-16, before any number in it was computed

## Why

Part 2 found that trading less saves about as much in costs as it gives up in return before costs, and read
this as a property of the signal's horizon: the models are fitted to next-week returns. That reading was an
inference. This plan tests it in two steps, both fixed before any of their numbers were computed. The Part 1
and Part 2 results on the same samples were known when it was written, so the evaluation below is a further
iteration on samples that have been looked at, not an untouched confirmation.

## Step A. Diagnostics from the existing Part 1 predictions (no refitting)

For 12-1 momentum, ridge and LightGBM on both datasets, over the full Part 1 out-of-sample span:

1. **Decay.** Mean weekly rank IC of the week-t prediction with the one-week return j weeks later, j = 0..12
   (j = 0 is the Part 1 IC), with Newey-West t (4 lags); and the mean rank autocorrelation of predictions at
   lags 1, 2, 4, 8, 13 weeks.
2. **Quintiles.** Mean one-week return of each prediction quintile (equal weight), and the number of adjacent
   quintile pairs that are in order.
3. **Breakeven cost.** The one-way cost at which the mean net return is zero: (return before costs - borrow) /
   turnover, for the Part 1 portfolios and for each Part 2 rule at its 5 bps parameters.
4. **Factor attribution.** Regression of the weekly net long-short return on the market, size and value
   factors compounded over the same holding period, and on those plus the 12-1 momentum long-short return of
   the same universe (for ridge and LightGBM), with Newey-West t (4 lags). The published momentum factor is
   not used; the same-universe momentum portfolio is the stricter control here.

## Step B. Longer label horizons

**Question.** Does fitting the same models to a longer forward return give a slower signal that keeps more
net performance, with everything else in Part 1 unchanged?

| Element | Choice |
|---|---|
| Models | Ridge and LightGBM, features and hyper-parameters exactly as in Part 1 |
| Label | Cross-sectional rank of the return from entry of week k to the exit of week k+H-1; H = 1 is Part 1 |
| Horizons | H = 4 (primary), H = 13 (secondary) |
| Purge | A week j is used for training at refit week k only if the exit of week j+H-1 is before the signal date of k minus the one-week embargo |
| Test weeks | The Part 1 out-of-sample weeks, unchanged (the first refits train on H-1 fewer weeks) |
| Portfolio, costs, execution | As in Part 1: weekly rebalance to the top and bottom 20%, 5 bps and 50 bps/yr borrow, cost grid 0/2/5/10 |
| Evaluation | Weekly holding-period returns, exactly as in Part 1 |

**Primary estimand, per dataset:** the mean over ridge and LightGBM of [net Sharpe (H = 4) - net Sharpe
(H = 1)] at 5 bps on the full out-of-sample span, with a 95% interval from a stationary bootstrap that
resamples the same weeks for every series (mean block 8 weeks, 2,000 draws; blocks 4 and 16 as a check).

**Secondary:** the same for H = 13; each model separately; the cost grid; rank IC against the one-week
return, turnover, and return before costs for each horizon; the Part 2 test span alone.

**Reading rule (fixed in advance).** A longer label "keeps more net performance" only if the primary interval
is above zero on both datasets. Above zero on one dataset is reported as dataset-specific. Otherwise the result
is reported as no detectable difference, with the turnover change stated separately.

**What would contradict the Part 2 reading.** If the IC in Step A does not decay with j, or if H = 4 lowers
turnover without lowering the return before costs, the statement that the cost drag comes from the signal's
horizon is wrong as written and will be revised.
