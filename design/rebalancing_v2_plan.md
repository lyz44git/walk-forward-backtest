# Rebalancing rules, second design (written 2026-11-16, before any v2 test-span number was computed)

## Why a second design

The first design (`rebalancing_v1_preregistration.md`) compared full rebalancing, a fixed partial-adjustment
rate, a per-asset no-trade band and a PPO policy, and chose the two baseline parameters by net Sharpe on the
training span. Its frozen test results are kept in `reports/rebalancing/v1/`. A check of the weekly output
after that evaluation showed that the selection rule admits a degenerate solution: a band at least as wide
as one target position never opens a new position, so the book runs off. For the ETF LightGBM case the
selected band had a mean short-leg notional of 0.03 against a target of 1.00, and its net return was exactly
zero in 359 of 408 test weeks. A Sharpe ratio on a book that is almost empty says nothing about trading
less, so the v1 band column cannot support a claim about rebalancing.

**Status of this design.** The v1 test-span results were known when this plan was written. Everything
below is therefore a second iteration on a test span that has been looked at once, not an untouched
confirmation. All choices here are fixed before any v2 test number is computed, and every parameter is
selected on the training span only.

## Question

At the same signals, trading dates, cost model and gross exposure as the main backtest, how much of the
net-of-cost performance lost to turnover can a simple rebalancing rule recover, and does a learned policy
add anything over the simple rules?

## Data, signals, spans (unchanged from v1)

Six cases: {30 ETFs, 49 industry portfolios} x {12-1 momentum, ridge, LightGBM}, with the walk-forward
predictions and targets of the main backtest. Training span and test span as in v1 (ETFs: test from the
first signal on or after 2019-01-01; industries: 2001-01-01). Every rule starts the test span from the full
target of the week before it.

## Cost model (unchanged)

5 bps per unit of one-way turnover and 50 bps per year on short notional. Sensitivity grid 0 / 2 / 5 / 10
bps, as in the main backtest; at each cost level the rule parameters are re-selected on the training span
at that cost.

## Rules

| Rule | Definition | Grid (selected on the training span) |
|---|---|---|
| Full | Trade to the weekly target (top and bottom 20%, equal weight) | none |
| Partial adjustment | Trade a fixed fraction eta of the gap between drifted holdings and the target | eta in {0.05, 0.10, ..., 1.00} |
| No-trade band | Per asset, trade only to the edge of [target - b, target + b], b = m / k | m in {0, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2} |
| Rank buffer | A held name stays in its leg while its rank is inside the top (bottom) m x 20%; vacancies are filled from the best-ranked names; k names per leg, equal weight, traded in full | m in {1, 1.25, 1.5, 1.75, 2} |
| Signal smoothing | Exponentially weighted mean of past weekly standardized predictions (half-life h weeks), then the usual top and bottom 20%, traded in full | h in {0, 1, 2, 4, 8} |
| PPO (v1, frozen) | The five v1 seed policies, unchanged; trained at 5 bps | none |

Selection criterion: net Sharpe on the training span, ties to the grid value closest to full rebalancing.

**Exposure constraint (new).** A grid value is admissible only if its mean short-leg notional on the training
span is between 0.90 and 1.10 (target 1.00). Full rebalancing is always admissible. Rank buffer and signal
smoothing hold k names per leg at 1/k by construction. The realized test-span exposure of every rule is
reported next to its Sharpe ratio.

The rank buffer and signal smoothing use only information available at the signal date: the previous
week's holdings and past predictions.

## Estimands and inference

- **Primary, per dataset and rule:** the mean over the three signals of [net Sharpe(rule) - net Sharpe(full
  rebalancing)] on the test span. 95% interval from a stationary bootstrap (mean block 8 weeks, 2,000
  draws) that resamples the same weeks for every signal and both strategies. Averaging over signals keeps
  the number of primary comparisons at 2 datasets x 5 rules.
- **Secondary:** the same difference for each of the six cases; the difference between PPO and the best
  simple rule (the one selected on the training span is not re-chosen on the test span: PPO is compared
  with each simple rule); block lengths 4 and 16; the cost grid.
- Reported for every rule: annual turnover, gross and net annual return, cost drag, mean short notional.
- A bootstrap draw with zero return variance is dropped and the number of valid draws is reported.

## Reading rule (fixed in advance)

A rule "recovers performance" only if its primary interval is above zero on **both** datasets. A primary
interval above zero on one dataset is reported as dataset-specific. Anything else is reported as no
detectable difference, with the turnover reduction stated separately. No per-case result is promoted to a
headline.
