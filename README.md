# Weekly equity signals under trading costs

How much of a weekly cross-sectional return forecast is left after trading costs, and can a different
rebalancing rule keep more of it? A pre-specified walk-forward study on 30 ETFs and 49 U.S. industry portfolios.

[![tests](https://github.com/lyz44git/walk-forward-backtest/actions/workflows/tests.yml/badge.svg)](https://github.com/lyz44git/walk-forward-backtest/actions/workflows/tests.yml)

<img src="figures/figure1.png" alt="Four-panel figure: cumulative net returns on industry portfolios with ETFs inset; real rank IC against placebo distributions; net Sharpe ratios with intervals; net Sharpe on ETFs against trading cost." width="100%">

**Figure 1 | Headline results.** **a**, Value of 1 invested in each long-short signal after costs, out of sample, on 49 industry portfolios (log scale); the dashed line splits the sample into halves, labeled with the LightGBM net Sharpe ratio in each. Inset: the same on 30 ETFs. **b**, Mean weekly rank IC of ridge models trained on shuffled targets (100 seeds per dataset) against the real IC (vertical lines), which exceeds every placebo seed. **c**, Net Sharpe ratio with stationary block-bootstrap 95% intervals. Every ETF interval spans zero, so no signal passes the rule fixed in advance on both datasets. **d**, Net Sharpe ratio on ETFs as the assumed one-way trading cost rises from 0 to 10 bps; the reported results use 5 bps. Drawn by [`figures/make_figure1.py`](figures/make_figure1.py) from `reports/`.

> **Takeaways.**
> 1. **Prediction.** Ridge and LightGBM rank next-week returns out of sample (rank IC 0.033–0.040, above every
>    shuffled-target placebo seed). After 5 bps costs they do no better than plain 12-1 momentum, and on tradable
>    ETFs no net Sharpe ratio is distinguishable from zero. Under the pass rule fixed in advance, nothing passes.
> 2. **Rebalancing.** Trading less does not rescue the result at 5 bps. Four simple rules and a PPO policy cut
>    turnover by 31–70% at the same exposure, but give up about as much return before costs as they save in costs:
>    no rule's net Sharpe differs detectably from full rebalancing. The simple rules pay off only at higher
>    costs (10 bps, industry portfolios: +0.11 to +0.19 net Sharpe), and the learned policy never beats them.

[Reproduce the results](#reproduce) · [Prediction design](#part-1-design) · [Rebalancing design](design/rebalancing_v2_plan.md)

**Question.** Do simple price-based features predict next-week relative returns, out of sample and after
trading costs? The same pipeline is run on two public datasets:

- **30 liquid U.S.-listed ETFs** (Yahoo Finance; equity indices, sectors, international, bonds, commodities).
  Out of sample 2011-04 to 2026-10, 810 weeks, entering at the next day's open.
- **49 U.S. industry portfolios** (Kenneth French data library, value-weighted, CRSP-based). Out of sample
  1975-07 to 2026-09, 2,672 weeks, entering at the next day's close.

The design (features, models, hyperparameters, costs, metrics and a pass/fail rule) was written to
`design/preregistration.json` before any out-of-sample number was computed. This is a self-written plan, not
an external registry. A first run was stopped part-way, and five changes were made after a code audit; all
of them are logged, with reasons, in `design/deviations.md`. Models, costs and the pass/fail rule never
changed.

## Part 1. Design

| Element | Choice |
|---|---|
| Rebalance | Weekly. The signal uses the close of the last trading day of the week. |
| Execution | ETFs: enter at the next day's open, exit at the open after the next signal (dividend-adjusted opens). Industries: enter at the next day's close (one-day lag). |
| Features (as of the signal date) | Returns over 5/21/63/126 days, 12-1 momentum, 21/63-day volatility, gap to the 200-day mean, 63-day drawdown, and (ETFs only) a dollar-volume surge. Each is ranked across assets every week. |
| Signals | 12-1 momentum and 1-week reversal (no fitting); ridge (alpha 1); LightGBM (300 trees, 7 leaves, fixed in advance) |
| Training | Expanding window, refit every 4 weeks. Purged: a week is used only after its holding period has ended, plus a 1-week embargo. Initial span 156 / 260 weeks. |
| Portfolio | Long the top 20%, short the bottom 20%, equal weight, 100% per leg; weights drift between rebalances |
| Costs | 5 bps one-way × turnover, plus 50 bps/yr borrow on the short side (grid: 0/2/5/10 bps) |
| Inference | Newey-West t for the mean rank IC; stationary block-bootstrap 95% CI for Sharpe; deflated Sharpe (N = 4); alpha vs the equal-weight universe's excess return |
| Pass rule (fixed in advance) | On both datasets: mean IC > 0 with t > 2, and the net Sharpe CI excludes 0 |

## Part 1. Results (`reports/main_results.csv`, `reports/decision.csv`)

| Dataset | Signal | Mean IC (NW t) | Sharpe before costs | Sharpe after costs (95% CI) | Turnover / yr |
|---|---|---|---|---|---|
| ETFs | 12-1 momentum | 0.035 (2.6) | 0.27 | 0.16 (−0.37, 0.65) | 29× |
| ETFs | Ridge | 0.033 (2.7) | 0.48 | 0.15 (−0.32, 0.59) | 94× |
| ETFs | LightGBM | 0.037 (3.6) | 0.58 | 0.15 (−0.35, 0.62) | 105× |
| ETFs | 1-week reversal | 0.016 (1.4) | 0.31 | −0.16 (−0.54, 0.23) | 159× |
| Industries | 12-1 momentum | 0.038 (6.8) | 0.49 | 0.38 (0.11, 0.65) | 26× |
| Industries | Ridge | 0.039 (7.6) | 0.55 | 0.31 (0.02, 0.60) | 63× |
| Industries | LightGBM | 0.040 (8.6) | 0.66 | 0.34 (0.05, 0.62) | 77× |
| Industries | 1-week reversal | 0.008 (1.8) | 0.25 | −0.36 (−0.64, −0.09) | 159× |

**What this shows**

1. **The prediction signal is real.** Ridge and LightGBM have a significant out-of-sample rank IC on both
   datasets. In a placebo test (training targets shuffled within each week, with a fresh shuffle at every
   refit) the real IC is above every placebo seed: ridge 100 of 100 seeds, LightGBM 10 of 10
   (`reports/placebo_summary.csv`, `fig2`). The placebo also shows that the Newey-West t over-rejects on the short
   ETF sample (|t| > 2 in 20% of LightGBM and 6% of ridge placebo seeds), so the ETF t-statistics of 2.6–3.6 are
   weaker evidence than they look.
2. **It survives costs only on the industry portfolios, which cannot be traded directly** (their 5 bps cost
   is an assumption).
   - Industries: model net Sharpe 0.31–0.34 (12-1 momentum 0.38), with CIs above zero, and an intercept of
     about 6% a year against the equal-weight universe's excess return (t 3.0–3.3; one benchmark, no
     momentum-factor control).
   - With features lagged one more week, the industry models' CIs include zero.
   - ETFs: net Sharpe is 0.15, and every CI spans zero. LightGBM's before-cost Sharpe of 0.58 is used up by
     about 100× annual turnover.
3. **The models do not beat plain 12-1 momentum after costs.** They rank slightly better (higher IC), but
   they trade two to three times as much.
4. **Verdict under the pass rule fixed in advance: no signal passes on both datasets.** The result is reported as not robust.
   Deflated Sharpe probabilities are 0.33–0.52.
5. **The edge decays.**
   - Industries: model net Sharpe is 0.6–0.7 in 1975–2001 and 0.09 in 2001–2026 (`reports/halves.csv`).
   - 1-week reversal loses money after costs on both datasets.

**Robustness** (`reports/robustness.csv`, `reports/cost_grid.csv`)
- **Features lagged one extra week:** the momentum and model ICs barely change (the signals are slow);
  reversal disappears. On industries the ridge and LightGBM net Sharpe CIs now include zero (lower bounds −0.03 and
  −0.02); on ETFs LightGBM's net Sharpe turns negative (−0.11).
- **ETFs entered at the next close instead of the next open:** the ICs barely change, but LightGBM's net Sharpe
  turns negative (−0.09); momentum and ridge stay near 0.15–0.19, with CIs spanning zero as before.
- **Costs:**
  - At 10 bps the ETF models turn negative.
  - Industry momentum keeps a net Sharpe of 0.30.

## Part 2. Rebalancing rules (`reports/rebalancing/`)

Part 1 leaves a question: the models lose half or more of their Sharpe ratio to costs, because they turn the book
over 63–105 times a year. Part 2 holds the signals, trading dates and cost model of Part 1 fixed and
changes only how the weekly target is traded. The plan, [`design/rebalancing_v2_plan.md`](design/rebalancing_v2_plan.md),
was committed before any number below was computed.

<img src="figures/figure2.png" alt="Left: net Sharpe ratio of each rebalancing rule minus full rebalancing, with 95% intervals, at 5 and 10 bps, for ETFs and industry portfolios, with annual turnover next to each rule. Right: mean net Sharpe ratio of each rule against the one-way trading cost." width="100%">

**Figure 2 | What trading less is worth.** Left, net Sharpe ratio of each rule minus full rebalancing, averaged over the three signals (12-1 momentum, ridge, LightGBM), with paired stationary-bootstrap 95% intervals at 5 bps (filled) and 10 bps (open); annual turnover at 5 bps and its change from full rebalancing are on the right of each row. Right, mean net Sharpe ratio against the assumed one-way cost; rule parameters are re-selected on the training span at each cost, and the PPO policy is the one trained at 5 bps. Drawn by [`figures/make_figure2.py`](figures/make_figure2.py) from `reports/rebalancing/`.

| Element | Choice |
|---|---|
| Cases | 2 datasets × 3 signals, with the Part 1 walk-forward predictions and targets |
| Spans | ETFs: select 2012-04 to 2018-12, test 2019-01 to 2026-10 (408 weeks). Industries: select 1976-07 to 2000-12, test 2001-01 to 2026-09 (1,342 weeks) |
| Full rebalancing | Trade to the target every week (the Part 1 portfolio) |
| Partial adjustment | Trade a fixed fraction of the gap between holdings and target |
| No-trade band | Per asset, trade only to the edge of a band around its target weight |
| Rank buffer | A held name stays in its leg while it ranks inside a wider quantile; vacancies go to the best-ranked names |
| Signal smoothing | Exponentially weighted mean of past weekly predictions, then the usual top and bottom 20% |
| PPO policy | Actor-critic with a Beta policy over the adjustment fraction, 9 state features known at the signal date, written in PyTorch; 5 seeds, trained on the selection span only, with gradients averaged across 8 processes (`torch.distributed`) |
| Selection | Each rule's one parameter is chosen by net Sharpe on the selection span, among values that keep the short-leg notional within 0.90–1.10 of target |
| Estimand | Per dataset: mean over the three signals of net Sharpe (rule) − net Sharpe (full), with a stationary bootstrap that resamples the same weeks for every series (mean block 8 weeks, 2,000 draws) |
| Reading rule (fixed in advance) | A rule recovers performance only if its interval is above zero on both datasets |

Test-span results at 5 bps, averaged over the three signals (`rules_results.csv`, `rules_pooled.csv`):

| Dataset | Rule | Turnover / yr | Return before costs | Costs | Net Sharpe | Difference from full, 5 bps (95% CI) | Difference from full, 10 bps (95% CI) |
|---|---|---|---|---|---|---|---|
| ETFs | Full rebalancing | 73× | 11.3% | 4.2% | 0.42 | — | — |
| ETFs | Partial adjustment | 30× | 7.8% | 2.0% | 0.39 | −0.03 (−0.18, +0.14) | +0.09 (−0.08, +0.27) |
| ETFs | No-trade band | 49× | 9.3% | 3.0% | 0.39 | −0.03 (−0.14, +0.09) | +0.04 (−0.07, +0.16) |
| ETFs | Rank buffer | 27× | 8.5% | 1.9% | 0.41 | −0.02 (−0.22, +0.20) | +0.12 (−0.10, +0.35) |
| ETFs | Signal smoothing | 43× | 7.3% | 2.7% | 0.27 | −0.15 (−0.35, +0.05) | +0.05 (−0.20, +0.32) |
| ETFs | PPO policy | 22× | 5.5% | 1.6% | 0.26 | −0.17 (−0.39, +0.07) | −0.02 (−0.24, +0.22) |
| Industries | Full rebalancing | 55× | 5.8% | 3.2% | 0.14 | — | — |
| Industries | Partial adjustment | 34× | 4.8% | 2.2% | 0.15 | +0.01 (−0.04, +0.06) | +0.11 (+0.02, +0.20) |
| Industries | No-trade band | 28× | 5.4% | 2.0% | 0.23 | +0.09 (−0.01, +0.19) | +0.19 (+0.07, +0.30) |
| Industries | Rank buffer | 20× | 4.7% | 1.5% | 0.19 | +0.04 (−0.05, +0.15) | +0.15 (+0.06, +0.26) |
| Industries | Signal smoothing | 28× | 5.0% | 1.9% | 0.17 | +0.03 (−0.05, +0.12) | +0.15 (+0.03, +0.27) |
| Industries | PPO policy | 38× | 4.6% | 2.4% | 0.13 | −0.01 (−0.07, +0.04) | +0.03 (−0.03, +0.09) |

**What this shows**

1. **Every rule trades much less at the same exposure.** Turnover falls by 33–70% on ETFs and 31–63% on
   industries, with the short leg between 0.90 and 1.11 of target on the test span.
2. **At 5 bps the saving is offset by lost return.** On ETFs the rank buffer saves 2.3 points of costs a year
   and gives up 2.8 points of return before costs. The forecasts are for one week, so a position that is not
   refreshed loses its edge about as fast as it saves costs. No interval excludes zero on either dataset; under
   the reading rule, no rule recovers performance.
3. **The value of trading less rises with the cost.** At 10 bps all four simple rules beat full rebalancing on
   the industry portfolios (+0.11 to +0.19, intervals above zero); the ETF intervals still include zero. The cost
   grid is a pre-specified sensitivity, not the headline.
4. **The learned policy adds nothing over the simple rules.** Averaged over signals, PPO is below every
   simple rule on both datasets; on industries it is worse than the no-trade band (−0.10, interval −0.19 to −0.02). The conclusion is that the
   cost drag in Part 1 is a property of the signal's horizon, not of the trading rule.
5. **Block length does not matter.** With mean blocks of 4 and 16 weeks every 5 bps conclusion is unchanged.

**How this design came about.** A first design ([`design/rebalancing_v1_preregistration.md`](design/rebalancing_v1_preregistration.md),
results in `reports/rebalancing/v1/`) compared full rebalancing, partial adjustment, a no-trade band and PPO, with
PPO meeting its improvement rule in 0 of 18 comparisons. A check of its weekly output showed that selecting the band
width by Sharpe alone admitted a degenerate rule: a band as wide as a position never opens one, and the selected ETF
LightGBM band held 3% of its target exposure, with a return of exactly zero in 359 of 408 weeks. The second design
adds the exposure constraint, two rules that stay fully invested by construction, a pooled estimand and paired
intervals. It was written after the first test-span results were known, so it is a second iteration on a span that
had been looked at once.

**Distributed training.** The same PPO script runs in one process or under `torchrun`. On one machine, one seed
of one case took 91 s in one process and 56 s in eight, with identical parameters, weekly actions and test metrics
(`v1/timing.csv`, `v1/distributed_parity.csv`).

## Verification

- **Tests:** 18 pytest tests on synthetic data (`tests/`). Eight check the engine: features use no future
  prices, the target spans exactly entry → exit, purging works, pure noise gives no skill, a planted lagged
  signal is found, and the long-short accounting is correct. Ten check the rebalancing rules: cost and
  borrow accounting, that the neutral parameter of each rule returns the Part 1 targets, that both legs stay
  full and disjoint, that a rule cannot see later predictions, and that the degenerate band is caught. Seven
  more run when the cached targets are present: full rebalancing reproduces `reports/ls_weekly.csv` to 1e-12
  for all six cases, and the PPO state uses only completed weeks.
- **Regeneration:** after the rebalancing code was merged, the Part 1 weekly returns and all five first-design
  result files were regenerated byte for byte.
- **Code review:** a structured review by lens (timing and look-ahead, walk-forward, accounting, statistics,
  data) found two material errors, both fixed before the reported run: a placebo that reused one permutation
  across refits and an alpha regression that did not subtract the risk-free rate. Three lower-priority issues were
  also fixed. All five changes are in `design/deviations.md`.
- **Independent recomputation:** `scripts/06_recompute_check.py` recomputes the mean IC, Newey-West t, hit rate,
  gross and net Sharpe, net return and turnover for all eight dataset-signal pairs from the saved weekly series,
  with formulas written separately from the package. The largest difference from the reported values is 1.4e-14
  (`reports/recompute_check.csv`). It also redraws the Sharpe bootstrap with 20 seeds: whether the CI excludes
  zero does not depend on the seed for any signal (`reports/bootstrap_seed_sensitivity.csv`).
- **Data quality:** `reports/data_quality.json` covers calendar gaps, split and dividend adjustments, and
  large moves; every outlier traced to a real market event.

## Limitations

- **Survivorship and selection bias.** The ETF universe is today's 30 surviving funds; no fund can drop out.
- **Execution.** Prices are Yahoo's consolidated opens. Some crisis-day opens may not have been tradable at
  size. Industry portfolios are not tradable, so their costs are hypothetical.
- **No capacity or market-impact model.** Borrow cost is a flat 50 bps a year.
- **Weekly horizon, price-only features.** No fundamental, options or intraday data.

## Reproduce

```bash
pip install -r requirements.txt
python3 scripts/01_download.py      # Yahoo (yfinance) + Ken French files into data/raw (git-ignored)
python3 scripts/02_check_data.py
python3 -m pytest tests -q
python3 scripts/03_backtest.py      # ~35 min on a 32-core machine
python3 scripts/04_placebo.py && python3 scripts/04_placebo.py --summary
python3 scripts/05_figures.py
python3 scripts/06_recompute_check.py   # reads reports/ only

# Part 2
python3 scripts/07_build_targets.py     # weekly targets of the six cases into tmp/rebalance
pip install -r requirements-ppo.txt     # PyTorch, only for the PPO policy
scripts/run_ppo.sh main                 # 5 seeds x 6 cases, 8 processes
python3 scripts/09_rebalancing_rules.py # all rules, cost grid, paired intervals (about 30 s)
python3 scripts/10_ppo_v1_evaluate.py   # first-design tables, parity check and timing
python3 figures/make_figure2.py
```

Raw data are not redistributed: Yahoo data are for personal use, and the Ken French files can be downloaded
from the library.

## License

Code is released under the MIT License (see `LICENSE`). Yahoo Finance and Kenneth French data are not covered and
are not redistributed here.

**Use.** This is a non-commercial research project. The data and all derived outputs in this repository are used
for research only.

## Notes

Developed with an AI-assisted coding workflow. The research questions, analysis plans and interpretation of results
were led by the author.

## Layout

```
design/             preregistration.json and deviations.md (Part 1); rebalancing_v2_plan.md, rebalancing_v1_* (Part 2)
src/qmb/            data.py (loaders), engine.py (schedule, features, walk-forward, portfolios), stats.py (metrics)
src/qmb/rebalance/  cases.py (targets), env.py (accounting, simulator), rules.py (rank buffer, smoothing), ppo.py, splits.py
scripts/            01 download · 02 data checks · 03 backtests + robustness · 04 placebo · 05 figures · 06 recompute check
                    07 targets · 08 PPO training · 09 rebalancing rules · 10 first-design evaluation
tests/              synthetic-data tests of the engine and the rebalancing rules
reports/            result tables (CSV/JSON) and figures; reports/rebalancing/ for Part 2
data/, tmp/         raw and clean data, caches, logs (git-ignored)
```
