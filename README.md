# Pre-specified walk-forward backtest: cross-sectional return prediction on ETFs and industry portfolios

[![tests](https://github.com/lyz44git/walk-forward-backtest/actions/workflows/tests.yml/badge.svg)](https://github.com/lyz44git/walk-forward-backtest/actions/workflows/tests.yml)

<img src="figures/figure1.png" alt="Four-panel figure: cumulative net returns on industry portfolios with ETFs inset; real rank IC against placebo distributions; net Sharpe ratios with intervals; net Sharpe on ETFs against trading cost." width="100%">

**Figure 1 | Headline results.** **a**, Value of 1 invested in each long-short signal after costs, out of sample, on 49 industry portfolios (log scale); the dashed line splits the sample into halves, labeled with the LightGBM net Sharpe ratio in each. Inset: the same on 30 ETFs. **b**, Mean weekly rank IC of ridge models trained on shuffled targets (100 seeds per dataset) against the real IC (vertical lines), which exceeds every placebo seed. **c**, Net Sharpe ratio with stationary block-bootstrap 95% intervals. Every ETF interval spans zero, so no signal passes the rule fixed in advance on both datasets. **d**, Net Sharpe ratio on ETFs as the assumed one-way trading cost rises from 0 to 10 bps; the reported results use 5 bps. Drawn by [`figures/make_figure1.py`](figures/make_figure1.py) from `reports/`.

> **Takeaway.** Ridge and LightGBM find a real out-of-sample ranking signal (it beats every placebo seed), but after
> trading costs they do no better than plain 12-1 momentum, and on tradable ETFs no signal has a net Sharpe ratio
> distinguishable from zero. Under the pass rule fixed in advance, nothing passes.

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

## Design

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

## Results (`reports/main_results.csv`, `reports/decision.csv`)

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

## Verification

- **Tests:** 8 pytest tests on synthetic data (`tests/`). They check that features use no future prices,
  that the target spans exactly entry → exit, that purging works, that pure noise gives no skill, that a
  planted lagged signal is found, and that the long-short accounting is correct.
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
design/      preregistration.json, deviations.md
src/qmb/     data.py (loaders), engine.py (schedule, features, walk-forward, portfolios), stats.py (metrics)
scripts/     01 download · 02 data checks · 03 backtests + robustness · 04 placebo · 05 figures · 06 recompute check
tests/       synthetic-data tests of the engine
reports/     result tables (CSV/JSON) and figures
data/, tmp/  raw and clean data, logs (git-ignored)
```
