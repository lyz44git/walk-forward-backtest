# Non-price features (Part 4): plan written 2026-11-17, before any of the new data was downloaded or read

## Why

Part 3 found that the ridge and LightGBM models built on price features are mostly 12-1 momentum plus a
faster component that does not pay for its turnover: after controlling for the momentum portfolio of the same
universe, the industry models' alpha is 1.1-1.5% a year (t 0.9-1.3) and the ETF models' alpha is negative.
Rebalancing rules (Part 2) and longer labels (Part 3) did not change this. More models on the same price
features would be a search over samples already looked at.

This part asks a different question: does information that is not a function of the assets' own past prices
add to the price-based models?

## What is and is not untouched

The test weeks are the Part 1 out-of-sample weeks, and the price-based results on them are known. The new
feature columns have not been downloaded, plotted or evaluated when this plan is written. The feature list,
timing rules and reading rule below are fixed here; any later change is logged in `design/deviations.md`.

The published evidence on these variables is known in general terms: industry-level book-to-market is a weak
predictor of industry returns, and carry is a documented predictor across asset classes. A null result on
industries is the more likely outcome and will be reported as such.

## Data

Both sources are public and free; raw files stay git-ignored as in Part 1.

| Dataset | New source | Fields |
|---|---|---|
| Industries | Kenneth French data library, `49_Industry_Portfolios_CSV.zip` (monthly/annual file; one new download) | number of firms (monthly), average firm size (monthly), sum of BE / sum of ME (annual) |
| Industries | The daily file already downloaded | equal-weighted daily returns |
| ETFs | The Yahoo files already downloaded | cash distributions (`Dividends` column, by ex-date) |

## Features (fixed list)

Industries:

| Name | Definition | Available from |
|---|---|---|
| `log_bm` | log of sum BE / sum ME for year t | first signal date in July of year t, held to the following June |
| `log_bm_timely` | `log_bm` minus the industry's cumulative log value-weighted return from the end of December t-1 to the signal date | same |
| `log_size` | log of average firm size for month m | first signal date after the end of month m+1 |
| `firm_growth_12m` | log number of firms in month m minus the same in month m-12 | same |
| `breadth_63d` | 63-day equal-weighted return minus 63-day value-weighted return | signal date |

ETFs:

| Name | Definition | Available from |
|---|---|---|
| `carry_12m` | sum of distributions with ex-date in the last 252 trading days, divided by the unadjusted close | signal date |
| `carry_12m_in_class` | `carry_12m` minus its mean over the ETFs of the same class (U.S. equity indices, international equity, sectors and REITs, bonds, commodities, as grouped in `scripts/01_download.py`) | signal date |

Rules applied to every new feature:

- Cross-sectional rank on each signal date, scaled to [-0.5, 0.5], as in Part 1. Ties get the average rank.
- A missing value is set to 0 after ranking (the middle of the range), so that the rows used for fitting and
  testing are exactly the Part 1 rows and the comparison with Part 1 is paired week by week.
- `capital gains` distributions are not counted in carry.

**Step 0, before any return is joined to the new columns:** read the source's own description of each field
and confirm the dating convention assumed above (in particular that BE/ME for year t uses book equity of the
fiscal year ending in t-1 and is therefore public by the end of June t); tabulate coverage by year and
industry. If a convention differs, the availability date is moved later, never earlier, and the change is
logged. If a field is missing for more than 20% of industry-weeks in the training-plus-test span, that feature
is dropped and the drop is logged.

## Models and evaluation

| Element | Choice |
|---|---|
| Models | Ridge and LightGBM, hyper-parameters exactly as in Part 1 |
| Feature sets | P: Part 1 price features. P+N: price plus the new features of that dataset. N: new features only |
| Label, training, purge, embargo | As in Part 1 (one-week label) |
| Test weeks | The Part 1 out-of-sample weeks |
| Portfolio, costs | As in Part 1: top and bottom 20%, weekly, 5 bps and 50 bps/yr borrow, cost grid 0/2/5/10 |
| Single sorts (no fitting) | Each new feature alone, signed by its published direction: high `log_bm`, `log_bm_timely`, `carry_12m`, `carry_12m_in_class` long; high `log_size`, `firm_growth_12m` short; `breadth_63d` long |

**Primary estimand, per dataset:** the mean over ridge and LightGBM of [net Sharpe (P+N) - net Sharpe (P)] at
5 bps on the full out-of-sample span, with a 95% interval from a stationary bootstrap that resamples the same
weeks for every series (mean block 8 weeks, 2,000 draws; blocks 4 and 16 as a check).

**Secondary:**

1. Alpha of the P+N net long-short return on market, size, value and the same-universe 12-1 momentum portfolio
   (Newey-West, 4 lags), the regression that Part 3 used.
2. Rank IC, turnover, return before costs and breakeven cost for P, P+N and N.
3. Each model separately; the cost grid; the first and second half of the out-of-sample span.
4. The single sorts: IC, net Sharpe with interval, and correlation of each with the momentum portfolio.
5. Placebo: the new features permuted across assets within each week before fitting P+N. The primary
   difference should be indistinguishable from zero.
6. Timing check: the new industry features lagged a further 12 months (annual) and 3 months (monthly).
7. Ridge coefficients and LightGBM gain share of the new features across refits.

**Reading rule (fixed in advance).**

- "The new information adds to the price-based models" only if the primary interval is above zero on that
  dataset. The two datasets use different information and are read separately.
- "An edge beyond momentum" additionally requires the alpha in secondary item 1 to have t > 2 for both models.
- Otherwise the result is reported as no detectable gain, with the secondary numbers shown as they are.
- No feature is added, removed or redefined after results are seen. A second round of features would be a new
  plan with its own date and would say that the first round's results were known.

## Known limits

- The French files are the current vintage, not point-in-time: CRSP and Compustat revisions and backfilled
  book values are in them. The availability lags above reduce but do not remove this.
- Industry BE/ME is annual and firm counts are monthly; the test is weekly. The new features move slowly, so
  any gain should show as lower turnover at similar return before costs, not as a faster signal.
- In a 30-ETF universe that mixes asset classes, raw carry mostly sorts bonds against commodities, which pay
  nothing. `carry_12m_in_class` is included for that reason; with four to ten ETFs per class it is noisy.
- The ETF out-of-sample span is 810 weeks; the Part 2 and Part 3 intervals on it were wide and will be here.

## Not in this plan

Customer-supplier links between industries, analyst or earnings data, short interest, fund flows and
single-stock fundamentals. The first needs an input-output concordance that would add judgement calls; the
others are not available from public sources at the quality this repository requires.
