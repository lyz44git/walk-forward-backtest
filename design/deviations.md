# Changes to the pre-registered design

`design/preregistration.json` was written before any out-of-sample result was computed. A first full run was
stopped part-way (the ETF stages had finished; their numbers were printed to a log but never used). A
structured code review by lens (timing, walk-forward, accounting, statistics, data) then ran, with each medium or
higher finding checked again before it was accepted. The changes below came out of that audit. Every result in `reports/` comes from the run made
after these changes.

| # | Change | Why | Effect on reported numbers |
|---|---|---|---|
| 1 | **Placebo test redesigned.** Training targets are still permuted across assets within each week, but a fresh permutation is drawn at every refit, and the test is repeated over many seeds (ridge 100, LightGBM 10). The real model's mean IC is compared with that placebo distribution (empirical p-value). | The pre-registered version drew one permutation and reused it across nested expanding windows. That gives a persistent random coefficient direction rather than a null distribution; a fresh permutation at every refit removes it. | The placebo rows only. Main results are unaffected. |
| 2 | **Alpha regression uses the benchmark's excess return.** The long-short net return is regressed on (equal-weight return minus the risk-free rate compounded over the same holding week). | The long-short return is already an excess return; regressing it on a total return biases the intercept by beta × RF. RF comes from the Fama-French daily file, which ends 2026-09-30; the last five ETF weeks carry the last daily rate forward. | `ls_alpha_vs_ew_annual` and `ls_alpha_t` only. |
| 3 | **Initial training span made exact.** The first out-of-sample week is now the one whose first refit trains on exactly 156 (ETF) / 260 (industries) weeks. | The first version counted the span before applying the purge and embargo, so the first refit used 154 / 258 weeks. | The out-of-sample period starts two weeks later (ETF 2011-04-21, industries 1975-07-11). |
| 4 | **Decision rule computed in code** (`reports/decision.csv`). | The pre-registered rule was stated but not evaluated by the script. | None; it makes the verdict explicit. |
| 5 | **Deflated Sharpe in robustness rows uses the main run's N = 4 trials.** | The first version passed only the variant's own signals (N = 2 for the placebo). | Robustness rows only. |

Disclosed limitations (no code change):
- **ETF universe chosen with hindsight.** The 30 ETFs are today's surviving, liquid funds. No ETF can drop out, so the sample has survivorship and selection bias. A point-in-time universe rule would need delisted-fund data, which Yahoo does not provide.
- **Execution prices.** Opens are Yahoo's consolidated opening prints. Some crisis-day opens (for example XLB on 2020-03-16, a circuit-breaker day) may not have been tradable at size. Industry portfolios are not tradable instruments; their costs are hypothetical.
- **LightGBM threads.** `num_threads` was raised from 8 to 16 for speed (`deterministic=True` is kept). This is not a design change.

## Part 4 (non-price features): notes from step 0 and the run, 2026-11-17

Nothing in `design/nonprice_features_plan.md` was changed after results were seen. The points below were
settled at step 0, before any return was joined to the new columns, or are details the plan left open.

- **Dating of BE/ME.** Neither the industry file nor its description page states the convention for the
  "Sum of BE / Sum of ME" table. The plan's assumption (the row for year t uses book equity of the fiscal year
  ending in t-1 and market equity of December t-1) is the library's general definition and is consistent with
  a row for 2026 in a file built from the September 2026 database. It was not confirmed by an explicit
  statement for this file; the 12-month timing check covers the case where it is wrong.
- **Negative book equity.** The log is undefined where the industry's summed book equity is negative (0.4% of
  rows, for example tobacco in recent years). Those values are treated as missing and set to the middle rank,
  as the plan specifies for missing values.
- **Coverage.** No feature came near the 20% rule; all seven were kept.
- **Placebo seeds.** The plan did not fix the number: 20 for ridge and 3 for LightGBM per dataset.
- **Placebo not centred on zero for LightGBM on ETFs.** The plan expected the placebo difference to be
  indistinguishable from zero. It is for ridge; for LightGBM on ETFs it is -0.20 to -0.39. This is reported in
  the README and does not change the reading of the primary estimand.
- **Ties in carry.** ETFs that pay nothing have equal carry and equal rank; in the single sort the portfolio
  rule breaks the tie by ticker order.
- **Distributions as provided.** Yahoo's distribution records are used without correction (for example a
  large XLE entry on 2019-12-30).
