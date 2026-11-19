# Part 6, what trading the ETFs costs, measured from NASDAQ order-book messages: plan written 2026-11-19, before any order, quote or trade message of the sample was read

## Question

Parts 1 to 5 charge the ETF portfolios an assumed cost of 5 bps per unit of one-way turnover, with a sensitivity
grid of 0, 2, 5 and 10 bps. At full rebalancing the ETF models break even at 6.9–7.5 bps and 12-1 momentum at
14.1 bps (`reports/diagnostics/breakeven_cost.csv`). How large is the cost of a small marketable order in these
30 ETFs when it is measured from the full NASDAQ order book, and where does it fall on that grid?

**What is known when this is written.** Only the first 4 MB of one file were decoded, to confirm the format; they
held system-event, stock-directory and administrative messages and no order, quote or trade. No result of this
part has been computed.

## Data

- NASDAQ TotalView-ITCH 5.0 sample files published by NASDAQ (`emi.nasdaq.com/ITCH/Nasdaq ITCH/`): the full days
  of 8 December 2025 (`S120825-v50.txt.gz`, 8,775,891,119 bytes) and 9 December 2025 (`S120925-v50.txt.gz`).
  They were chosen before any content was read, as two consecutive regular sessions with files under 9 GB.
- The files are not redistributed. Only per-ETF aggregates are kept in `reports/tick/`.
- Universe: the 30 ETFs of Part 1.

## Book

- NASDAQ's displayed order book is rebuilt for the 30 ETFs from add, execute, cancel, delete and replace messages.
- The best bid and ask are NASDAQ's best displayed prices. They are at or outside the national best bid and offer,
  so a NASDAQ quoted spread is an upper bound on the national quoted spread at the same moment.
- Executions against displayed orders are signed by the side of the resting order, which the messages report: an
  executed resting sell means the incoming order bought. No trade-classification algorithm is needed for them.
- Executions against non-displayed orders (trade messages) carry no usable side indicator; they are reported as a
  share of executed volume, with an effective half-spread signed by the quote rule (above the midpoint = buy) as a
  secondary measure. Executions at the midpoint are left unsigned and excluded from that measure.

## Measures (per ETF and day, regular session 9:30:00 to 16:00:00 Eastern)

1. **Quoted half-spread:** (ask − bid) / 2 over the midpoint, in bps, time-weighted over the time both sides are
   present and the ask exceeds the bid.
2. The same over the first five minutes (9:30 to 9:35), the window nearest the backtest's execution at the open.
3. **Effective half-spread:** D × (P − M) / M in bps for each printable execution against a displayed order,
   where D = +1 when the incoming order bought and M is the NASDAQ midpoint just before the execution;
   share-weighted. Crosses are excluded and reported separately.
4. **Price impact** at 1 and 5 minutes: D × (M at t + delta − M at t) / M at t, share-weighted, with M at
   t + delta the midpoint at the first whole second at or after t + delta; only executions at least delta before
   16:00 enter. **Realized half-spread** = effective − impact on the same executions.
5. **Depth** at the best bid and ask in dollars, time-weighted.
6. **Opening cross** price and shares, where NASDAQ runs one.

## Estimand and reading

- **Primary:** the equal-weighted mean across the 30 ETFs of the share-weighted effective half-spread, per day,
  with the median and range across ETFs. The equal weight matches the equal-weight legs of the backtest.
- **Reading, fixed in advance.** If the primary is below 2 bps on both days, the 5 bps assumption overstates the
  cost of a small marketable order on NASDAQ on these days, and the ETF results are read at the 0 to 2 bps end of
  the grid. If it is between 2 and 5 bps, the assumption is of the right size. If it is above 5 bps on either
  day, the assumption understates it.
- **Not measured:** the cost of the opening auction on the primary listing exchange (most of these ETFs list on
  NYSE Arca, not NASDAQ), orders larger than the displayed depth, borrow costs and days other than these two.
- No number in Parts 1 to 5 is changed by this part.

## Checks (reported, and failures stop the run)

- Every message the parser reads (system event, stock directory, add, execute, cancel, delete, replace, trade,
  cross trade, broken trade) has the length the ITCH 5.0 specification gives its type.
- Every execute, cancel, delete and replace refers to an order added earlier the same day.
- No price level ever holds negative shares.
- The count of crossed-book events (bid at or above ask) after each message is reported.
- Tests on synthetic message streams written from the specification check the parser and the book.
