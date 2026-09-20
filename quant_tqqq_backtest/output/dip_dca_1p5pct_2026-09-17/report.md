# Waiting for a 1.5% dip versus weekly investing

User-confirmed interpretation: only new/idle money waits; previously purchased TQQQ is never sold. This is NOT daily all-account switching and is NOT the earlier 200-day trend strategy.

## Rules

- Deposit $100 at the first session of every ISO week, including a partial starting week. No initial lump sum, borrowing, withdrawals, or trading authorization.
- Signal: QQQ split/dividend-adjusted close-to-close return <= -1.5%, observed before the execution session. The preceding session may be before the window starts.
- At the execution time, if the prior session triggered: sell all idle bond units, use the proceeds plus available deposits to buy TQQQ. Hold existing TQQQ indefinitely.
- Otherwise, buy short Treasury units with new deposits (or retain 0%-yield cash in the control). Continue weekly saving even while waiting.
- Fractional adjusted-price units approximate reinvested distributions. Charge 5 bps on each buy and sell. No taxes, commissions, terminal liquidation, or settlement constraints modeled.
- Risk measures below are nominal equity/cumulative contributions minus one. They are NOT drawdowns, annualized returns, or inflation-adjusted wealth.

## Recent five years: actual ETF data

QQQ, QLD, TQQQ, and SGOV adjusted historical prices. Primary execution is the next session OPEN, not the signal close. A next-session CLOSE sensitivity is also provided.

| Execution | Strategy | Dates | Contributed | Ending value | End ROI | Worst interim ROI | Mean deployed-dollar wait (calendar days) |
|---|---|---|---:|---:|---:|---:|---:|
| next_open | Weekly QQQ | 2021-09-16 to 2026-09-16 | $26,200 | $45,127 | 72.24% | -22.78% | N/A |
| next_open | Weekly QLD | 2021-09-16 to 2026-09-16 | $26,200 | $60,979 | 132.74% | -42.61% | N/A |
| next_open | Weekly TQQQ | 2021-09-16 to 2026-09-16 | $26,200 | $75,467 | 188.04% | -58.51% | N/A |
| next_open | Weekly 50/50, no rebalance | 2021-09-16 to 2026-09-16 | $26,200 | $60,297 | 130.14% | -40.64% | N/A |
| next_open | Wait for dip, idle cash | 2021-09-16 to 2026-09-16 | $26,200 | $74,547 | 184.53% | -57.81% | 14.2 |
| next_open | Wait for dip, idle short Treasuries | 2021-09-16 to 2026-09-16 | $26,200 | $74,594 | 184.71% | -57.83% | 14.2 |
| next_close | Weekly QQQ | 2021-09-16 to 2026-09-16 | $26,200 | $45,072 | 72.03% | -22.81% | N/A |
| next_close | Weekly QLD | 2021-09-16 to 2026-09-16 | $26,200 | $60,845 | 132.23% | -42.63% | N/A |
| next_close | Weekly TQQQ | 2021-09-16 to 2026-09-16 | $26,200 | $75,252 | 187.22% | -58.44% | N/A |
| next_close | Weekly 50/50, no rebalance | 2021-09-16 to 2026-09-16 | $26,200 | $60,162 | 129.63% | -40.63% | N/A |
| next_close | Wait for dip, idle cash | 2021-09-16 to 2026-09-16 | $26,200 | $74,525 | 184.45% | -58.07% | 14.2 |
| next_close | Wait for dip, idle short Treasuries | 2021-09-16 to 2026-09-16 | $26,200 | $74,572 | 184.63% | -58.10% | 14.2 |

## Including 2000: synthetic stress tests

These are NOT actual pre-inception fund returns. Reuse the prior daily-reset Nasdaq-100 total-return simulation with 0.20% 1x fees; 0.95% 2x/3x fees; historical fed funds + 0.50% financing on extra exposure. Funds are assumed to survive and trade continuously.
Historical QQQ supplies the signal. All strategies execute at the next session CLOSE for comparability because the synthetic funds have no reliable opening prices. Waiting assets are a yield-carry PROXY using lagged DGS3MO less 0.10% annual costs, not actual SGOV or a bond total-return index. No bond price sensitivity is modeled. Compare the zero-yield cash control to see whether this approximation drives results.
Monthly starting dates, complete five-/ten-year windows. Windows overlap and are not independent observations or probabilities. Each best/worst is selected retrospectively, not a future bound.

| Years | Strategy | Worst end ROI | Median end ROI | Best end ROI | Worst interim ROI | Wins versus weekly TQQQ | Median matched wealth ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| 5 | Weekly QQQ | -32.92% | 47.57% | 113.15% | -57.71% | 66/270 | 0.6985x |
| 5 | Weekly QLD | -67.08% | 89.67% | 280.43% | -83.58% | 76/270 | 0.8720x |
| 5 | Weekly TQQQ | -86.06% | 119.22% | 482.25% | -93.71% | 0/270 | 1.0000x |
| 5 | Weekly 50/50, no rebalance | -59.49% | 83.84% | 297.70% | -74.82% | 66/270 | 0.8492x |
| 5 | Wait for dip, idle cash | -85.68% | 115.38% | 462.81% | -93.38% | 109/270 | 0.9949x |
| 5 | Wait for dip, idle short Treasuries | -85.67% | 115.29% | 462.69% | -93.38% | 107/270 | 0.9943x |
| 10 | Weekly QQQ | -21.77% | 130.41% | 257.52% | -57.71% | 37/210 | 0.3995x |
| 10 | Weekly QLD | -60.60% | 308.28% | 919.26% | -83.58% | 50/210 | 0.7110x |
| 10 | Weekly TQQQ | -81.87% | 516.67% | 2328.99% | -93.71% | 0/210 | 1.0000x |
| 10 | Weekly 50/50, no rebalance | -51.82% | 324.58% | 1293.26% | -74.82% | 37/210 | 0.6998x |
| 10 | Wait for dip, idle cash | -81.77% | 515.29% | 2284.95% | -93.38% | 53/210 | 0.9960x |
| 10 | Wait for dip, idle short Treasuries | -81.77% | 514.93% | 2282.94% | -93.38% | 47/210 | 0.9957x |

## Interpretation limits

- A 1.5% daily loss is not a valuation, bottom, or uptrend signal. Prices may first rise more than the later dip; existing positions remain exposed throughout bear markets.
- Equal external deposits isolate entry timing. New contributions can reduce the loss percentage mechanically without recovering earlier investment losses; do not confuse that with an investment return.
- New deposits shrink relative to accumulated assets as the account grows. Idle Treasury holdings do not protect TQQQ already purchased; there is no enduring bond allocation or loss cap.
- Results assume uninterrupted deposits, no withdrawals, immediate tradability of sale proceeds, and no account-level cash settlement restrictions. Taxable bond income and sales are not modeled.
- This run uses the user-specified fixed 1.5% threshold. Comparing thresholds on the same history is exploratory, not out-of-sample validation. Previous strategy exploration and choice of Nasdaq exposure introduce selection risk; historical fit is not validation for live execution.

Sources:

- [TQQQ daily leverage objective](https://www.proshares.com/our-etfs/leveraged-and-inverse/tqqq)
- [SGOV short Treasury mandate and inception](https://www.ishares.com/us/products/314116/ishares-0-3-month-treasury-bond-etf)
- [FRED Treasury yield, NOT a total-return series](https://fred.stlouisfed.org/series/DGS3MO)
- [Nasdaq-100 total return](https://fred.stlouisfed.org/series/NASDAQXNDX)
- Actual adjusted prices use saved Yahoo historical data; source hashes are in provenance.json.
