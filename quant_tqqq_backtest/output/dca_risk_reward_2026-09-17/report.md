# DCA downside, upside, and contribution sizing

Uses the prior Nasdaq-100 synthetic daily-reset model, 1999-03-04 to 2026-09-16, illustrative costs. This is not actual pre-inception TQQQ/QLD history.
All loss percentages below are relative to cumulative nominal contributions, NOT drawdowns from a previous portfolio peak.

## Conditional downside

Conditional median means only the losing windows; it is not the median outcome of all windows.

| Years | Windows | Losing windows | Median loss among losers | Worst ending loss | Ending losses over 30% | Worst interim loss across windows |
|---|---:|---:|---:|---:|---:|---:|
| 5 | 270 | 42 | -24.60% | -86.06% | 17 | -93.71% |
| 10 | 210 | 20 | -42.74% | -81.87% | 16 | -93.71% |

## Paired terminal wealth ratios

Compare final account values within the SAME window with identical $100/week deposits. These are wealth ratios, not profit ratios or annualized returns.

| Years | 3x relative to | Minimum | Median | 90th percentile | Maximum |
|---|---|---:|---:|---:|---:|
| 5 | 1x | 0.21x | 1.43x | 2.38x | 2.73x |
| 5 | 2x | 0.42x | 1.15x | 1.50x | 1.59x |
| 10 | 1x | 0.23x | 2.50x | 5.49x | 6.90x |
| 10 | 2x | 0.46x | 1.41x | 2.18x | 2.56x |

## Allocation comparison

QQQ/QLD/TQQQ labels refer to synthetic 1x/2x/3x here. Except the monthly-rebalance strategy, weights apply only to NEW CONTRIBUTIONS; market weights drift. The monthly strategy restores 50/50 at the first trading day's close each month, splits other weekly deposits 50/50, and charges 5 bps on all buys AND sells. Cash earns 0%, has no nominal loss, and is kept separate. Except 50_TQQQ_only ($50/week), all rows save/invest $100/week in total.

| Years | Allocation | Worst end ROI | Median end ROI | Best end ROI | Worst interim loss | Worst ending contributed | Worst ending value |
|---|---|---:|---:|---:|---:|---:|---:|
| 5 | 100_QLD | -67.08% | 89.67% | 280.43% | -83.58% | $26,200 | $8,626 |
| 5 | 100_QQQ | -32.92% | 47.57% | 113.15% | -57.71% | $26,200 | $17,574 |
| 5 | 100_TQQQ | -86.06% | 119.22% | 482.25% | -93.71% | $26,200 | $3,653 |
| 5 | 30_TQQQ_70_cash | -25.82% | 35.76% | 144.67% | -28.11% | $26,200 | $19,436 |
| 5 | 50_50_monthly_rebalance | -66.62% | 90.46% | 282.71% | -83.34% | $26,200 | $8,746 |
| 5 | 50_TQQQ_50_QQQ | -59.49% | 83.84% | 297.70% | -74.82% | $26,200 | $10,613 |
| 5 | 50_TQQQ_50_cash | -43.03% | 59.61% | 241.12% | -46.85% | $26,200 | $14,927 |
| 5 | 50_TQQQ_only | -86.06% | 119.22% | 482.25% | -93.71% | $13,100 | $1,827 |
| 10 | 100_QLD | -60.60% | 308.28% | 919.26% | -83.58% | $52,300 | $20,608 |
| 10 | 100_QQQ | -21.77% | 130.41% | 257.52% | -57.71% | $52,300 | $40,913 |
| 10 | 100_TQQQ | -81.87% | 516.67% | 2328.99% | -93.71% | $52,300 | $9,481 |
| 10 | 30_TQQQ_70_cash | -24.56% | 155.00% | 698.70% | -28.11% | $52,300 | $39,454 |
| 10 | 50_50_monthly_rebalance | -60.32% | 313.16% | 934.99% | -83.34% | $52,300 | $20,755 |
| 10 | 50_TQQQ_50_QQQ | -51.82% | 324.58% | 1293.26% | -74.82% | $52,300 | $25,197 |
| 10 | 50_TQQQ_50_cash | -40.94% | 258.33% | 1164.50% | -46.85% | $52,300 | $30,891 |
| 10 | 50_TQQQ_only | -81.87% | 516.67% | 2328.99% | -93.71% | $26,150 | $4,741 |

## Mixed portfolios versus the benchmarks

Median of matched-window final-wealth ratios, not a ratio of unpaired medians and not a forecast.

| Years | Mix | Benchmark | Median wealth ratio | Mix higher (windows) | All windows |
|---|---|---|---:|---:|---:|
| 5 | 50_TQQQ_50_QQQ | 100_QQQ | 1.216x | 204 | 270 |
| 5 | 50_TQQQ_50_QQQ | 100_QLD | 0.997x | 128 | 270 |
| 5 | 50_50_monthly_rebalance | 100_QQQ | 1.266x | 212 | 270 |
| 5 | 50_50_monthly_rebalance | 100_QLD | 1.004x | 194 | 270 |
| 10 | 50_TQQQ_50_QQQ | 100_QQQ | 1.751x | 173 | 210 |
| 10 | 50_TQQQ_50_QQQ | 100_QLD | 1.007x | 109 | 210 |
| 10 | 50_50_monthly_rebalance | 100_QQQ | 1.790x | 183 | 210 |
| 10 | 50_50_monthly_rebalance | 100_QLD | 1.013x | 198 | 210 |

## Strong upside examples, checked using real ETF data

Dates selected retrospectively as the synthetic 3x's highest-return windows, then recalculated using REAL split/dividend-adjusted ETF prices at weekly closes. These are deliberately favorable examples, not representative forecasts. $100/week, 5 bps buying slippage, no taxes.

| Years | Dates | Contributed | QQQ ending | QLD ending | TQQQ ending | 50/50 deposits ending | Monthly 50/50 ending |
|---|---|---:|---:|---:|---:|---:|---:|
| 5 | 2016-09-01 to 2021-09-01 | $26,200 | $55,770 | $98,804 | $150,372 | $103,071 | $99,521 |
| 10 | 2011-09-01 to 2021-09-01 | $52,300 | $186,556 | $526,162 | $1,248,659 | $717,608 | $536,044 |

## Interpretation

- Halving a pure TQQQ contribution halves dollars invested, terminal dollars, profits, and dollar losses. Its percentage return and percentage losses are unchanged.
- Do not count the uninvested $50 as a protective cash sleeve unless it is actually retained and included in the same portfolio denominator.
- A 50/50 contribution split is NOT a 50% portfolio-weight cap. Gains can make the leveraged sleeve dominate later. The separate monthly reset mitigates drift but does not bound losses. Tax costs of selling are omitted, so taxable-account results may be worse.
- At a 50/50 market-value split the approximate daily index exposure is 0.5 * 1 + 0.5 * 3 = 2x. This is not half the risk of a Nasdaq allocation; compare it directly with QLD. Monthly 50/50 is not mechanically identical to a daily-reset 2x fund.
- $30 TQQQ plus $70 cash is an illustrative accounting case, not a recommendation. With ring-fenced cash, no borrowing/transfers, and fund losses limited to invested capital, losing the entire TQQQ sleeve loses 30% of nominal cumulative contributions; portfolio peak drawdowns can still exceed 30%, and cash has inflation/credit risks outside this idealization.
- All windows overlap; historical counts, quantiles, and maxima are not estimated probabilities or future loss bounds. Model assumes fund survival and continuous trading/contributions. Retirement suitability and current portfolio exposures are not established by this exercise.
- Model funding and fees: 1x 0.20%/year; 2x and 3x 0.95%/year plus historical federal funds rate + 0.50% on additional borrowed exposure. Daily reset, dividends included; no tax, inflation, or withdrawal modeling.

Sources and prior methodology:

- [Nasdaq total return via FRED](https://fred.stlouisfed.org/series/NASDAQXNDX)
- [Historical financing benchmark](https://fred.stlouisfed.org/series/DFF)
- [TQQQ issuer daily objective and risks](https://www.proshares.com/our-etfs/leveraged-and-inverse/tqqq)
- Real fund checks use the saved Yahoo adjusted-price data from the previous five-year comparison.
