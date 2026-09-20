# Five-year weekly DCA comparison

Actual common window: 2021-09-16 through 2026-09-16.

No initial lump sum. $100 per week. 262 purchases; total contributed $26,200 per ETF.

| ETF | Final value | Profit | Profit / deposits | XIRR | Flow-adjusted max drawdown | Sharpe (0% rf) |
|---|---:|---:|---:|---:|---:|---:|
| QQQ | $45,126.70 | $18,926.70 | 72.24% | 22.17% | -35.18% | 0.685 |
| QLD | $60,979.17 | $34,779.17 | 132.74% | 34.99% | -63.73% | 0.569 |
| TQQQ | $75,466.77 | $49,266.77 | 188.04% | 44.36% | -81.68% | 0.538 |

## What the investor experienced

Drawdown removes deposits using overnight/intraday chain linking. It is not the loss percentage on all money deposited to date.

| ETF | Worst loss / deposits | Date | Deposited then | Account then | Drawdown peak | Drawdown trough | Recovered |
|---|---:|---|---:|---:|---|---|---|
| QQQ | -22.78% | 2022-06-16 | $4,000 | $3,089 | 2021-11-19 | 2022-11-03 | 2023-12-13 |
| QLD | -42.61% | 2022-06-16 | $4,000 | $2,295 | 2021-11-19 | 2022-12-28 | 2024-05-28 |
| TQQQ | -58.51% | 2022-06-16 | $4,000 | $1,660 | 2021-11-19 | 2022-12-28 | 2024-12-04 |

## Same-window lump-sum counterfactual

This is a separate comparison, not the DCA outcome. A dollar fully invested at the first open is held to the final close.

| ETF | Lump-sum total return |
|---|---:|
| QQQ | 92.68% |
| QLD | 118.03% |
| TQQQ | 95.81% |

## Different five-year starting dates

Monthly starting dates from 2010-03-01 through 2021-09-01; 139 overlapping five-year windows.
Windows share most of their observations. These are descriptive sensitivity checks, not independent trials or forecasts.

| ETF | Highest final value (windows) | Lowest XIRR | Median XIRR | Highest XIRR |
|---|---:|---:|---:|---:|
| QQQ | 4 | 6.29% | 18.30% | 31.16% |
| QLD | 14 | 3.01% | 30.19% | 56.66% |
| TQQQ | 121 | -8.08% | 41.49% | 76.79% |

Selected January-start windows (all monthly windows are in the companion data):

| Window | Contributed | QQQ final | QLD final | TQQQ final |
|---|---:|---:|---:|---:|
| 2011-01-03 to 2015-12-31 | $26,100 | $40,379 | $59,291 | $84,975 |
| 2012-01-03 to 2016-12-30 | $26,100 | $37,116 | $49,858 | $65,956 |
| 2013-01-02 to 2017-12-29 | $26,100 | $41,670 | $62,796 | $94,064 |
| 2014-01-02 to 2018-12-31 | $26,200 | $34,938 | $41,721 | $47,412 |
| 2015-01-02 to 2019-12-31 | $26,200 | $42,047 | $59,733 | $81,092 |
| 2016-01-04 to 2020-12-31 | $26,100 | $52,950 | $87,982 | $126,309 |
| 2017-01-03 to 2021-12-31 | $26,100 | $53,914 | $93,463 | $139,556 |
| 2018-01-02 to 2022-12-30 | $26,100 | $30,525 | $28,135 | $21,295 |
| 2019-01-02 to 2023-12-29 | $26,100 | $40,544 | $49,723 | $52,336 |
| 2020-01-02 to 2024-12-31 | $26,200 | $41,827 | $52,694 | $60,079 |
| 2021-01-04 to 2025-12-31 | $26,100 | $42,545 | $53,852 | $62,494 |

## Methodology and limitations

- Buy at the first available regular-session open of each ISO calendar week. The partial first week also receives one deposit. Same dates and dollars across ETFs.
- Fractional shares; 5 bps adverse price slippage per purchase; zero commissions and taxes; no sales or terminal liquidation cost.
- Yahoo adjusted open = raw open multiplied by adjusted-close / close. Adjusted-price units are synthetic total-return units, NOT literal historical broker shares. This approximates dividend reinvestment and handles splits without creating artificial gains.
- ETF management fees and realized leverage financing/tracking effects are already embedded in historical prices; do not deduct the current expense ratio again.
- XIRR uses the actual deposit dates and terminal marked-to-market value; it is annualized. Profit / deposits is NOT annualized. Sharpe uses daily flow-adjusted returns and assumes 0% risk-free, not contemporaneous Treasury rates.
- No historical index-return multiplication or invented pre-inception TQQQ series. Common data start in February 2010, so these results omit the 2000 and 2008 crises.
- Dollar price level alone cannot inflate percentage returns: rescaling every price leaves the backtest unchanged. Starting valuation and subsequent market path can change results greatly.
- Latest fetched common complete daily bar is 2026-09-16 although the download was requested through 2026-09-17. No later price is fabricated or spliced into the adjusted series.
- Historical total returns are exploratory estimates, not a recommendation or a prediction. Continuous weekly contributions during unemployment or drawdowns are assumed.

## Independent aggregate check

Compare Yahoo adjusted-close CAGR for 2021-08-31 to 2026-08-31 with ProShares published five-year market-price annualized return as of 2026-08-31. This checks aggregates only, not every bar or DCA execution. Issuer returns use closing bid/ask midpoints and distribution assumptions may differ.

| ETF | Yahoo calculated | Issuer reported | Difference (bps/year) |
|---|---:|---:|---:|
| QLD | 17.3348% | 17.27% | +6.48 |
| TQQQ | 15.0925% | 15.12% | -2.75 |

Sources:

- [QQQ daily prices](https://finance.yahoo.com/quote/QQQ/history/)
- [QLD daily prices](https://finance.yahoo.com/quote/QLD/history/)
- [TQQQ daily prices](https://finance.yahoo.com/quote/TQQQ/history/)
- [QLD issuer objective and performance](https://www.proshares.com/our-etfs/leveraged-and-inverse/qld)
- [TQQQ issuer objective and performance](https://www.proshares.com/our-etfs/leveraged-and-inverse/tqqq)
