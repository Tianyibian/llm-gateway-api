# Fixed-$500 strategy experiments

Research run: 2026-09-17. No brokerage orders placed.

## Common sample

2010-02-11 to 2026-09-16. Initial $500, no subsequent deposits.

| Strategy | CAGR | Max drawdown | Sharpe (0% rf) | 2020 return | 2022 return |
|---|---:|---:|---:|---:|---:|
| QQQ hold | 19.37% | -35.12% | 0.96 | 48.41% | -32.58% |
| QLD hold | 32.67% | -63.68% | 0.90 | 88.90% | -60.52% |
| TQQQ hold | 42.35% | -81.66% | 0.89 | 110.05% | -79.09% |
| 70% TQQQ trend / QQQ | 31.81% | -52.36% | 0.89 | 90.76% | -44.64% |
| QLD trend / cash | 23.11% | -46.02% | 0.82 | 59.89% | -24.75% |
| QLD trend + volatility / cash | 14.70% | -33.87% | 0.76 | 10.03% | -16.69% |

## Rules and methodology

- QLD trend: QQQ adjusted close >= 1.04 x SMA200 enables 100% QLD; <= 0.97 x SMA200 switches to cash. Inside the band, preserve state.
- QLD volatility variant: same trend filter; in bull periods target QLD weight = min(75%, 25% / (2 x QQQ trailing 63-session annualized volatility)); rest is cash. This is only an approximate volatility target, not a risk guarantee.
- Prior 70% TQQQ strategy: bull 70% TQQQ + 30% QQQ; bear 100% QQQ; thresholds 1.02/0.98. Different thresholds are explicitly retained, not retuned.
- Targets use completed prior-session signals, including first entry. Trade at next adjusted open on month changes or switches into/out of an asset; otherwise hold.
- 5 bps one-way adverse slippage, fractional shares, zero commissions, no taxes, cash interest 0%. Expense drag is embedded in ETF market-price history. Adjusted prices approximate reinvested dividends.
- Drawdown includes initial capital and costs, is measured at daily closes, and is not a future loss limit. Constant holdings have no external borrowing.
- Cached Yahoo daily histories; no independent full-history reconciliation. No fabricated pre-inception TQQQ prices.
- These are exploratory, previously observed periods, not untouched out-of-sample tests. No rule has been proved optimal. No single-stock backtest was used to justify today's stock selection.

## Period checks

| Dates | Strategy | CAGR | Max drawdown |
|---|---|---:|---:|
| 2006-06-21 to 2026-09-16 | QQQ hold | 16.39% | -53.40% |
| 2006-06-21 to 2026-09-16 | QLD hold | 24.73% | -83.13% |
| 2006-06-21 to 2026-09-16 | QLD trend / cash | 20.70% | -46.02% |
| 2006-06-21 to 2026-09-16 | QLD trend + volatility / cash | 13.56% | -33.87% |
| 2007-01-03 to 2009-12-31 | QQQ hold | 2.12% | -53.40% |
| 2007-01-03 to 2009-12-31 | QLD hold | -8.86% | -83.13% |
| 2007-01-03 to 2009-12-31 | QLD trend / cash | 14.25% | -41.78% |
| 2007-01-03 to 2009-12-31 | QLD trend + volatility / cash | 11.59% | -25.59% |
| 2018-01-02 to 2026-09-16 | QQQ hold | 19.60% | -35.12% |
| 2018-01-02 to 2026-09-16 | QLD hold | 29.55% | -63.68% |
| 2018-01-02 to 2026-09-16 | TQQQ hold | 33.23% | -81.66% |
| 2018-01-02 to 2026-09-16 | 70% TQQQ trend / QQQ | 33.10% | -52.36% |
| 2018-01-02 to 2026-09-16 | QLD trend / cash | 26.48% | -46.02% |
| 2018-01-02 to 2026-09-16 | QLD trend + volatility / cash | 14.23% | -33.87% |
| 2022-01-03 to 2026-09-16 | QQQ hold | 13.50% | -34.83% |
| 2022-01-03 to 2026-09-16 | QLD hold | 15.42% | -63.08% |
| 2022-01-03 to 2026-09-16 | TQQQ hold | 11.92% | -81.02% |
| 2022-01-03 to 2026-09-16 | 70% TQQQ trend / QQQ | 21.22% | -47.31% |
| 2022-01-03 to 2026-09-16 | QLD trend / cash | 16.75% | -34.75% |
| 2022-01-03 to 2026-09-16 | QLD trend + volatility / cash | 11.03% | -22.13% |

## Higher execution costs

| Strategy | One-way bps | CAGR |
|---|---:|---:|
| 70% TQQQ trend / QQQ | 5 | 31.81% |
| QLD trend / cash | 5 | 23.11% |
| QLD trend + volatility / cash | 5 | 14.70% |
| 70% TQQQ trend / QQQ | 25 | 30.94% |
| QLD trend / cash | 25 | 22.77% |
| QLD trend + volatility / cash | 25 | 14.44% |

## Prior-report corrections

The prior smart_dca simulation used same-day information for its first entry and treated opening deposits as ending deposits when calculating time-weighted returns. Both were corrected in smart_dca.py and its report regenerated. Previously quoted metrics should be treated as superseded exploratory estimates.
