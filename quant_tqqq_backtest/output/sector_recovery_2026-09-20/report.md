# Software and hardware crisis recoveries

Descriptive study dated 2026-09-20. Not a trading rule or a sector-wide statistical test.

## Method

- Daily adjusted-close series from Yahoo Finance: splits and dividend reinvestment proxy. Nominal USD; no investor tax, inflation, commissions or additional deposits.
- Within each stated crisis window, identify the largest drawdown from the running high. Recover when a later adjusted close first regains that episode's peak. Duration starts at that peak, not the trough. Recovery does not mean it stays recovered.
- These peaks/troughs are selected retrospectively for description. A trading strategy could not know them in advance. Different crisis windows may change the selected episode.
- IGV and SOXX both launched in July 2001: neither is used for the complete 2000 bubble. IGV includes selected media/services; SOXX is semiconductors, NOT the whole hardware sector. Fund benchmarks and constituents evolve.
- Individual stocks are a deliberately small set of surviving examples, not a representative contemporaneous universe. Acquired, delisted and failed companies are missing. Do not infer a sector-average recovery or win rate.
- A fast recovery from a shallower fall need not deliver a higher return from a common purchase date. This study is not a DCA or dip-entry backtest.

## ETF episodes

| Crisis | ETF | Peak | Trough | Drawdown | Recovered | Years from peak |
|---|---|---|---|---:|---|---:|
| GFC 2007-2009 | IGV | 2007-10-31 | 2008-11-20 | -49.8% | 2010-10-26 | 2.99 |
| COVID 2020 | IGV | 2020-02-19 | 2020-03-16 | -30.3% | 2020-05-29 | 0.27 |
| Rates 2021-2022 | IGV | 2021-11-09 | 2022-11-04 | -45.9% | 2024-09-24 | 2.87 |
| GFC 2007-2009 | SOXX | 2007-07-17 | 2008-11-20 | -66.8% | 2013-09-18 | 6.17 |
| COVID 2020 | SOXX | 2020-02-19 | 2020-03-20 | -34.3% | 2020-06-05 | 0.29 |
| Rates 2021-2022 | SOXX | 2021-12-27 | 2022-10-14 | -45.8% | 2023-12-13 | 1.96 |

## Dot-com surviving-company examples

| Company | Group | Peak | Trough | Drawdown | Recovered | Years from peak |
|---|---|---|---|---:|---|---:|
| MSFT | Software case study | 1999-12-27 | 2000-12-20 | -65.2% | 2014-07-16 | 14.55 |
| ADBE | Software case study | 2000-11-06 | 2002-08-05 | -79.9% | 2006-11-15 | 6.02 |
| ORCL | Software case study; evolving cloud infrastructure exposure | 2000-09-01 | 2002-06-03 | -84.2% | 2014-12-18 | 14.29 |
| AAPL | Hardware/platform case study | 2000-03-22 | 2002-10-09 | -81.1% | 2005-01-26 | 4.85 |
| CSCO | Networking case study; also sells software/services | 2000-03-27 | 2002-10-08 | -89.3% | 2021-08-24 | 21.41 |
| INTC | Semiconductor case study | 2000-08-31 | 2002-10-08 | -82.2% | 2018-03-06 | 17.51 |
| NVDA | Semiconductor/platform case study | 2002-01-03 | 2002-10-09 | -89.7% | 2006-11-13 | 4.86 |

## ETF calendar-year total returns

| Year | IGV | SOXX |
|---|---:|---:|
| 2008 | -38.43% | -51.71% |
| 2009 | 47.53% | 74.94% |
| 2020 | 52.86% | 52.72% |
| 2022 | -35.65% | -35.09% |
| 2023 | 58.56% | 67.12% |
| 2024 | 23.41% | 12.92% |
| 2025 | 5.56% | 40.74% |

## Validation and sources

Algorithm checked for known recovery, no recovery, scale invariance, invalid prices, and excluding later crises from trough selection. ETF 2022 and 2023 returns checked against issuer market-price total returns, tolerance 0.20 percentage points.

- [IGV issuer definition and performance](https://www.ishares.com/us/products/239771/ishares-north-american-techsoftware-etf)
- [SOXX issuer definition and performance](https://www.ishares.com/us/products/239705/ishares-phlx-semiconductor-etf)
- [IGV daily-history data](https://query1.finance.yahoo.com/v8/finance/chart/IGV): 2001-07-17 through 2026-09-18.
- [SOXX daily-history data](https://query1.finance.yahoo.com/v8/finance/chart/SOXX): 2001-07-13 through 2026-09-18.
- [MSFT daily-history data](https://query1.finance.yahoo.com/v8/finance/chart/MSFT): 1998-01-02 through 2026-09-18.
- [ADBE daily-history data](https://query1.finance.yahoo.com/v8/finance/chart/ADBE): 1998-01-02 through 2026-09-18.
- [ORCL daily-history data](https://query1.finance.yahoo.com/v8/finance/chart/ORCL): 1998-01-02 through 2026-09-18.
- [AAPL daily-history data](https://query1.finance.yahoo.com/v8/finance/chart/AAPL): 1998-01-02 through 2026-09-18.
- [CSCO daily-history data](https://query1.finance.yahoo.com/v8/finance/chart/CSCO): 1998-01-02 through 2026-09-18.
- [INTC daily-history data](https://query1.finance.yahoo.com/v8/finance/chart/INTC): 1998-01-02 through 2026-09-18.
- [NVDA daily-history data](https://query1.finance.yahoo.com/v8/finance/chart/NVDA): 1999-01-22 through 2026-09-18.
