# Smarter weekly contribution comparison

Period: 2010-02-11 through 2026-09-16

Assumptions: $10,000 initial contribution, $100 each week, monthly rebalancing, 5 bps adverse slippage, fractional shares, no commissions or taxes.
Signals strictly precede each execution, including initial entry. Returns treat deposits as opening flows, chaining overnight and intraday returns separately; initial trading costs are included. Sharpe assumes a zero risk-free rate.

| Strategy | Final value | XIRR | TWR CAGR | Max drawdown | Volatility | Sharpe | Avg TQQQ weight |
|---|---:|---:|---:|---:|---:|---:|---:|
| QQQ only | $718,098 | 19.38% | 19.36% | -35.12% | 20.65% | 0.96 | 0.00% |
| 90% QQQ / 10% TQQQ | $978,078 | 22.18% | 22.32% | -41.75% | 24.53% | 0.95 | 10.14% |
| 80% QQQ / 20% TQQQ | $1,313,630 | 24.86% | 25.12% | -47.94% | 28.42% | 0.93 | 20.17% |
| 70% QQQ / 30% TQQQ | $1,741,638 | 27.42% | 27.80% | -53.64% | 32.34% | 0.92 | 30.18% |
| Trend 0% or 30% TQQQ | $1,415,297 | 25.53% | 25.26% | -40.98% | 28.54% | 0.93 | 25.23% |
| Volatility target 0-30% TQQQ | $1,124,319 | 23.45% | 23.73% | -40.98% | 26.92% | 0.93 | 21.52% |
| TQQQ only | $7,795,043 | 41.30% | 42.36% | -81.66% | 61.13% | 0.89 | 100.00% |

The trend strategy holds 30% TQQQ only in the buffered QQQ bull regime. The volatility-target strategy uses the same regime and scales TQQQ from 0% to 30% to target approximately 30% annualized portfolio volatility.
