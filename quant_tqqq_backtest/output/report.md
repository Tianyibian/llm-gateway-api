# QQQ/TQQQ regime strategy backtest

Period: 2010-02-11 through 2026-09-16

## Assumptions

- Bull: QQQ adjusted close >= 1.04 x its 200-day SMA.
- Bear: QQQ adjusted close <= 0.97 x its 200-day SMA.
- The prior regime persists inside the buffer band.
- In a bull regime, a QQQ close-to-close return <= -1.00% invests all available cash in TQQQ.
- A bear-to-bull transition invests all available cash; a bull-to-bear transition sells all TQQQ.
- Signals use confirmed closes and execute at the next trading day's adjusted open.
- Each execution includes 5.0 bps of adverse slippage; fractional shares and zero commissions are assumed.
- Adjusted Yahoo Finance prices are used as an exploratory data source and approximate dividend reinvestment.

## $10,000 lump-sum comparison

| Portfolio | Final value | Total return | CAGR | Max drawdown | Volatility | Sharpe (0% rf) | Trades | Exposure |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Strategy (initial bull entry) | $938,027 | 9,280.27% | 31.48% | -62.09% | 47.12% | 0.81 | 23 | 80.50% |
| Strategy (strict triggers) | $840,740 | 8,307.40% | 30.61% | -62.09% | 47.09% | 0.81 | 23 | 80.31% |
| Buy & hold TQQQ | $3,511,040 | 35,010.40% | 42.36% | -81.66% | 61.12% | 0.88 | 1 | 100.00% |
| Buy & hold QQQ | $188,801 | 1,788.01% | 19.37% | -35.12% | 20.64% | 0.96 | 1 | 100.00% |

## $100 weekly contribution comparison

| Portfolio | Contributions | Final value | Profit | Money-weighted annual return | Trades | Exposure |
|---|---:|---:|---:|---:|---:|---:|
| Strategy (initial bull entry) | $86,700 | $2,047,974 | $1,961,274 | 32.89% | 327 | 80.50% |
| Strategy (strict triggers) | $86,700 | $2,047,001 | $1,960,301 | 32.89% | 326 | 80.31% |
| Weekly TQQQ | $86,700 | $4,284,003 | $4,197,303 | 40.36% | 867 | 100.00% |
| Weekly QQQ | $86,700 | $529,297 | $442,597 | 19.38% | 867 | 100.00% |

## Risk and robustness checks

- Worst strategy drawdown ran from 2020-02-19 to 2020-03-12 and reached -62.09%.
- The prior peak was recovered on 2020-08-20.
- Across 27 neighboring parameter combinations, lump-sum CAGR ranged from 27.99% to 31.52%.
- Across those combinations, maximum drawdown ranged from -62.09% to -53.21%.
- Weekly-contribution money-weighted returns ranged from 28.73% to 34.58%.

## Latest signal in the downloaded data

- Date: 2026-09-16
- QQQ adjusted close: $704.72
- 200-day SMA: $660.74
- Bull threshold: $687.17
- Bear threshold: $640.92
- Regime: bull
- Daily return: 0.03%

## Important limitations

- This is a historical simulation, not a forecast or investment recommendation.
- Results are sensitive to the data vendor, adjustment method, execution time, spread, slippage, taxes, and the launch-day initialization rule.
- TQQQ targets 3x the Nasdaq-100's daily return. Daily reset and compounding can make long-horizon results diverge substantially from 3x QQQ.
- Autonomous execution should not be enabled until signals, duplicate-order protection, stale-data handling, and pre-trade review have been tested in observation mode.
