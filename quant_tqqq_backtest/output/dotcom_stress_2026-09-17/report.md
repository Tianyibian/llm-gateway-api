# Dot-com crisis: synthetic daily 1x / 2x / 3x Nasdaq-100 DCA

IMPORTANT: counterfactual simulations, NOT actual historical QLD/TQQQ returns before inception.

Index data: 1999-03-04 to 2026-09-16. Stress start: 2000-03-27, the retrospectively selected 2000 closing peak. Post-crash trough anchor: 2002-10-07.

## Fixed assumptions

- $100 each ISO week's first trading day at the CLOSE, including the first partial week. No initial lump sum. This differs from the previous real-ETF test's opening executions; figures must not be spliced together.
- Deterministic schedule: new money receives no earlier same-day gain. Fractional synthetic units; 5 bps purchase slippage; no taxes or final liquidation cost. No tactical switches, stops or hindsight trading signals.
- Benchmark is the Nasdaq-100 TOTAL RETURN index (NASDAQXNDX), including dividends, not the Nasdaq Composite. FRED distributes Nasdaq data; any index back-history is not a tradable fund record.
- Daily factor = 1 + leverage * daily_index_return - financing_cost - fund_fee. Compound each day; NEVER multiply a five-year cumulative return by 2 or 3.
- Ideal: zero fund/financing costs, still with purchase slippage. Illustrative cost: annual fees 0.20% for 1x, 0.95% for 2x/3x; borrowed notional (L-1) charged historical effective federal funds rate + 0.50% spread. Higher financing: spread 1.50%. These are declared scenarios, not calibrated replicas or actual historical swap contracts.
- Accrue financing on calendar days using prior-day rates, actual/365; include weekends and holidays. Daily rebalancing occurs at closes. Financing, rebalancing, and intraday exposures of actual ETFs may differ.
- Assume the hypothetical fund survives, remains liquid, permits ongoing investment, and executes daily leverage throughout. Fund closure, derivatives counterparty failures, changing regulations, taxes, trading halts, and intraday path-dependent liquidation are NOT modeled. A nonpositive daily modeled NAV factor raises an error rather than resurrecting capital.
- Breakeven is nominal account value >= cumulative cash contributed. It is NOT inflation-adjusted, risk-adjusted, or equivalent to outperforming 1x. No withdrawals or contribution interruptions are assumed.

## Starting at the 2000 peak

| Cost scenario | Years | Deposited | 1x final | 2x final | 3x final |
|---|---:|---:|---:|---:|---:|
| ideal_no_cost | 5 | $26,100 | $26,059 | $24,925 | $23,702 |
| ideal_no_cost | 10 | $52,200 | $66,965 | $65,310 | $51,816 |
| illustrative_cost | 5 | $26,100 | $25,942 | $23,493 | $21,666 |
| illustrative_cost | 10 | $52,200 | $66,296 | $54,839 | $40,645 |
| higher_financing | 5 | $26,100 | $25,942 | $23,033 | $20,892 |
| higher_financing | 10 | $52,200 | $66,296 | $52,500 | $37,990 |

## Recovery definitions

First recovery is searched AFTER the 2002 market trough, to avoid mistaking a brief early bounce for surviving the crisis. Last recovery is retrospective: the date after the final observed loss versus deposits, through the sample end, NOT a guarantee of future safety.

| Scenario | Leverage | DCA first recovery after trough | Last observed DCA recovery | First dollar recovery after trough | First dollar value at sample end | Worst DCA loss / deposits | Max NAV drawdown |
|---|---:|---|---|---|---:|---:|---:|
| ideal_no_cost | 1x | 2004-01-05 | 2009-07-16 | 2014-11-26 | $7.50922 | -53.15% | -82.873% |
| ideal_no_cost | 2x | 2004-01-05 | 2010-07-07 | 2019-12-16 | $8.32974 | -79.25% | -98.565% |
| ideal_no_cost | 3x | 2004-01-05 | 2011-10-04 | 2025-10-27 | $1.32834 | -90.14% | -99.962% |
| illustrative_cost | 1x | 2004-01-05 | 2009-07-20 | 2015-02-20 | $7.12173 | -53.19% | -82.960% |
| illustrative_cost | 2x | 2004-01-08 | 2010-09-15 | 2020-09-02 | $3.31864 | -79.74% | -98.898% |
| illustrative_cost | 3x | 2004-01-06 | 2012-01-03 | Not recovered | $0.27038 | -90.41% | -99.982% |
| higher_financing | 1x | 2004-01-05 | 2009-07-20 | 2015-02-20 | $7.12173 | -53.19% | -82.960% |
| higher_financing | 2x | 2004-01-08 | 2010-10-05 | 2021-06-14 | $2.54580 | -79.81% | -98.992% |
| higher_financing | 3x | 2004-01-07 | 2012-01-11 | Not recovered | $0.15899 | -90.45% | -99.985% |

## Rolling monthly-start windows

Five-year and ten-year windows overlap heavily. Counts describe the historical sample; they are not independent observations or future probabilities. The sample includes two major crises but not every possible future market path.

| Scenario | Years | Leverage | Windows | Highest final value | Below deposits at end | Worst loss / deposits | Worst dates |
|---|---:|---:|---:|---:|---:|---:|---|
| higher_financing | 5 | 1x | 270 | 63 | 21 | -32.92% | 2003-12-01 to 2008-12-01 |
| higher_financing | 5 | 2x | 270 | 29 | 36 | -67.89% | 2003-12-01 to 2008-12-01 |
| higher_financing | 5 | 3x | 270 | 178 | 47 | -86.67% | 2003-12-01 to 2008-12-01 |
| higher_financing | 10 | 1x | 210 | 36 | 4 | -21.77% | 1999-04-01 to 2009-04-01 |
| higher_financing | 10 | 2x | 210 | 15 | 16 | -62.16% | 1999-04-01 to 2009-04-01 |
| higher_financing | 10 | 3x | 210 | 159 | 23 | -82.99% | 1999-04-01 to 2009-04-01 |
| ideal_no_cost | 5 | 1x | 270 | 35 | 19 | -32.57% | 2003-12-01 to 2008-12-01 |
| ideal_no_cost | 5 | 2x | 270 | 19 | 26 | -62.44% | 2003-12-01 to 2008-12-01 |
| ideal_no_cost | 5 | 3x | 270 | 216 | 35 | -82.60% | 2003-12-01 to 2008-12-01 |
| ideal_no_cost | 10 | 1x | 210 | 18 | 4 | -21.02% | 1999-04-01 to 2009-04-01 |
| ideal_no_cost | 10 | 2x | 210 | 24 | 9 | -52.56% | 1999-04-01 to 2009-04-01 |
| ideal_no_cost | 10 | 3x | 210 | 168 | 16 | -76.13% | 1999-04-01 to 2009-04-01 |
| illustrative_cost | 5 | 1x | 270 | 56 | 21 | -32.92% | 2003-12-01 to 2008-12-01 |
| illustrative_cost | 5 | 2x | 270 | 20 | 32 | -67.08% | 2003-12-01 to 2008-12-01 |
| illustrative_cost | 5 | 3x | 270 | 194 | 42 | -86.06% | 2003-12-01 to 2008-12-01 |
| illustrative_cost | 10 | 1x | 210 | 30 | 4 | -21.77% | 1999-04-01 to 2009-04-01 |
| illustrative_cost | 10 | 2x | 210 | 20 | 16 | -60.60% | 1999-04-01 to 2009-04-01 |
| illustrative_cost | 10 | 3x | 210 | 160 | 20 | -81.87% | 1999-04-01 to 2009-04-01 |

## Model compared with real funds after 2010

Not fitted to the fund results. Comparison is close-to-close with reinvested distributions, no contribution schedule; differences quantify model limitations and can materially affect long-horizon recovery dates.

| Scenario | ETF | Actual CAGR | Model CAGR | Daily return correlation |
|---|---|---:|---:|---:|
| ideal_no_cost | QQQ | 18.70% | 18.95% | 0.99867 |
| ideal_no_cost | QLD | 31.21% | 35.44% | 0.99904 |
| ideal_no_cost | TQQQ | 41.84% | 49.62% | 0.99863 |
| illustrative_cost | QQQ | 18.70% | 18.71% | 0.99867 |
| illustrative_cost | QLD | 31.21% | 31.49% | 0.99903 |
| illustrative_cost | TQQQ | 41.84% | 42.34% | 0.99861 |
| higher_financing | QQQ | 18.70% | 18.71% | 0.99867 |
| higher_financing | QLD | 31.21% | 30.18% | 0.99902 |
| higher_financing | TQQQ | 41.84% | 39.52% | 0.99860 |

## Interpretation

3x is helped by strong sustained growth, lower volatility and financing costs, and a large recovery after early contributions acquired cheap units. Large late drawdowns hit a much larger accumulated balance; five or ten years is not a guaranteed recovery horizon.
DCA account recovery can coexist with the first deposit remaining almost worthless. Later money bought at lower prices is earning the recovery; the original purchase has not magically healed.

## Sources

- [Nasdaq-100 total return via FRED](https://fred.stlouisfed.org/series/NASDAQXNDX)
- [Historical effective federal funds rate](https://fred.stlouisfed.org/series/DFF)
- [Nasdaq index description and back-history caution](https://indexes.nasdaq.com/Index/Overview/XNDX)
- [TQQQ daily objective and risk](https://www.proshares.com/our-etfs/leveraged-and-inverse/tqqq)
- [QLD daily objective and risk](https://www.proshares.com/our-etfs/leveraged-and-inverse/qld)
