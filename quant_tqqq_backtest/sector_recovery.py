"""Descriptive crisis recoveries, not an executable trading strategy."""

from datetime import date
import json
from pathlib import Path

import numpy as np
import pandas as pd

from quant_tqqq_backtest.backtest import download_yahoo


ROOT = Path(__file__).resolve().parent
END = date(2026, 9, 18)
OUT = ROOT / "output" / "sector_recovery_2026-09-20"
CACHE = ROOT / "data" / "sector_recovery_2026-09-20"
GROUPS = {
    "IGV": "Software ETF (includes selected media/services)",
    "SOXX": "Semiconductor ETF (not all hardware)",
    "MSFT": "Software case study",
    "ADBE": "Software case study",
    "ORCL": "Software case study; evolving cloud infrastructure exposure",
    "AAPL": "Hardware/platform case study",
    "CSCO": "Networking case study; also sells software/services",
    "INTC": "Semiconductor case study",
    "NVDA": "Semiconductor/platform case study",
}
EPISODES = {
    "Dot-com 1999-2002": ("1999-01-01", "2002-12-31"),
    "GFC 2007-2009": ("2007-01-01", "2009-06-30"),
    "COVID 2020": ("2020-01-01", "2020-06-30"),
    "Rates 2021-2022": ("2021-01-01", "2022-12-31"),
}
OFFICIAL_MARKET_RETURNS = {
    "IGV": {2022: -35.65, 2023: 58.56},
    "SOXX": {2022: -35.09, 2023: 67.13},
}
SOURCES = {
    "IGV": "https://www.ishares.com/us/products/239771/ishares-north-american-techsoftware-etf",
    "SOXX": "https://www.ishares.com/us/products/239705/ishares-phlx-semiconductor-etf",
    "daily": "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
}


def episode_recovery(series, start, end):
    """Locate the worst peak-to-trough episode, then its first total recovery."""
    if not series.index.is_unique or not series.index.is_monotonic_increasing:
        raise ValueError("Dates must be unique and sorted.")
    if series.isna().any() or not np.isfinite(series).all() or (series <= 0).any():
        raise ValueError("Prices must be finite and positive.")
    window = series.loc[start:end]
    if len(window) < 2:
        raise ValueError("Insufficient data.")
    drawdown = window / window.cummax() - 1
    trough = drawdown.idxmin()
    peak = window.loc[:trough].idxmax()
    peak_value = series.loc[peak]
    future = series.loc[series.index > trough]
    recovered = future[future >= peak_value * (1 - 1e-10)]
    recovery = recovered.index[0] if len(recovered) else None
    return {
        "peak": str(peak.date()),
        "trough": str(trough.date()),
        "max_drawdown_pct": float(drawdown.min() * 100),
        "recovered": str(recovery.date()) if recovery is not None else None,
        "peak_to_recovery_years": (recovery - peak).days / 365.2425 if recovery is not None else None,
        "trough_to_recovery_years": (recovery - trough).days / 365.2425 if recovery is not None else None,
        "observation_end": str(series.index[-1].date()),
    }


def verify_algorithm():
    dates = pd.date_range("2020-01-01", periods=6)
    s = pd.Series([100, 120, 60, 80, 120, 30], index=dates)
    r = episode_recovery(s, "2020-01-01", "2020-01-04")
    assert r["peak"] == "2020-01-02" and r["trough"] == "2020-01-03"
    assert r["max_drawdown_pct"] == -50 and r["recovered"] == "2020-01-05"
    assert episode_recovery(s * 7, "2020-01-01", "2020-01-04") == r
    assert episode_recovery(s.iloc[:4], "2020-01-01", "2020-01-04")["recovered"] is None
    try:
        episode_recovery(s * 0, "2020-01-01", "2020-01-04")
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid prices must fail.")


def main():
    verify_algorithm()
    OUT.mkdir(parents=True, exist_ok=True)
    assets, records, annual, checks, provenance = {}, [], [], [], {}
    for symbol, group in GROUPS.items():
        frame = download_yahoo(symbol, date(1998, 1, 1), END, CACHE)
        frame = frame.loc[:str(END)]
        assets[symbol] = frame
        provenance[symbol] = {
            "requested_start": "1998-01-01", "requested_end": str(END),
            "actual_start": str(frame.index[0].date()), "actual_end": str(frame.index[-1].date()),
            "rows": len(frame), "source_url": SOURCES["daily"].format(symbol=symbol),
            "group": group,
        }
        for episode, (start, stop) in EPISODES.items():
            if symbol in ("IGV", "SOXX") and episode.startswith("Dot-com"):
                continue
            if frame.index[0] > pd.Timestamp(start) + pd.Timedelta(days=31):
                raise ValueError(f"{symbol} does not cover {episode}.")
            records.append({"symbol": symbol, "group": group, "episode": episode,
                            **episode_recovery(frame.adj_close, start, stop)})
        for year in (2008, 2009, 2020, 2022, 2023, 2024, 2025):
            previous = frame.loc[:f"{year-1}-12-31", "adj_close"]
            current = frame.loc[:f"{year}-12-31", "adj_close"]
            value = float((current.iloc[-1] / previous.iloc[-1] - 1) * 100)
            annual.append({"symbol": symbol, "year": year, "total_return_pct": value})
            if symbol in OFFICIAL_MARKET_RETURNS and year in OFFICIAL_MARKET_RETURNS[symbol]:
                reference = OFFICIAL_MARKET_RETURNS[symbol][year]
                checks.append({"symbol": symbol, "year": year, "calculated_pct": value,
                               "issuer_market_return_pct": reference,
                               "difference_percentage_points": value-reference,
                               "pass": abs(value-reference) < 0.2,
                               "source_url": SOURCES[symbol]})
        print(symbol, provenance[symbol], flush=True)
    if not all(c["pass"] for c in checks):
        raise ValueError(f"ETF issuer cross-check failed: {checks}")
    payload = {"as_of_requested": str(END), "recovery_basis": "Yahoo adjusted daily closes, dividend reinvestment proxy",
               "algorithm_checks_passed": True, "provenance": provenance,
               "episodes": EPISODES, "recoveries": records, "calendar_returns": annual,
               "issuer_crosschecks": checks, "sources": SOURCES}
    (OUT / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = ["# Software and hardware crisis recoveries", "",
             "Descriptive study dated 2026-09-20. Not a trading rule or a sector-wide statistical test.", "",
             "## Method", "",
             "- Daily adjusted-close series from Yahoo Finance: splits and dividend reinvestment proxy. Nominal USD; no investor tax, inflation, commissions or additional deposits.",
             "- Within each stated crisis window, identify the largest drawdown from the running high. Recover when a later adjusted close first regains that episode's peak. Duration starts at that peak, not the trough. Recovery does not mean it stays recovered.",
             "- These peaks/troughs are selected retrospectively for description. A trading strategy could not know them in advance. Different crisis windows may change the selected episode.",
             "- IGV and SOXX both launched in July 2001: neither is used for the complete 2000 bubble. IGV includes selected media/services; SOXX is semiconductors, NOT the whole hardware sector. Fund benchmarks and constituents evolve.",
             "- Individual stocks are a deliberately small set of surviving examples, not a representative contemporaneous universe. Acquired, delisted and failed companies are missing. Do not infer a sector-average recovery or win rate.",
             "- A fast recovery from a shallower fall need not deliver a higher return from a common purchase date. This study is not a DCA or dip-entry backtest.", "",
             "## ETF episodes", "", "| Crisis | ETF | Peak | Trough | Drawdown | Recovered | Years from peak |",
             "|---|---|---|---|---:|---|---:|"]
    for r in records:
        if r["symbol"] in ("IGV", "SOXX"):
            y = f'{r["peak_to_recovery_years"]:.2f}' if r["recovered"] else "Not recovered"
            lines.append(f'| {r["episode"]} | {r["symbol"]} | {r["peak"]} | {r["trough"]} | {r["max_drawdown_pct"]:.1f}% | {r["recovered"]} | {y} |')
    lines += ["", "## Dot-com surviving-company examples", "", "| Company | Group | Peak | Trough | Drawdown | Recovered | Years from peak |", "|---|---|---|---|---:|---|---:|"]
    for r in records:
        if r["episode"].startswith("Dot-com"):
            y = f'{r["peak_to_recovery_years"]:.2f}' if r["recovered"] else "Not recovered"
            lines.append(f'| {r["symbol"]} | {r["group"]} | {r["peak"]} | {r["trough"]} | {r["max_drawdown_pct"]:.1f}% | {r["recovered"]} | {y} |')
    lines += ["", "## ETF calendar-year total returns", "", "| Year | IGV | SOXX |", "|---|---:|---:|"]
    for year in (2008, 2009, 2020, 2022, 2023, 2024, 2025):
        values = {r["symbol"]: r["total_return_pct"] for r in annual if r["year"] == year}
        lines.append(f'| {year} | {values["IGV"]:.2f}% | {values["SOXX"]:.2f}% |')
    lines += ["", "## Validation and sources", "",
              "Algorithm checked for known recovery, no recovery, scale invariance, invalid prices, and excluding later crises from trough selection. ETF 2022 and 2023 returns checked against issuer market-price total returns, tolerance 0.20 percentage points.", ""]
    for symbol in ("IGV", "SOXX"):
        lines.append(f'- [{symbol} issuer definition and performance]({SOURCES[symbol]})')
    for symbol in GROUPS:
        lines.append(f'- [{symbol} daily-history data]({SOURCES["daily"].format(symbol=symbol)}): {provenance[symbol]["actual_start"]} through {provenance[symbol]["actual_end"]}.')
    (OUT / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"ETF": [r for r in records if r["symbol"] in ("IGV", "SOXX")],
                      "dotcom": [r for r in records if r["episode"].startswith("Dot-com")],
                      "issuer_crosschecks": checks}, indent=2))


if __name__ == "__main__":
    main()
