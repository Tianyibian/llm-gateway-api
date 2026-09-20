"""Compare identical weekly contributions into QQQ, QLD, and TQQQ."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, PercentFormatter

from quant_tqqq_backtest.backtest import dca_metrics, simulate_buy_hold


SYMBOLS = ("QQQ", "QLD", "TQQQ")
COLORS = {"QQQ": "#2563a6", "QLD": "#16836b", "TQQQ": "#cd454c"}


def simulate_weekly(asset: pd.DataFrame, weekly: float = 100.0,
                    slippage_bps: float = 5.0) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    prices = asset[["adj_open", "adj_close"]]
    if len(asset) < 2 or not asset.index.is_unique or not asset.index.is_monotonic_increasing:
        raise ValueError("At least two unique, sorted trading dates are required.")
    if not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any().any():
        raise ValueError("Prices must be finite and positive.")
    if not np.isfinite([weekly, slippage_bps]).all() or weekly <= 0 or not 0 <= slippage_bps < 10000:
        raise ValueError("Invalid contribution or trading cost.")

    result = simulate_buy_hold(asset, 0.0, weekly, slippage_bps, "weekly_dca")
    history = pd.DataFrame({"equity": result.equity, "deposit": result.contributions})
    history["deposited"] = history.deposit.cumsum()
    bought = result.trades.groupby("execution_date").shares.sum().reindex(asset.index, fill_value=0.0)
    prior_units = bought.cumsum().shift(1, fill_value=0.0)
    opening_before_deposit = prior_units * prices.adj_open
    opening_after_deposit = opening_before_deposit + history.deposit
    # Split each day at the opening cash flow so deposits cannot mask losses.
    overnight = opening_before_deposit / history.equity.shift(1)
    overnight.iloc[0] = 1.0
    history["daily_return"] = overnight * (history.equity / opening_after_deposit) - 1.0
    history["unit_wealth"] = (1.0 + history.daily_return).cumprod()
    history["drawdown"] = history.unit_wealth / history.unit_wealth.cummax().clip(lower=1.0) - 1.0
    history["profit"] = history.equity - history.deposited
    history["profit_on_deposited"] = history.profit / history.deposited

    metrics = dca_metrics(result)
    years = (asset.index[-1] - asset.index[0]).days / 365.2425
    std = float(history.daily_return.std())
    trough = history.drawdown.idxmin()
    peak = history.unit_wealth.loc[:trough].idxmax()
    recoveries = history.unit_wealth.loc[trough:]
    recoveries = recoveries[recoveries >= history.unit_wealth.loc[peak]]
    worst_cost = history.profit_on_deposited.idxmin()
    metrics.update({
        "start": str(asset.index[0].date()), "end": str(asset.index[-1].date()),
        "profit_on_contributed": metrics["profit"] / metrics["contributed"],
        "twr_cagr": float(history.unit_wealth.iloc[-1] ** (1.0 / years) - 1),
        "max_drawdown": float(history.drawdown.min()),
        "drawdown_peak": str(peak.date()), "drawdown_trough": str(trough.date()),
        "drawdown_recovered": str(recoveries.index[0].date()) if len(recoveries) else None,
        "sharpe_0rf": float(history.daily_return.mean() / std * np.sqrt(252)) if std > 1e-12 else None,
        "volatility": std * np.sqrt(252),
        "worst_loss_on_contributed": float(history.profit_on_deposited.min()),
        "worst_loss_date": str(worst_cost.date()),
        "worst_loss_deposited": float(history.at[worst_cost, "deposited"]),
        "worst_loss_value": float(history.at[worst_cost, "equity"]),
        "lump_sum_return": float(prices.adj_close.iloc[-1] / (prices.adj_open.iloc[0] * (1 + slippage_bps / 10000)) - 1),
    })
    return history, metrics, result.trades


def aligned_window(assets: dict[str, pd.DataFrame], start: pd.Timestamp,
                   end: pd.Timestamp) -> dict[str, pd.DataFrame]:
    frames = {s: f.loc[start:end] for s, f in assets.items()}
    reference = frames[SYMBOLS[0]].index
    if any(not reference.equals(f.index) for f in frames.values()):
        raise ValueError("Inconsistent trading calendars: do not silently discard missing bars.")
    return frames


def run_windows(assets: dict[str, pd.DataFrame], starts: pd.DatetimeIndex,
                weekly: float, slippage_bps: float) -> pd.DataFrame:
    records = []
    for start in starts:
        end = start + pd.DateOffset(years=5)
        for symbol, frame in aligned_window(assets, start, end).items():
            _, metrics, _ = simulate_weekly(frame, weekly, slippage_bps)
            records.append({"window_start": str(start.date()), "symbol": symbol, **metrics})
    return pd.DataFrame(records)


def chart(histories: dict[str, pd.DataFrame], output: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True, layout="constrained",
                             gridspec_kw={"height_ratios": [1.5, 1, 1]})
    fig.set_facecolor("#ffffff")
    for symbol, frame in histories.items():
        axes[0].plot(frame.index, frame.equity, label=f"{symbol}: ${frame.equity.iloc[-1]:,.0f}",
                     color=COLORS[symbol], linewidth=2)
        axes[1].plot(frame.index, frame.profit_on_deposited, color=COLORS[symbol], linewidth=1.5)
        axes[2].plot(frame.index, frame.drawdown, color=COLORS[symbol], linewidth=1.5)
    reference = histories["QQQ"]
    axes[0].plot(reference.index, reference.deposited, "--", color="#777777", label="Contributions", linewidth=1.3)
    axes[0].set_title(f"Weekly $100 DCA | {reference.index[0].date()} to {reference.index[-1].date()}", loc="left", pad=14)
    axes[0].set_ylabel("Account value")
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x / 1000:,.0f}k"))
    axes[0].legend(loc="upper left", frameon=False, ncol=2)
    axes[1].set_ylabel("Profit / contributions")
    axes[1].axhline(0, color="#777777", linewidth=.8)
    axes[2].set_ylabel("Flow-adjusted drawdown")
    for ax in axes[1:]:
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    for ax in axes:
        ax.grid(axis="y", alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output / "comparison.png", dpi=170)
    plt.close(fig)


def write_report(summary: pd.DataFrame, rolling: pd.DataFrame, checks: pd.DataFrame,
                 output: Path, weekly: float, slippage_bps: float) -> None:
    first = summary.iloc[0]
    winners = rolling.pivot(index="window_start", columns="symbol", values="final_value").idxmax(axis=1).value_counts()
    lines = ["# Five-year weekly DCA comparison", "",
             f"Actual common window: {first.start} through {first.end}.", "",
             f"No initial lump sum. ${weekly:,.0f} per week. {int(first.trades)} purchases; total contributed ${first.contributed:,.0f} per ETF.", "",
             "| ETF | Final value | Profit | Profit / deposits | XIRR | Flow-adjusted max drawdown | Sharpe (0% rf) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for s, r in summary.iterrows():
        lines.append(f"| {s} | ${r.final_value:,.2f} | ${r.profit:,.2f} | {r.profit_on_contributed:.2%} | {r.xirr:.2%} | {r.max_drawdown:.2%} | {r.sharpe_0rf:.3f} |")
    lines += ["", "## What the investor experienced", "",
              "Drawdown removes deposits using overnight/intraday chain linking. It is not the loss percentage on all money deposited to date.", "",
              "| ETF | Worst loss / deposits | Date | Deposited then | Account then | Drawdown peak | Drawdown trough | Recovered |",
              "|---|---:|---|---:|---:|---|---|---|"]
    for s, r in summary.iterrows():
        lines.append(f"| {s} | {r.worst_loss_on_contributed:.2%} | {r.worst_loss_date} | ${r.worst_loss_deposited:,.0f} | ${r.worst_loss_value:,.0f} | {r.drawdown_peak} | {r.drawdown_trough} | {r.drawdown_recovered or 'Not recovered'} |")
    lines += ["", "## Same-window lump-sum counterfactual", "",
              "This is a separate comparison, not the DCA outcome. A dollar fully invested at the first open is held to the final close.", "",
              "| ETF | Lump-sum total return |", "|---|---:|"]
    for s, r in summary.iterrows():
        lines.append(f"| {s} | {r.lump_sum_return:.2%} |")
    lines += ["", "## Different five-year starting dates", "",
              f"Monthly starting dates from {rolling.window_start.min()} through {rolling.window_start.max()}; {len(rolling) // 3} overlapping five-year windows.",
              "Windows share most of their observations. These are descriptive sensitivity checks, not independent trials or forecasts.", "",
              "| ETF | Highest final value (windows) | Lowest XIRR | Median XIRR | Highest XIRR |",
              "|---|---:|---:|---:|---:|"]
    for s in SYMBOLS:
        series = rolling.loc[rolling.symbol.eq(s), "xirr"]
        lines.append(f"| {s} | {winners.get(s, 0)} | {series.min():.2%} | {series.median():.2%} | {series.max():.2%} |")
    lines += ["", "Selected January-start windows (all monthly windows are in the companion data):", "",
              "| Window | Contributed | QQQ final | QLD final | TQQQ final |", "|---|---:|---:|---:|---:|"]
    for start, group in rolling[rolling.window_start.str.endswith("-01-01")].groupby("window_start"):
        values = group.set_index("symbol")
        lines.append(f"| {group.iloc[0].start} to {group.iloc[0].end} | ${group.iloc[0].contributed:,.0f} | ${values.at['QQQ', 'final_value']:,.0f} | ${values.at['QLD', 'final_value']:,.0f} | ${values.at['TQQQ', 'final_value']:,.0f} |")
    lines += ["", "## Methodology and limitations", "",
              "- Buy at the first available regular-session open of each ISO calendar week. The partial first week also receives one deposit. Same dates and dollars across ETFs.",
              f"- Fractional shares; {slippage_bps:g} bps adverse price slippage per purchase; zero commissions and taxes; no sales or terminal liquidation cost.",
              "- Yahoo adjusted open = raw open multiplied by adjusted-close / close. Adjusted-price units are synthetic total-return units, NOT literal historical broker shares. This approximates dividend reinvestment and handles splits without creating artificial gains.",
              "- ETF management fees and realized leverage financing/tracking effects are already embedded in historical prices; do not deduct the current expense ratio again.",
              "- XIRR uses the actual deposit dates and terminal marked-to-market value; it is annualized. Profit / deposits is NOT annualized. Sharpe uses daily flow-adjusted returns and assumes 0% risk-free, not contemporaneous Treasury rates.",
              "- No historical index-return multiplication or invented pre-inception TQQQ series. Common data start in February 2010, so these results omit the 2000 and 2008 crises.",
              "- Dollar price level alone cannot inflate percentage returns: rescaling every price leaves the backtest unchanged. Starting valuation and subsequent market path can change results greatly.",
              "- Latest fetched common complete daily bar is 2026-09-16 although the download was requested through 2026-09-17. No later price is fabricated or spliced into the adjusted series.",
              "- Historical total returns are exploratory estimates, not a recommendation or a prediction. Continuous weekly contributions during unemployment or drawdowns are assumed.",
              "", "## Independent aggregate check", "",
              "Compare Yahoo adjusted-close CAGR for 2021-08-31 to 2026-08-31 with ProShares published five-year market-price annualized return as of 2026-08-31. This checks aggregates only, not every bar or DCA execution. Issuer returns use closing bid/ask midpoints and distribution assumptions may differ.", "",
              "| ETF | Yahoo calculated | Issuer reported | Difference (bps/year) |", "|---|---:|---:|---:|"]
    for r in checks.itertuples():
        lines.append(f"| {r.symbol} | {r.calculated_cagr:.4%} | {r.issuer_cagr:.2%} | {r.difference_bps:+.2f} |")
    lines += ["", "Sources:", "",
              "- [QQQ daily prices](https://finance.yahoo.com/quote/QQQ/history/)",
              "- [QLD daily prices](https://finance.yahoo.com/quote/QLD/history/)",
              "- [TQQQ daily prices](https://finance.yahoo.com/quote/TQQQ/history/)",
              "- [QLD issuer objective and performance](https://www.proshares.com/our-etfs/leveraged-and-inverse/qld)",
              "- [TQQQ issuer objective and performance](https://www.proshares.com/our-etfs/leveraged-and-inverse/tqqq)", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weekly", type=float, default=100.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    data = root / "data" / "five_year_dca_2026-09-17"
    output = root / "output" / "five_year_dca_2026-09-17"
    output.mkdir(parents=True, exist_ok=True)
    paths = {s: data / f"{s.lower()}_2010-01-01_2026-09-17.csv" for s in SYMBOLS}
    assets = {s: pd.read_csv(p, index_col="date", parse_dates=True) for s, p in paths.items()}
    end = min(f.index[-1] for f in assets.values())
    start = end - pd.DateOffset(years=5)
    histories, rows = {}, {}
    for s, frame in aligned_window(assets, start, end).items():
        history, metrics, trades = simulate_weekly(frame, args.weekly, args.slippage_bps)
        histories[s], rows[s] = history, metrics
        history.to_csv(output / f"{s}_history.csv", index_label="date")
        trades.to_csv(output / f"{s}_trades.csv", index=False)
    summary = pd.DataFrame(rows).T
    summary.to_csv(output / "summary.csv", index_label="symbol")
    first_common = max(f.index[0] for f in assets.values())
    starts = pd.date_range(first_common, start, freq="MS")
    rolling = run_windows(assets, starts, args.weekly, args.slippage_bps)
    rolling.to_csv(output / "rolling_five_year_windows.csv", index=False)

    records = []
    for s, published in {"QLD": .1727, "TQQQ": .1512}.items():
        prices = assets[s].adj_close
        cagr = (prices.loc["2026-08-31"] / prices.loc["2021-08-31"]) ** .2 - 1
        records.append({"symbol": s, "calculated_cagr": cagr, "issuer_cagr": published,
                        "difference_bps": (cagr - published) * 10000})
    checks = pd.DataFrame(records)
    checks.to_csv(output / "issuer_crosscheck.csv", index=False)
    provenance = {"run_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                  "requested_end": "2026-09-17", "actual_end": str(end.date()),
                  "weekly": args.weekly, "slippage_bps": args.slippage_bps,
                  "source": "Yahoo chart API; split/dividend adjusted daily open and close",
                  "files": {s: {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                                "rows": len(assets[s])} for s, p in paths.items()}}
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    chart(histories, output)
    write_report(summary, rolling, checks, output, args.weekly, args.slippage_bps)
    print(summary.to_string())
    print("\nRolling window winners:")
    print(rolling.pivot(index="window_start", columns="symbol", values="final_value").idxmax(axis=1).value_counts().to_string())
    print(f"\nReport: {output / 'report.md'}")


if __name__ == "__main__":
    main()
