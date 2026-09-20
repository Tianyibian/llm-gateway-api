#!/usr/bin/env python3
"""Backtest a QQQ regime strategy that holds TQQQ during bull regimes."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"


@dataclass(frozen=True)
class StrategyConfig:
    sma_days: int = 200
    bull_multiple: float = 1.04
    bear_multiple: float = 0.97
    dip_threshold: float = -0.01
    slippage_bps: float = 5.0


@dataclass
class SimulationResult:
    equity: pd.Series
    trades: pd.DataFrame
    contributions: pd.Series
    exposure: pd.Series


def download_yahoo(symbol: str, start: date, end: date, cache_dir: Path) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{symbol.lower()}_{start}_{end}.csv"
    if cache_path.exists():
        return pd.read_csv(cache_path, index_col="date", parse_dates=True)

    period1 = int(datetime.combine(start, datetime.min.time(), tzinfo=timezone.utc).timestamp())
    period2 = int(
        datetime.combine(end + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc).timestamp()
    )
    query = urllib.parse.urlencode(
        {
            "period1": period1,
            "period2": period2,
            "interval": "1d",
            "events": "div,splits",
            "includeAdjustedClose": "true",
        }
    )
    request = urllib.request.Request(
        f"{YAHOO_CHART_URL.format(symbol=symbol)}?{query}",
        headers={"User-Agent": "Mozilla/5.0 quant-backtest/1.0"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.load(response)

    result = payload["chart"]["result"][0]
    quote = result["indicators"]["quote"][0]
    adjclose = result["indicators"]["adjclose"][0]["adjclose"]
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(result["timestamp"], unit="s", utc=True).tz_convert(None).normalize(),
            "open": quote["open"],
            "high": quote["high"],
            "low": quote["low"],
            "close": quote["close"],
            "volume": quote["volume"],
            "adj_close": adjclose,
        }
    ).set_index("date")
    frame = frame.dropna(subset=["open", "close", "adj_close"]).sort_index()
    adjustment = frame["adj_close"] / frame["close"]
    frame["adj_open"] = frame["open"] * adjustment
    frame.to_csv(cache_path, index_label="date")
    return frame


def compute_regime(close: pd.Series, config: StrategyConfig) -> pd.DataFrame:
    signal = pd.DataFrame(index=close.index)
    signal["close"] = close
    signal["sma"] = close.rolling(config.sma_days, min_periods=config.sma_days).mean()
    signal["daily_return"] = close.pct_change()
    signal["regime"] = pd.Series(index=signal.index, dtype="object")

    previous: str | None = None
    regimes: list[str | None] = []
    for row in signal.itertuples():
        if pd.isna(row.sma):
            regimes.append(None)
            continue
        if row.close >= row.sma * config.bull_multiple:
            previous = "bull"
        elif row.close <= row.sma * config.bear_multiple:
            previous = "bear"
        regimes.append(previous)
    signal["regime"] = regimes
    return signal


def compute_actions(signal: pd.DataFrame, config: StrategyConfig) -> pd.Series:
    actions = pd.Series("hold", index=signal.index, dtype="object")
    previous_regime = signal["regime"].shift(1)
    bull_transition = signal["regime"].eq("bull") & previous_regime.ne("bull")
    bear_transition = signal["regime"].eq("bear") & previous_regime.eq("bull")
    bull_dip = signal["regime"].eq("bull") & signal["daily_return"].le(config.dip_threshold)
    actions.loc[bull_transition | bull_dip] = "buy"
    actions.loc[bear_transition] = "sell"
    return actions


def simulate_strategy(
    signal: pd.DataFrame,
    actions: pd.Series,
    asset: pd.DataFrame,
    config: StrategyConfig,
    initial_cash: float,
    weekly_contribution: float,
    enter_if_initial_bull: bool,
) -> SimulationResult:
    dates = asset.index
    cash = float(initial_cash)
    shares = 0.0
    last_week: tuple[int, int] | None = None
    first_day = True
    equity_values: list[float] = []
    exposure_values: list[bool] = []
    contribution_values: list[float] = []
    trades: list[dict[str, object]] = []
    slip = config.slippage_bps / 10_000.0

    for current_date in dates:
        contribution = 0.0
        iso = current_date.isocalendar()
        week = (int(iso.year), int(iso.week))
        if weekly_contribution and week != last_week:
            contribution = weekly_contribution
            cash += contribution
            last_week = week

        prior_signals = signal.index[signal.index < current_date]
        action = "hold"
        signal_date = pd.NaT
        regime = None
        if len(prior_signals):
            signal_date = prior_signals[-1]
            regime = signal.at[signal_date, "regime"]
            action = actions.at[signal_date]
            if first_day and enter_if_initial_bull and regime == "bull":
                action = "buy"

        open_price = float(asset.at[current_date, "adj_open"])
        if action == "sell" and shares > 0:
            execution_price = open_price * (1.0 - slip)
            proceeds = shares * execution_price
            trades.append(
                {
                    "execution_date": current_date,
                    "signal_date": signal_date,
                    "side": "sell",
                    "price": execution_price,
                    "shares": shares,
                    "notional": proceeds,
                    "regime": regime,
                }
            )
            cash += proceeds
            shares = 0.0
        elif action == "buy" and cash > 0.01:
            execution_price = open_price * (1.0 + slip)
            quantity = cash / execution_price
            trades.append(
                {
                    "execution_date": current_date,
                    "signal_date": signal_date,
                    "side": "buy",
                    "price": execution_price,
                    "shares": quantity,
                    "notional": cash,
                    "regime": regime,
                }
            )
            shares += quantity
            cash = 0.0

        close_price = float(asset.at[current_date, "adj_close"])
        equity_values.append(cash + shares * close_price)
        exposure_values.append(shares > 0)
        contribution_values.append(contribution)
        first_day = False

    return SimulationResult(
        equity=pd.Series(equity_values, index=dates, name="strategy"),
        trades=pd.DataFrame(trades),
        contributions=pd.Series(contribution_values, index=dates, name="contribution"),
        exposure=pd.Series(exposure_values, index=dates, name="exposed"),
    )


def simulate_buy_hold(
    asset: pd.DataFrame,
    initial_cash: float,
    weekly_contribution: float,
    slippage_bps: float,
    name: str,
) -> SimulationResult:
    cash = float(initial_cash)
    shares = 0.0
    last_week: tuple[int, int] | None = None
    equity_values: list[float] = []
    contribution_values: list[float] = []
    trades: list[dict[str, object]] = []
    slip = slippage_bps / 10_000.0

    for current_date, row in asset.iterrows():
        contribution = 0.0
        iso = current_date.isocalendar()
        week = (int(iso.year), int(iso.week))
        if weekly_contribution and week != last_week:
            contribution = weekly_contribution
            cash += contribution
            last_week = week
        if cash > 0.01:
            execution_price = float(row["adj_open"]) * (1.0 + slip)
            quantity = cash / execution_price
            trades.append(
                {
                    "execution_date": current_date,
                    "signal_date": pd.NaT,
                    "side": "buy",
                    "price": execution_price,
                    "shares": quantity,
                    "notional": cash,
                    "regime": "always",
                }
            )
            shares += quantity
            cash = 0.0
        equity_values.append(cash + shares * float(row["adj_close"]))
        contribution_values.append(contribution)

    dates = asset.index
    return SimulationResult(
        equity=pd.Series(equity_values, index=dates, name=name),
        trades=pd.DataFrame(trades),
        contributions=pd.Series(contribution_values, index=dates, name="contribution"),
        exposure=pd.Series(True, index=dates, name="exposed"),
    )


def max_drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def drawdown_period(equity: pd.Series) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp | None]:
    drawdown = equity / equity.cummax() - 1.0
    trough = drawdown.idxmin()
    peak = equity.loc[:trough].idxmax()
    recovered = equity.loc[trough:][equity.loc[trough:] >= equity.loc[peak]]
    recovery = recovered.index[0] if len(recovered) else None
    return peak, trough, recovery


def lump_sum_metrics(result: SimulationResult, initial_cash: float) -> dict[str, float]:
    years = (result.equity.index[-1] - result.equity.index[0]).days / 365.2425
    final_value = float(result.equity.iloc[-1])
    total_return = final_value / initial_cash - 1.0
    cagr = (final_value / initial_cash) ** (1.0 / years) - 1.0
    daily_returns = result.equity.pct_change().dropna()
    volatility = float(daily_returns.std() * math.sqrt(252))
    sharpe = float(daily_returns.mean() / daily_returns.std() * math.sqrt(252))
    return {
        "final_value": final_value,
        "total_return": total_return,
        "cagr": cagr,
        "max_drawdown": max_drawdown(result.equity),
        "volatility": volatility,
        "sharpe_0rf": sharpe,
        "trades": float(len(result.trades)),
        "exposure": float(result.exposure.mean()),
    }


def xirr(contributions: pd.Series, final_value: float) -> float:
    cash_flows: list[tuple[pd.Timestamp, float]] = [
        (timestamp, -float(value))
        for timestamp, value in contributions.items()
        if value > 0
    ]
    cash_flows.append((contributions.index[-1], float(final_value)))
    start = cash_flows[0][0]

    def npv(rate: float) -> float:
        return sum(
            amount / ((1.0 + rate) ** ((timestamp - start).days / 365.2425))
            for timestamp, amount in cash_flows
        )

    low, high = -0.9999, 10.0
    while npv(high) > 0 and high < 1_000_000:
        high *= 2.0
    for _ in range(200):
        middle = (low + high) / 2.0
        if npv(middle) > 0:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def dca_metrics(result: SimulationResult) -> dict[str, float]:
    total_contributed = float(result.contributions.sum())
    final_value = float(result.equity.iloc[-1])
    return {
        "contributed": total_contributed,
        "final_value": final_value,
        "profit": final_value - total_contributed,
        "xirr": xirr(result.contributions, final_value),
        "trades": float(len(result.trades)),
        "exposure": float(result.exposure.mean()),
    }


def run_sensitivity(
    qqq_close: pd.Series,
    tqqq: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for bull, bear, dip in itertools.product(
        (1.02, 1.04, 1.06),
        (0.95, 0.97, 0.99),
        (-0.005, -0.01, -0.02),
    ):
        config = StrategyConfig(
            bull_multiple=bull,
            bear_multiple=bear,
            dip_threshold=dip,
        )
        signal = compute_regime(qqq_close, config)
        actions = compute_actions(signal, config)
        lump = simulate_strategy(signal, actions, tqqq, config, 10_000.0, 0.0, True)
        dca = simulate_strategy(signal, actions, tqqq, config, 0.0, 100.0, True)
        lump_summary = lump_sum_metrics(lump, 10_000.0)
        dca_summary = dca_metrics(dca)
        rows.append(
            {
                "bull_multiple": bull,
                "bear_multiple": bear,
                "dip_threshold": dip,
                "lump_cagr": lump_summary["cagr"],
                "lump_max_drawdown": lump_summary["max_drawdown"],
                "dca_xirr": dca_summary["xirr"],
                "lump_trades": lump_summary["trades"],
                "dca_trades": dca_summary["trades"],
            }
        )
    return pd.DataFrame(rows)


def format_percent(value: float) -> str:
    return f"{value * 100:,.2f}%"


def write_report(
    output_dir: Path,
    config: StrategyConfig,
    start: pd.Timestamp,
    end: pd.Timestamp,
    lump_results: dict[str, SimulationResult],
    dca_results: dict[str, SimulationResult],
    signal: pd.DataFrame,
    sensitivity: pd.DataFrame,
) -> None:
    lump_metrics = {name: lump_sum_metrics(result, 10_000.0) for name, result in lump_results.items()}
    dca_summary = {name: dca_metrics(result) for name, result in dca_results.items()}
    strategy = lump_results["Strategy (initial bull entry)"]
    peak, trough, recovery = drawdown_period(strategy.equity)
    latest_signal = signal.dropna(subset=["sma"]).iloc[-1]
    latest_date = signal.dropna(subset=["sma"]).index[-1]

    lines = [
        "# QQQ/TQQQ regime strategy backtest",
        "",
        f"Period: {start.date()} through {end.date()}",
        "",
        "## Assumptions",
        "",
        f"- Bull: QQQ adjusted close >= {config.bull_multiple:.2f} x its {config.sma_days}-day SMA.",
        f"- Bear: QQQ adjusted close <= {config.bear_multiple:.2f} x its {config.sma_days}-day SMA.",
        "- The prior regime persists inside the buffer band.",
        f"- In a bull regime, a QQQ close-to-close return <= {config.dip_threshold:.2%} invests all available cash in TQQQ.",
        "- A bear-to-bull transition invests all available cash; a bull-to-bear transition sells all TQQQ.",
        "- Signals use confirmed closes and execute at the next trading day's adjusted open.",
        f"- Each execution includes {config.slippage_bps:.1f} bps of adverse slippage; fractional shares and zero commissions are assumed.",
        "- Adjusted Yahoo Finance prices are used as an exploratory data source and approximate dividend reinvestment.",
        "",
        "## $10,000 lump-sum comparison",
        "",
        "| Portfolio | Final value | Total return | CAGR | Max drawdown | Volatility | Sharpe (0% rf) | Trades | Exposure |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in lump_metrics.items():
        lines.append(
            f"| {name} | ${metrics['final_value']:,.0f} | {format_percent(metrics['total_return'])} "
            f"| {format_percent(metrics['cagr'])} | {format_percent(metrics['max_drawdown'])} "
            f"| {format_percent(metrics['volatility'])} | {metrics['sharpe_0rf']:.2f} "
            f"| {metrics['trades']:.0f} | {format_percent(metrics['exposure'])} |"
        )

    lines.extend(
        [
            "",
            "## $100 weekly contribution comparison",
            "",
            "| Portfolio | Contributions | Final value | Profit | Money-weighted annual return | Trades | Exposure |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, metrics in dca_summary.items():
        lines.append(
            f"| {name} | ${metrics['contributed']:,.0f} | ${metrics['final_value']:,.0f} "
            f"| ${metrics['profit']:,.0f} | {format_percent(metrics['xirr'])} "
            f"| {metrics['trades']:.0f} | {format_percent(metrics['exposure'])} |"
        )

    lines.extend(
        [
            "",
            "## Risk and robustness checks",
            "",
            f"- Worst strategy drawdown ran from {peak.date()} to {trough.date()} and reached {format_percent(max_drawdown(strategy.equity))}.",
            f"- The prior peak was {'recovered on ' + str(recovery.date()) if recovery is not None else 'not recovered by the end of the sample'}.",
            f"- Across 27 neighboring parameter combinations, lump-sum CAGR ranged from {format_percent(sensitivity['lump_cagr'].min())} to {format_percent(sensitivity['lump_cagr'].max())}.",
            f"- Across those combinations, maximum drawdown ranged from {format_percent(sensitivity['lump_max_drawdown'].min())} to {format_percent(sensitivity['lump_max_drawdown'].max())}.",
            f"- Weekly-contribution money-weighted returns ranged from {format_percent(sensitivity['dca_xirr'].min())} to {format_percent(sensitivity['dca_xirr'].max())}.",
            "",
            "## Latest signal in the downloaded data",
            "",
            f"- Date: {latest_date.date()}",
            f"- QQQ adjusted close: ${latest_signal['close']:,.2f}",
            f"- 200-day SMA: ${latest_signal['sma']:,.2f}",
            f"- Bull threshold: ${latest_signal['sma'] * config.bull_multiple:,.2f}",
            f"- Bear threshold: ${latest_signal['sma'] * config.bear_multiple:,.2f}",
            f"- Regime: {latest_signal['regime']}",
            f"- Daily return: {format_percent(latest_signal['daily_return'])}",
        ]
    )

    lines.extend(
        [
            "",
            "## Important limitations",
            "",
            "- This is a historical simulation, not a forecast or investment recommendation.",
            "- Results are sensitive to the data vendor, adjustment method, execution time, spread, slippage, taxes, and the launch-day initialization rule.",
            "- TQQQ targets 3x the Nasdaq-100's daily return. Daily reset and compounding can make long-horizon results diverge substantially from 3x QQQ.",
            "- Autonomous execution should not be enabled until signals, duplicate-order protection, stale-data handling, and pre-trade review have been tested in observation mode.",
            "",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def plot_results(
    output_dir: Path,
    lump_results: dict[str, SimulationResult],
    dca_results: dict[str, SimulationResult],
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), constrained_layout=True)
    for name, result in lump_results.items():
        axes[0].plot(result.equity.index, result.equity, label=name)
    axes[0].set_yscale("log")
    axes[0].set_title("$10,000 lump-sum portfolio value (log scale)")
    axes[0].set_ylabel("Portfolio value ($)")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    for name, result in dca_results.items():
        axes[1].plot(result.equity.index, result.equity, label=name)
    axes[1].set_title("$100 weekly contribution portfolio value")
    axes[1].set_ylabel("Portfolio value ($)")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.savefig(output_dir / "equity_curves.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--end", type=date.fromisoformat, default=date.today())
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    cache_dir = root / "data"
    output_dir = root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.refresh:
        for path in cache_dir.glob("*.csv") if cache_dir.exists() else []:
            path.unlink()

    config = StrategyConfig()
    qqq = download_yahoo("QQQ", date(2008, 1, 1), args.end, cache_dir)
    tqqq = download_yahoo("TQQQ", date(2009, 1, 1), args.end, cache_dir)
    common_dates = tqqq.index.intersection(qqq.index)
    tqqq = tqqq.loc[common_dates]
    qqq_trade = qqq.loc[common_dates]

    signal = compute_regime(qqq["adj_close"], config)
    actions = compute_actions(signal, config)

    lump_results = {
        "Strategy (initial bull entry)": simulate_strategy(
            signal, actions, tqqq, config, 10_000.0, 0.0, True
        ),
        "Strategy (strict triggers)": simulate_strategy(
            signal, actions, tqqq, config, 10_000.0, 0.0, False
        ),
        "Buy & hold TQQQ": simulate_buy_hold(tqqq, 10_000.0, 0.0, config.slippage_bps, "TQQQ"),
        "Buy & hold QQQ": simulate_buy_hold(qqq_trade, 10_000.0, 0.0, config.slippage_bps, "QQQ"),
    }
    dca_results = {
        "Strategy (initial bull entry)": simulate_strategy(
            signal, actions, tqqq, config, 0.0, 100.0, True
        ),
        "Strategy (strict triggers)": simulate_strategy(
            signal, actions, tqqq, config, 0.0, 100.0, False
        ),
        "Weekly TQQQ": simulate_buy_hold(tqqq, 0.0, 100.0, config.slippage_bps, "TQQQ"),
        "Weekly QQQ": simulate_buy_hold(qqq_trade, 0.0, 100.0, config.slippage_bps, "QQQ"),
    }

    sensitivity = run_sensitivity(qqq["adj_close"], tqqq)
    sensitivity.to_csv(output_dir / "sensitivity.csv", index=False)
    annual_returns = pd.DataFrame(
        {
            name: result.equity.resample("YE").last().pct_change()
            for name, result in lump_results.items()
        }
    )
    annual_returns.iloc[0] = pd.Series(
        {
            name: result.equity.loc[: annual_returns.index[0]].iloc[-1] / 10_000.0 - 1.0
            for name, result in lump_results.items()
        }
    )
    annual_returns.to_csv(output_dir / "annual_returns.csv", index_label="year_end")

    for group_name, group in (("lump", lump_results), ("dca", dca_results)):
        for name, result in group.items():
            safe_name = (
                name.lower()
                .replace(" ", "_")
                .replace("&", "and")
                .replace("(", "")
                .replace(")", "")
            )
            result.trades.to_csv(output_dir / f"trades_{group_name}_{safe_name}.csv", index=False)

    write_report(
        output_dir,
        config,
        common_dates[0],
        common_dates[-1],
        lump_results,
        dca_results,
        signal,
        sensitivity,
    )
    plot_results(output_dir, lump_results, dca_results)
    print((output_dir / "report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
