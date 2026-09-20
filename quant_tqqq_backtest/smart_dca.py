#!/usr/bin/env python3
"""Compare fixed and risk-aware QQQ/TQQQ contribution strategies."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from quant_tqqq_backtest.backtest import download_yahoo, max_drawdown, xirr


SLIPPAGE_BPS = 5.0


@dataclass
class PortfolioResult:
    equity: pd.Series
    contributions: pd.Series
    opening_equity: pd.Series
    tqqq_weight: pd.Series
    trades: pd.DataFrame


def regime_with_buffer(
    close: pd.Series,
    sma_days: int = 200,
    bull_multiple: float = 1.02,
    bear_multiple: float = 0.98,
) -> pd.DataFrame:
    frame = pd.DataFrame(index=close.index)
    frame["close"] = close
    frame["sma"] = close.rolling(sma_days, min_periods=sma_days).mean()
    frame["realized_vol"] = close.pct_change().rolling(63, min_periods=42).std() * math.sqrt(252)

    previous: str | None = None
    values: list[str | None] = []
    for row in frame.itertuples():
        if pd.isna(row.sma):
            values.append(None)
            continue
        if row.close >= row.sma * bull_multiple:
            previous = "bull"
        elif row.close <= row.sma * bear_multiple:
            previous = "bear"
        elif previous is None:
            previous = "bull" if row.close >= row.sma else "bear"
        values.append(previous)
    frame["regime"] = values
    return frame


def fixed_target(index: pd.Index, tqqq_weight: float) -> pd.Series:
    return pd.Series(float(tqqq_weight), index=index, name="target_tqqq")


def trend_target(signal: pd.DataFrame, bull_weight: float = 0.30) -> pd.Series:
    target = pd.Series(0.0, index=signal.index, name="target_tqqq")
    target.loc[signal["regime"].eq("bull")] = bull_weight
    return target


def volatility_target(
    signal: pd.DataFrame,
    target_annual_vol: float = 0.30,
    max_tqqq_weight: float = 0.30,
) -> pd.Series:
    # A QQQ/TQQQ mix has approximate daily beta 1 + 2w, where w is TQQQ weight.
    raw_weight = (target_annual_vol / signal["realized_vol"] - 1.0) / 2.0
    target = raw_weight.clip(lower=0.0, upper=max_tqqq_weight).fillna(0.0)
    target.loc[signal["regime"].ne("bull")] = 0.0
    target.name = "target_tqqq"
    return target


def simulate_mix(
    qqq: pd.DataFrame,
    tqqq: pd.DataFrame,
    target_tqqq: pd.Series,
    initial_cash: float,
    weekly_contribution: float,
    slippage_bps: float = SLIPPAGE_BPS,
) -> PortfolioResult:
    dates = qqq.index.intersection(tqqq.index)
    qqq = qqq.loc[dates]
    tqqq = tqqq.loc[dates]
    target_tqqq = target_tqqq.sort_index()
    cash = float(initial_cash)
    qqq_shares = 0.0
    tqqq_shares = 0.0
    last_week: tuple[int, int] | None = None
    last_month: tuple[int, int] | None = None
    prior_target: float | None = None
    slip = slippage_bps / 10_000.0
    equity_values: list[float] = []
    contribution_values: list[float] = []
    opening_values: list[float] = []
    weight_values: list[float] = []
    trades: list[dict[str, object]] = []

    def execute(side: str, symbol: str, dollars: float, open_price: float, current_date: pd.Timestamp) -> float:
        nonlocal cash, qqq_shares, tqqq_shares
        if dollars <= 0.01:
            return 0.0
        if side == "sell":
            price = open_price * (1.0 - slip)
            shares = dollars / open_price
            if symbol == "QQQ":
                shares = min(shares, qqq_shares)
                qqq_shares -= shares
            else:
                shares = min(shares, tqqq_shares)
                tqqq_shares -= shares
            proceeds = shares * price
            cash += proceeds
            notional = proceeds
        else:
            dollars = min(dollars, cash)
            price = open_price * (1.0 + slip)
            shares = dollars / price
            if symbol == "QQQ":
                qqq_shares += shares
            else:
                tqqq_shares += shares
            cash -= dollars
            notional = dollars
        trades.append(
            {
                "date": current_date,
                "side": side,
                "symbol": symbol,
                "shares": shares,
                "price": price,
                "notional": notional,
            }
        )
        return notional

    for i, current_date in enumerate(dates):
        contribution = 0.0
        iso = current_date.isocalendar()
        week = (int(iso.year), int(iso.week))
        if week != last_week:
            contribution += weekly_contribution
            cash += weekly_contribution
            last_week = week
        if i == 0:
            contribution += initial_cash

        signal_position = target_tqqq.index.searchsorted(current_date, side="left") - 1
        target = float(target_tqqq.iloc[signal_position]) if signal_position >= 0 else 0.0
        month = (current_date.year, current_date.month)
        regime_flip = prior_target is not None and ((target == 0.0) != (prior_target == 0.0))
        rebalance = i == 0 or month != last_month or regime_flip
        q_open = float(qqq.at[current_date, "adj_open"])
        t_open = float(tqqq.at[current_date, "adj_open"])

        total_open = cash + qqq_shares * q_open + tqqq_shares * t_open
        opening_values.append(total_open)
        desired_t = total_open * target
        desired_q = total_open - desired_t
        q_value = qqq_shares * q_open
        t_value = tqqq_shares * t_open

        if rebalance:
            if t_value > desired_t:
                execute("sell", "TQQQ", t_value - desired_t, t_open, current_date)
            if q_value > desired_q:
                execute("sell", "QQQ", q_value - desired_q, q_open, current_date)

        total_open = cash + qqq_shares * q_open + tqqq_shares * t_open
        desired_t = total_open * target
        desired_q = total_open - desired_t
        t_gap = max(desired_t - tqqq_shares * t_open, 0.0)
        q_gap = max(desired_q - qqq_shares * q_open, 0.0)
        total_gap = t_gap + q_gap
        if cash > 0.01:
            if total_gap > 0:
                t_budget = min(cash, cash * t_gap / total_gap)
                q_budget = cash - t_budget
            else:
                t_budget = cash * target
                q_budget = cash - t_budget
            execute("buy", "TQQQ", t_budget, t_open, current_date)
            execute("buy", "QQQ", q_budget, q_open, current_date)

        q_close = float(qqq.at[current_date, "adj_close"])
        t_close = float(tqqq.at[current_date, "adj_close"])
        equity = cash + qqq_shares * q_close + tqqq_shares * t_close
        equity_values.append(equity)
        contribution_values.append(contribution)
        weight_values.append((tqqq_shares * t_close / equity) if equity else 0.0)
        last_month = month
        prior_target = target

    return PortfolioResult(
        equity=pd.Series(equity_values, index=dates, name="equity"),
        contributions=pd.Series(contribution_values, index=dates, name="contribution"),
        opening_equity=pd.Series(opening_values, index=dates, name="opening_equity"),
        tqqq_weight=pd.Series(weight_values, index=dates, name="tqqq_weight"),
        trades=pd.DataFrame(trades),
    )


def contribution_adjusted_returns(result: PortfolioResult) -> pd.Series:
    # Chain overnight and intraday returns around the opening deposit.
    previous_close = result.equity.shift(1)
    overnight = (result.opening_equity - result.contributions) / previous_close
    overnight.iloc[0] = 1.0
    intraday = result.equity / result.opening_equity
    return (overnight * intraday - 1.0).dropna()


def summarize(result: PortfolioResult) -> dict[str, float]:
    returns = contribution_adjusted_returns(result)
    wealth = (1.0 + returns).cumprod()
    years = (returns.index[-1] - returns.index[0]).days / 365.2425
    cagr = float(wealth.iloc[-1] ** (1.0 / years) - 1.0)
    volatility = float(returns.std() * math.sqrt(252))
    sharpe = float(returns.mean() / returns.std() * math.sqrt(252))
    return {
        "contributed": float(result.contributions.sum()),
        "final_value": float(result.equity.iloc[-1]),
        "xirr": xirr(result.contributions, float(result.equity.iloc[-1])),
        "twr_cagr": cagr,
        "max_drawdown": float((wealth / wealth.cummax().clip(lower=1.0) - 1.0).min()),
        "volatility": volatility,
        "sharpe": sharpe,
        "average_tqqq_weight": float(result.tqqq_weight.mean()),
        "trades": float(len(result.trades)),
    }


def main() -> None:
    root = Path(__file__).resolve().parent
    cache_dir = root / "data"
    output_dir = root / "output"
    output_dir.mkdir(exist_ok=True)
    end = date.today()
    qqq_full = download_yahoo("QQQ", date(2008, 1, 1), end, cache_dir)
    tqqq = download_yahoo("TQQQ", date(2009, 1, 1), end, cache_dir)
    dates = qqq_full.index.intersection(tqqq.index)
    qqq = qqq_full.loc[dates]
    signal = regime_with_buffer(qqq_full["adj_close"])

    targets = {
        "QQQ only": fixed_target(signal.index, 0.0),
        "90% QQQ / 10% TQQQ": fixed_target(signal.index, 0.10),
        "80% QQQ / 20% TQQQ": fixed_target(signal.index, 0.20),
        "70% QQQ / 30% TQQQ": fixed_target(signal.index, 0.30),
        "Trend 0% or 30% TQQQ": trend_target(signal, 0.30),
        "Volatility target 0-30% TQQQ": volatility_target(signal, 0.30, 0.30),
        "TQQQ only": fixed_target(signal.index, 1.0),
    }
    results = {
        name: simulate_mix(qqq, tqqq, target, 10_000.0, 100.0)
        for name, target in targets.items()
    }
    summary = pd.DataFrame({name: summarize(result) for name, result in results.items()}).T
    summary.to_csv(output_dir / "smart_dca_comparison.csv", index_label="strategy")

    lines = [
        "# Smarter weekly contribution comparison",
        "",
        f"Period: {dates[0].date()} through {dates[-1].date()}",
        "",
        "Assumptions: $10,000 initial contribution, $100 each week, monthly rebalancing, "
        "5 bps adverse slippage, fractional shares, no commissions or taxes.",
        "Signals strictly precede each execution, including initial entry. Returns treat deposits "
        "as opening flows, chaining overnight and intraday returns separately; initial trading costs "
        "are included. Sharpe assumes a zero risk-free rate.",
        "",
        "| Strategy | Final value | XIRR | TWR CAGR | Max drawdown | Volatility | Sharpe | Avg TQQQ weight |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in summary.iterrows():
        lines.append(
            f"| {name} | ${row['final_value']:,.0f} | {row['xirr']:.2%} | {row['twr_cagr']:.2%} "
            f"| {row['max_drawdown']:.2%} | {row['volatility']:.2%} | {row['sharpe']:.2f} "
            f"| {row['average_tqqq_weight']:.2%} |"
        )
    lines.extend(
        [
            "",
            "The trend strategy holds 30% TQQQ only in the buffered QQQ bull regime. "
            "The volatility-target strategy uses the same regime and scales TQQQ from 0% to 30% "
            "to target approximately 30% annualized portfolio volatility.",
            "",
        ]
    )
    report = "\n".join(lines)
    (output_dir / "smart_dca_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
