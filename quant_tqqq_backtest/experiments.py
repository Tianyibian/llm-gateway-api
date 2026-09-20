"""Fixed-budget, close-signal/next-open experiments with cash and ETF holdings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from quant_tqqq_backtest.smart_dca import regime_with_buffer


@dataclass
class Experiment:
    history: pd.DataFrame
    trades: pd.DataFrame
    initial_cash: float


def simulate(
    assets: dict[str, pd.DataFrame],
    targets: pd.DataFrame,
    initial_cash: float = 500.0,
    slippage_bps: float = 5.0,
) -> Experiment:
    symbols = list(assets)
    dates = assets[symbols[0]].index
    for frame in assets.values():
        dates = dates.intersection(frame.index)
    targets = targets.reindex(columns=symbols, fill_value=0.0).sort_index()
    if targets.isna().any().any() or (targets < 0).any().any() or (targets.sum(axis=1) > 1.00000001).any():
        raise ValueError("Weights must be finite, long-only, and sum to at most one.")
    if initial_cash <= 0 or not 0 <= slippage_bps < 10000:
        raise ValueError("Invalid capital or execution cost.")

    shares = np.zeros(len(symbols))
    cash = initial_cash
    slip = slippage_bps / 10000
    rows, trades = [], []
    previous_month = None
    previous_support = None
    target_values = targets.to_numpy(dtype=float)
    positions = targets.index.searchsorted(dates, side="left") - 1
    opens = np.column_stack([assets[s].loc[dates, "adj_open"] for s in symbols])
    closes = np.column_stack([assets[s].loc[dates, "adj_close"] for s in symbols])
    for i, timestamp in enumerate(dates):
        p = positions[i]
        weights = target_values[p] if p >= 0 else np.zeros(len(symbols))
        support = tuple(weights > 1e-12)
        month = (timestamp.year, timestamp.month)
        rebalance = month != previous_month or support != previous_support
        current_values = shares * opens[i]
        pretrade = cash + current_values.sum()
        cost = 0.0
        if rebalance:
            # Solve post-cost NAV so simultaneous sales/buys are fully self-financing.
            low, high = 0.0, pretrade
            for _ in range(45):
                middle = (low + high) / 2
                cost = slip * np.abs(weights * middle - current_values).sum()
                if middle + cost > pretrade:
                    high = middle
                else:
                    low = middle
            net = (low + high) / 2
            changes = weights * net - current_values
            cost = slip * np.abs(changes).sum()
            shares = weights * net / opens[i]
            cash = pretrade - cost - float((weights * net).sum())
            for j, change in enumerate(changes):
                if abs(change) > 0.000001:
                    trades.append({"execution_date": timestamp, "signal_date": targets.index[p] if p >= 0 else pd.NaT,
                                   "symbol": symbols[j], "side": "buy" if change > 0 else "sell",
                                   "notional_at_open": abs(change), "cost": slip * abs(change)})
        nav = cash + float((shares * closes[i]).sum())
        row = {"date": timestamp, "equity": nav, "cash": cash, "cost": cost}
        row.update({s: shares[j] * closes[i, j] / nav for j, s in enumerate(symbols)})
        rows.append(row)
        previous_month, previous_support = month, support
    return Experiment(pd.DataFrame(rows).set_index("date"), pd.DataFrame(trades), initial_cash)


def metrics(result: Experiment) -> dict[str, float]:
    equity = result.history.equity
    wealth = equity / result.initial_cash
    returns = equity / equity.shift(1).fillna(result.initial_cash) - 1
    years = ((equity.index[-1] - equity.index[0]).days + 1) / 365.2425
    annual = equity.resample("YE").last()
    annual_returns = annual / annual.shift(1).fillna(result.initial_cash) - 1
    return {"final_value": equity.iloc[-1], "cagr": wealth.iloc[-1] ** (1 / years) - 1,
            "max_drawdown": (wealth / wealth.cummax().clip(lower=1) - 1).min(),
            "sharpe_0rf": returns.mean() / returns.std() * np.sqrt(252),
            "return_2020": annual_returns.get(pd.Timestamp("2020-12-31"), np.nan),
            "return_2022": annual_returns.get(pd.Timestamp("2022-12-31"), np.nan),
            "trades": len(result.trades)}


def build_targets(close: pd.Series) -> dict[str, pd.DataFrame]:
    original = regime_with_buffer(close, bull_multiple=1.04, bear_multiple=0.97)
    recent = regime_with_buffer(close, bull_multiple=1.02, bear_multiple=0.98)
    bull = original.regime.eq("bull")

    def weights(**values):
        return pd.DataFrame(values, index=close.index).reindex(columns=["QQQ", "QLD", "TQQQ"], fill_value=0.0)

    trend_weight = recent.regime.eq("bull").astype(float) * .7
    vol = close.pct_change(fill_method=None).rolling(63, min_periods=63).std() * np.sqrt(252)
    risk_weight = (0.25 / (2 * vol)).clip(upper=.75).fillna(0).where(bull, 0)
    return {
        "QQQ hold": weights(QQQ=1.0),
        "QLD hold": weights(QLD=1.0),
        "TQQQ hold": weights(TQQQ=1.0),
        "70% TQQQ trend / QQQ": weights(TQQQ=trend_weight, QQQ=1-trend_weight),
        "QLD trend / cash": weights(QLD=bull.astype(float)),
        "QLD trend + volatility / cash": weights(QLD=risk_weight),
    }


def main() -> None:
    root = Path(__file__).resolve().parent
    data = root / "data"
    output = root / "output" / "experiments"
    output.mkdir(parents=True, exist_ok=True)
    filenames = {"QQQ": "qqq_2005-01-01_2026-09-17.csv", "QLD": "qld_2005-01-01_2026-09-17.csv",
                 "TQQQ": "tqqq_2009-01-01_2026-09-17.csv"}
    assets = {s: pd.read_csv(data / f, index_col="date", parse_dates=True) for s, f in filenames.items()}
    for frame in assets.values():
        if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
            raise ValueError("Duplicate or unsorted dates.")
        if frame[["adj_open", "adj_close"]].isna().any().any() or (frame[["adj_open", "adj_close"]] <= 0).any().any():
            raise ValueError("Missing or nonpositive prices.")
    targets = build_targets(assets["QQQ"].adj_close)
    results = {name: simulate(assets, target) for name, target in targets.items()}
    summary = pd.DataFrame({name: metrics(r) for name, r in results.items()}).T
    summary.to_csv(output / "common_sample.csv", index_label="strategy")
    rows = []
    for start, end in [("2006-06-21", "2026-09-16"), ("2007-01-01", "2009-12-31"),
                       ("2018-01-01", "2026-09-16"), ("2022-01-01", "2026-09-16")]:
        for name, target in targets.items():
            if start < "2010" and "TQQQ" in name:
                continue
            universe = {s: frame.loc[start:end] for s, frame in assets.items() if start >= "2010" or s != "TQQQ"}
            r = simulate(universe, target)
            rows.append({"start": r.history.index[0].date(), "end": r.history.index[-1].date(),
                         "strategy": name, **metrics(r)})
    windows = pd.DataFrame(rows)
    windows.to_csv(output / "period_checks.csv", index=False)
    sensitivity = []
    for bps in (5.0, 25.0):
        for name in list(targets)[3:]:
            sensitivity.append({"strategy": name, "slippage_bps": bps, **metrics(simulate(assets, targets[name], slippage_bps=bps))})
    costs = pd.DataFrame(sensitivity)
    costs.to_csv(output / "cost_checks.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True, layout="constrained")
    colors = ["#252525", "#037c70", "#ba3434", "#774baa", "#1565b5", "#b58404"]
    for (name, result), color in zip(results.items(), colors):
        nav = result.history.equity
        axes[0].plot(nav.index, nav, label=name, color=color)
        dd = nav / nav.cummax().clip(lower=result.initial_cash) - 1
        axes[1].plot(dd.index, dd * 100, color=color)
        result.history.to_csv(output / (name.split(" / ")[0].replace(" ", "_").replace("%", "pct") + "_history.csv"))
    axes[0].set(yscale="log", ylabel="Value of initial $500 (log scale)", title="Historical strategy experiments: fixed $500, no deposits")
    axes[0].legend(fontsize=8, ncol=2)
    axes[1].set(ylabel="Drawdown (%)")
    for ax in axes:
        ax.grid(alpha=.2)
    fig.savefig(output / "comparison.png", dpi=150)
    plt.close(fig)

    lines = ["# Fixed-$500 strategy experiments", "", "Research run: 2026-09-17. No brokerage orders placed.", "",
             "## Common sample", "", "2010-02-11 to 2026-09-16. Initial $500, no subsequent deposits.", "",
             "| Strategy | CAGR | Max drawdown | Sharpe (0% rf) | 2020 return | 2022 return |", "|---|---:|---:|---:|---:|---:|"]
    for name, m in summary.iterrows():
        lines.append(f"| {name} | {m.cagr:.2%} | {m.max_drawdown:.2%} | {m.sharpe_0rf:.2f} | {m.return_2020:.2%} | {m.return_2022:.2%} |")
    lines += ["", "## Rules and methodology", "",
              "- QLD trend: QQQ adjusted close >= 1.04 x SMA200 enables 100% QLD; <= 0.97 x SMA200 switches to cash. Inside the band, preserve state.",
              "- QLD volatility variant: same trend filter; in bull periods target QLD weight = min(75%, 25% / (2 x QQQ trailing 63-session annualized volatility)); rest is cash. This is only an approximate volatility target, not a risk guarantee.",
              "- Prior 70% TQQQ strategy: bull 70% TQQQ + 30% QQQ; bear 100% QQQ; thresholds 1.02/0.98. Different thresholds are explicitly retained, not retuned.",
              "- Targets use completed prior-session signals, including first entry. Trade at next adjusted open on month changes or switches into/out of an asset; otherwise hold.",
              "- 5 bps one-way adverse slippage, fractional shares, zero commissions, no taxes, cash interest 0%. Expense drag is embedded in ETF market-price history. Adjusted prices approximate reinvested dividends.",
              "- Drawdown includes initial capital and costs, is measured at daily closes, and is not a future loss limit. Constant holdings have no external borrowing.",
              "- Cached Yahoo daily histories; no independent full-history reconciliation. No fabricated pre-inception TQQQ prices.",
              "- These are exploratory, previously observed periods, not untouched out-of-sample tests. No rule has been proved optimal. No single-stock backtest was used to justify today's stock selection.",
              "", "## Period checks", "", "| Dates | Strategy | CAGR | Max drawdown |", "|---|---|---:|---:|"]
    for r in windows.itertuples():
        lines.append(f"| {r.start} to {r.end} | {r.strategy} | {r.cagr:.2%} | {r.max_drawdown:.2%} |")
    lines += ["", "## Higher execution costs", "", "| Strategy | One-way bps | CAGR |", "|---|---:|---:|"]
    for r in costs.itertuples():
        lines.append(f"| {r.strategy} | {r.slippage_bps:.0f} | {r.cagr:.2%} |")
    lines += ["", "## Prior-report corrections", "",
              "The prior smart_dca simulation used same-day information for its first entry and treated opening deposits as ending deposits when calculating time-weighted returns. Both were corrected in smart_dca.py and its report regenerated. Previously quoted metrics should be treated as superseded exploratory estimates.", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))
    print(windows.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
