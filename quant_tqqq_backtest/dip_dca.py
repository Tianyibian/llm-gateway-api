"""Compare weekly investing with waiting for a prior-day QQQ decline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from quant_tqqq_backtest.dotcom_stress import load_fred


STRATEGIES = ("weekly_qqq", "weekly_qld", "weekly_tqqq", "weekly_half_half",
              "dip_cash", "dip_bond")
LABELS = {"weekly_qqq": "Weekly QQQ", "weekly_qld": "Weekly QLD",
          "weekly_tqqq": "Weekly TQQQ", "weekly_half_half": "Weekly 50/50, no rebalance",
          "dip_cash": "Wait for dip, idle cash", "dip_bond": "Wait for dip, idle short Treasuries"}


def prior_close_signal(qqq_close: pd.Series, threshold: float = -.01) -> pd.Series:
    if not np.isfinite(threshold) or not -1 < threshold < 0:
        raise ValueError("Dip threshold must be a finite negative return greater than -100%.")
    returns = qqq_close.pct_change(fill_method=None)
    return returns.le(threshold + 1e-12).shift(1, fill_value=False)


def treasury_carry(dates: pd.DatetimeIndex, rates: pd.Series, annual_fee: float = .001) -> pd.Series:
    # This is yield carry, not a marked-to-market bond total-return index.
    calendar = pd.date_range(dates[0], dates[-1], freq="D")
    observation_dates = calendar - pd.offsets.BDay(2)
    available = rates.reindex(observation_dates, method="ffill")
    if available.isna().any():
        raise ValueError("Missing lagged Treasury yield observations.")
    daily = pd.Series((available.to_numpy() - annual_fee) / 365., index=calendar)
    integral = daily.cumsum().reindex(dates)
    return (100. * np.exp(integral - integral.iloc[0])).rename("bond")


def simulate(execution: pd.DataFrame, closes: pd.DataFrame, signal: pd.Series,
             strategy: str, weekly: float = 100., slippage_bps: float = 5.) -> pd.DataFrame:
    required = ["qqq", "qld", "tqqq", "bond"]
    if strategy not in STRATEGIES or execution.empty or not execution.index.equals(closes.index):
        raise ValueError("Invalid strategy or calendars.")
    if not execution.index.is_unique or not execution.index.is_monotonic_increasing:
        raise ValueError("Dates must be unique and sorted.")
    if not np.isfinite([weekly, slippage_bps]).all() or weekly <= 0 or not 0 <= slippage_bps < 10000:
        raise ValueError("Invalid contribution or execution costs.")
    for frame in (execution, closes):
        values = frame[required].to_numpy()
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError("Invalid asset prices.")
    active = signal.reindex(execution.index)
    if active.isna().any():
        raise ValueError("Missing signal observations.")
    iso = execution.index.isocalendar()
    first_in_week = (iso.year.ne(iso.year.shift()) | iso.week.ne(iso.week.shift())).fillna(True)
    deposits = first_in_week.to_numpy(dtype=float) * weekly
    units = np.zeros(4)
    cash, contributed = 0., 0.
    slip = slippage_bps / 10000.
    records = []
    waiting_lots: list[tuple[int, float]] = []
    static_weights = {"weekly_qqq": [1, 0, 0, 0], "weekly_qld": [0, 1, 0, 0],
                      "weekly_tqqq": [0, 0, 1, 0], "weekly_half_half": [.5, 0, .5, 0]}
    prices, marks = execution[required].to_numpy(), closes[required].to_numpy()
    for i, (date, price, mark, deposit, dip) in enumerate(zip(execution.index, prices, marks, deposits, active)):
        cash += deposit
        contributed += deposit
        if deposit and strategy.startswith("dip"):
            waiting_lots.append((i, deposit))
        cost, turnover, buy_count, sell_count = 0., 0., 0, 0
        deployed_principal, principal_wait_days = 0., 0.
        if strategy in static_weights:
            weights = np.array(static_weights[strategy])
            amounts = cash * weights / (1 + slip)
            units += amounts / price
            turnover += amounts.sum()
            cost += amounts.sum() * slip
            buy_count += int((amounts > 1e-10).sum())
            cash = 0.
        elif dip:
            if units[3] > 0:
                gross = units[3] * price[3]
                cash += gross * (1 - slip)
                cost += gross * slip
                turnover += gross
                units[3] = 0.
                sell_count += 1
            if cash > 1e-10:
                amount = cash / (1 + slip)
                units[2] += amount / price[2]
                cost += amount * slip
                turnover += amount
                buy_count += 1
                cash = 0.
                for origin, principal in waiting_lots:
                    deployed_principal += principal
                    principal_wait_days += principal * (date - execution.index[origin]).days
                waiting_lots.clear()
        elif strategy == "dip_bond" and cash > 1e-10:
            amount = cash / (1 + slip)
            units[3] += amount / price[3]
            cost += amount * slip
            turnover += amount
            buy_count += 1
            cash = 0.
        value = float(units @ mark + cash)
        records.append((deposit, contributed, value, units[2] * mark[2], units[3] * mark[3], cash,
                        cost, turnover, buy_count, sell_count, units[2],
                        deployed_principal, principal_wait_days))
    h = pd.DataFrame(records, index=execution.index, columns=[
        "deposit", "deposited", "equity", "tqqq_value", "bond_value", "cash",
        "transaction_cost", "turnover", "buy_count", "sell_count", "tqqq_units",
        "deployed_principal", "principal_wait_days"])
    h["profit"] = h.equity - h.deposited
    h["roi"] = h.profit / h.deposited
    return h


def metrics(h: pd.DataFrame) -> dict:
    worst_date = h.roi.idxmin()
    worst = h.loc[worst_date]
    last = h.iloc[-1]
    deployed = h.deployed_principal.sum()
    return {"actual_start": str(h.index[0].date()), "actual_end": str(h.index[-1].date()),
            "contributed": float(last.deposited), "final_value": float(last.equity),
            "profit": float(last.profit), "end_roi": float(last.roi),
            "worst_interim_roi": float(worst.roi), "worst_date": str(worst_date.date()),
            "deposited_at_worst": float(worst.deposited), "equity_at_worst": float(worst.equity),
            "ending_tqqq_weight": float(last.tqqq_value / last.equity),
            "mean_idle_weight": float(((h.bond_value + h.cash) / h.equity).mean()),
            "transaction_cost": float(h.transaction_cost.sum()),
            "turnover": float(h.turnover.sum()),
            "trades": int(h.buy_count.sum() + h.sell_count.sum()),
            "mean_calendar_wait_deployed_principal": float(h.principal_wait_days.sum() / deployed) if deployed else None,
            "deployed_principal": float(deployed)}


def read_prices(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col="date", parse_dates=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dip-percent", type=float, default=1.,
                        help="Minimum prior-session percentage decline, expressed positively (e.g. 1.5).")
    args = parser.parse_args()
    if not np.isfinite(args.dip_percent) or not 0 < args.dip_percent < 100:
        parser.error("--dip-percent must be greater than zero and less than 100.")
    threshold = -args.dip_percent / 100.
    suffix = "" if args.dip_percent == 1. else "_" + f"{args.dip_percent:g}".replace(".", "p") + "pct"
    root = Path(__file__).resolve().parent
    out = root / "output" / f"dip_dca{suffix}_2026-09-17"
    out.mkdir(parents=True, exist_ok=True)
    cache = root / "data" / "dip_dca_2026-09-17"
    actual_files = {s: root / "data" / "five_year_dca_2026-09-17" / f"{s}_2010-01-01_2026-09-17.csv"
                    for s in ("qqq", "qld", "tqqq")}
    actual_files["bond"] = cache / "sgov_2020-01-01_2026-09-17.csv"
    assets = {s: read_prices(p) for s, p in actual_files.items()}
    common = assets["qqq"].index
    for a in assets.values():
        common = common.intersection(a.index)
    closes = pd.DataFrame({s: a.adj_close.reindex(common) for s, a in assets.items()})
    opens = pd.DataFrame({s: a.adj_open.reindex(common) for s, a in assets.items()})
    signal = prior_close_signal(assets["qqq"].adj_close, threshold)
    end = common[-1]
    start = end - pd.DateOffset(years=5)
    actual_records = []
    for timing, execution in (("next_open", opens), ("next_close", closes)):
        for strategy in STRATEGIES:
            h = simulate(execution.loc[start:end], closes.loc[start:end], signal, strategy)
            h.to_csv(out / f"actual_{timing}_{strategy}.csv", index_label="date")
            actual_records.append({"execution": timing, "strategy": strategy, **metrics(h)})
    actual = pd.DataFrame(actual_records)
    actual.to_csv(out / "actual_five_year_summary.csv", index=False)

    source = root / "output" / "dotcom_stress_2026-09-17"
    path_file = source / "synthetic_paths_illustrative_cost.csv"
    paths = read_prices(path_file).rename(columns={"1": "qqq", "2": "qld", "3": "tqqq"})
    rates_file = cache / "fred_DGS3MO.csv"
    paths["bond"] = treasury_carry(paths.index, load_fred(rates_file, "DGS3MO") / 100.)
    full_qqq_file = root / "data" / "dotcom_stress_2026-09-17" / "qqq_1985-01-01_2026-09-17.csv"
    historical_signal = prior_close_signal(read_prices(full_qqq_file).adj_close, threshold)
    windows = []
    for years in (5, 10):
        for start in pd.date_range(paths.index[0], paths.index[-1] - pd.DateOffset(years=years), freq="MS"):
            window = paths.loc[start:start + pd.DateOffset(years=years)]
            for strategy in STRATEGIES:
                h = simulate(window, window, historical_signal, strategy)
                windows.append({"years": years, "window_start": str(start.date()), "strategy": strategy, **metrics(h)})
    windows = pd.DataFrame(windows)
    windows.to_csv(out / "synthetic_rolling_windows.csv", index=False)
    summary = []
    for (years, strategy), group in windows.groupby(["years", "strategy"], sort=False):
        baseline = windows[windows.years.eq(years) & windows.strategy.eq("weekly_tqqq")].set_index("window_start")
        ratio = group.set_index("window_start").final_value / baseline.final_value
        worst = group.loc[group.end_roi.idxmin()]
        best = group.loc[group.end_roi.idxmax()]
        interim = group.loc[group.worst_interim_roi.idxmin()]
        summary.append({"years": years, "strategy": strategy, "windows": len(group),
                        "worst_end_roi": group.end_roi.min(), "median_end_roi": group.end_roi.median(),
                        "best_end_roi": group.end_roi.max(), "ending_loss_count": int(group.end_roi.lt(0).sum()),
                        "worst_interim_roi": group.worst_interim_roi.min(),
                        "wins_vs_weekly_tqqq": int(ratio.gt(1 + 1e-10).sum()),
                        "median_wealth_ratio_vs_weekly_tqqq": ratio.median(),
                        "worst_window_start": worst.actual_start, "worst_window_end": worst.actual_end,
                        "worst_final_value": worst.final_value, "worst_contributed": worst.contributed,
                        "best_window_start": best.actual_start, "best_window_end": best.actual_end,
                        "best_final_value": best.final_value, "best_contributed": best.contributed,
                        "interim_worst_window_start": interim.actual_start, "interim_worst_date": interim.worst_date,
                        "interim_worst_contributed": interim.deposited_at_worst, "interim_worst_equity": interim.equity_at_worst})
    summary = pd.DataFrame(summary)
    summary.to_csv(out / "synthetic_rolling_summary.csv", index=False)

    # Re-run the extreme windows to keep the result auditable at daily resolution.
    for row in summary[summary.strategy.isin(["weekly_tqqq", "dip_bond"])].itertuples():
        for kind, date in (("worst_end", row.worst_window_start), ("best_end", row.best_window_start),
                           ("worst_interim", row.interim_worst_window_start)):
            start = pd.Timestamp(date).replace(day=1)
            w = paths.loc[start:start + pd.DateOffset(years=int(row.years))]
            simulate(w, w, historical_signal, row.strategy).to_csv(
                out / f"synthetic_{row.years}y_{row.strategy}_{kind}.csv", index_label="date")
    source_files = [*actual_files.values(), path_file, rates_file, full_qqq_file]
    provenance = {"created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                  "strategy_status": "User-confirmed: only idle/new money waits; existing TQQQ is never sold.",
                  "threshold": threshold, "weekly_deposit": 100, "slippage_bps_each_side": 5,
                  "actual_execution": f"Next session open after adjusted QQQ close-to-close loss >= {args.dip_percent:g}%",
                  "synthetic_execution": "Next session close, because reliable synthetic intraday prices are unavailable",
                  "treasury_proxy": "Prior available 3-month Treasury yield, lagged 2 business days, actual/365 carry less 0.10% pa; no bond price change modeling",
                  "sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}}
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    report(actual, summary, out, args.dip_percent)
    print("ACTUAL FIVE YEARS\n" + actual.to_string(index=False))
    print("\nSYNTHETIC ROLLING\n" + summary.to_string(index=False))


def report(actual: pd.DataFrame, summary: pd.DataFrame, out: Path, dip_percent: float = 1.) -> None:
    lines = [f"# Waiting for a {dip_percent:g}% dip versus weekly investing", "",
             "User-confirmed interpretation: only new/idle money waits; previously purchased TQQQ is never sold. This is NOT daily all-account switching and is NOT the earlier 200-day trend strategy.", "",
             "## Rules", "",
             "- Deposit $100 at the first session of every ISO week, including a partial starting week. No initial lump sum, borrowing, withdrawals, or trading authorization.",
             f"- Signal: QQQ split/dividend-adjusted close-to-close return <= -{dip_percent:g}%, observed before the execution session. The preceding session may be before the window starts.",
             "- At the execution time, if the prior session triggered: sell all idle bond units, use the proceeds plus available deposits to buy TQQQ. Hold existing TQQQ indefinitely.",
             "- Otherwise, buy short Treasury units with new deposits (or retain 0%-yield cash in the control). Continue weekly saving even while waiting.",
             "- Fractional adjusted-price units approximate reinvested distributions. Charge 5 bps on each buy and sell. No taxes, commissions, terminal liquidation, or settlement constraints modeled.",
             "- Risk measures below are nominal equity/cumulative contributions minus one. They are NOT drawdowns, annualized returns, or inflation-adjusted wealth.", "",
             "## Recent five years: actual ETF data", "",
             "QQQ, QLD, TQQQ, and SGOV adjusted historical prices. Primary execution is the next session OPEN, not the signal close. A next-session CLOSE sensitivity is also provided.", "",
             "| Execution | Strategy | Dates | Contributed | Ending value | End ROI | Worst interim ROI | Mean deployed-dollar wait (calendar days) |",
             "|---|---|---|---:|---:|---:|---:|---:|"]
    for r in actual.itertuples():
        wait = f"{r.mean_calendar_wait_deployed_principal:.1f}" if pd.notna(r.mean_calendar_wait_deployed_principal) else "N/A"
        lines.append(f"| {r.execution} | {LABELS[r.strategy]} | {r.actual_start} to {r.actual_end} | ${r.contributed:,.0f} | ${r.final_value:,.0f} | {r.end_roi:.2%} | {r.worst_interim_roi:.2%} | {wait} |")
    lines += ["", "## Including 2000: synthetic stress tests", "",
              "These are NOT actual pre-inception fund returns. Reuse the prior daily-reset Nasdaq-100 total-return simulation with 0.20% 1x fees; 0.95% 2x/3x fees; historical fed funds + 0.50% financing on extra exposure. Funds are assumed to survive and trade continuously.",
              "Historical QQQ supplies the signal. All strategies execute at the next session CLOSE for comparability because the synthetic funds have no reliable opening prices. Waiting assets are a yield-carry PROXY using lagged DGS3MO less 0.10% annual costs, not actual SGOV or a bond total-return index. No bond price sensitivity is modeled. Compare the zero-yield cash control to see whether this approximation drives results.",
              "Monthly starting dates, complete five-/ten-year windows. Windows overlap and are not independent observations or probabilities. Each best/worst is selected retrospectively, not a future bound.", "",
              "| Years | Strategy | Worst end ROI | Median end ROI | Best end ROI | Worst interim ROI | Wins versus weekly TQQQ | Median matched wealth ratio |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in summary.itertuples():
        lines.append(f"| {r.years} | {LABELS[r.strategy]} | {r.worst_end_roi:.2%} | {r.median_end_roi:.2%} | {r.best_end_roi:.2%} | {r.worst_interim_roi:.2%} | {r.wins_vs_weekly_tqqq}/{r.windows} | {r.median_wealth_ratio_vs_weekly_tqqq:.4f}x |")
    lines += ["", "## Interpretation limits", "",
              f"- A {dip_percent:g}% daily loss is not a valuation, bottom, or uptrend signal. Prices may first rise more than the later dip; existing positions remain exposed throughout bear markets.",
              "- Equal external deposits isolate entry timing. New contributions can reduce the loss percentage mechanically without recovering earlier investment losses; do not confuse that with an investment return.",
              "- New deposits shrink relative to accumulated assets as the account grows. Idle Treasury holdings do not protect TQQQ already purchased; there is no enduring bond allocation or loss cap.",
              "- Results assume uninterrupted deposits, no withdrawals, immediate tradability of sale proceeds, and no account-level cash settlement restrictions. Taxable bond income and sales are not modeled.",
              f"- This run uses the user-specified fixed {dip_percent:g}% threshold. Comparing thresholds on the same history is exploratory, not out-of-sample validation. Previous strategy exploration and choice of Nasdaq exposure introduce selection risk; historical fit is not validation for live execution.",
              "", "Sources:", "",
              "- [TQQQ daily leverage objective](https://www.proshares.com/our-etfs/leveraged-and-inverse/tqqq)",
              "- [SGOV short Treasury mandate and inception](https://www.ishares.com/us/products/314116/ishares-0-3-month-treasury-bond-etf)",
              "- [FRED Treasury yield, NOT a total-return series](https://fred.stlouisfed.org/series/DGS3MO)",
              "- [Nasdaq-100 total return](https://fred.stlouisfed.org/series/NASDAQXNDX)",
              "- Actual adjusted prices use saved Yahoo historical data; source hashes are in provenance.json.", ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
