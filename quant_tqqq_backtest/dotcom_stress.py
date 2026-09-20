"""Counterfactual daily-reset Nasdaq-100 leverage, including the dot-com crash.

All modeled paths are synthetic, not pre-inception QLD/TQQQ fund histories.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


COLORS = {1: "#2563a6", 2: "#16836b", 3: "#cd454c"}


@dataclass(frozen=True)
class Costs:
    name: str
    fee_1x: float
    fee_levered: float
    financing_spread: float
    use_financing: bool


SCENARIOS = (
    Costs("ideal_no_cost", 0, 0, 0, False),
    Costs("illustrative_cost", .002, .0095, .005, True),
    Costs("higher_financing", .002, .0095, .015, True),
)


def load_fred(path: Path, column: str) -> pd.Series:
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    values = pd.to_numeric(frame[column], errors="coerce").dropna().sort_index()
    if not values.index.is_unique or not np.isfinite(values).all():
        raise ValueError(f"Invalid FRED series: {column}")
    return values


def financing_accrual(dates: pd.DatetimeIndex, annual_rates: pd.Series) -> pd.Series:
    calendar = pd.date_range(dates[0], dates[-1], freq="D")
    # Each calendar day's cost uses the prior day's available overnight rate.
    daily = annual_rates.reindex(calendar - pd.Timedelta(days=1), method="ffill")
    if daily.isna().any():
        raise ValueError("Missing financing rate history; no backfilling from the future.")
    integral = pd.Series(daily.to_numpy() / 365.0, index=calendar).cumsum()
    return integral.reindex(dates).diff().fillna(0)


def synthesize(index: pd.Series, annual_rates: pd.Series, costs: Costs) -> dict[int, pd.Series]:
    if len(index) < 2 or not index.index.is_unique or not index.index.is_monotonic_increasing:
        raise ValueError("Invalid market dates.")
    if (index <= 0).any() or not np.isfinite(index).all():
        raise ValueError("Index levels must be positive and finite.")
    returns = index.pct_change(fill_method=None).fillna(0)
    years = index.index.to_series().diff().dt.days.fillna(0) / 365.0
    financing = financing_accrual(index.index, annual_rates)
    paths = {}
    for leverage in (1, 2, 3):
        fee = costs.fee_1x if leverage == 1 else costs.fee_levered
        borrowed_cost = ((financing + costs.financing_spread * years) * (leverage - 1)
                         if costs.use_financing else 0)
        factor = 1 + leverage * returns - borrowed_cost - fee * years
        if (factor <= 0).any():
            raise ValueError("Daily NAV exhaustion: model must terminate, not resurrect the fund.")
        paths[leverage] = (100 * factor.cumprod()).rename(f"synthetic_{leverage}x")
    return paths


def simulate_close_dca(path: pd.Series, weekly: float = 100,
                       slippage_bps: float = 5, stop_after: pd.Timestamp | None = None) -> pd.DataFrame:
    if path.empty or weekly <= 0 or not 0 <= slippage_bps < 10000:
        raise ValueError("Invalid contributions or execution cost.")
    if not path.index.is_unique or not path.index.is_monotonic_increasing:
        raise ValueError("Dates must be unique and sorted.")
    if not np.isfinite(path).all() or (path <= 0).any():
        raise ValueError("Invalid synthetic NAV.")
    iso = path.index.isocalendar()
    first_in_week = (iso.year.ne(iso.year.shift(1)) | iso.week.ne(iso.week.shift(1))).fillna(True)
    deposits = first_in_week.astype(float) * weekly
    if stop_after is not None:
        deposits.loc[deposits.index > stop_after] = 0
    units = (deposits / (path * (1 + slippage_bps / 10000))).cumsum()
    history = pd.DataFrame({"synthetic_nav": path, "deposit": deposits, "deposited": deposits.cumsum(),
                            "equity": units * path})
    history["profit"] = history.equity - history.deposited
    history["profit_on_deposited"] = history.equity / history.deposited - 1
    history["initial_dollar_value"] = path / path.iloc[0] / (1 + slippage_bps / 10000)
    history["nav_drawdown"] = path / path.cummax() - 1
    return history


def first_nonnegative_after(series: pd.Series, after: pd.Timestamp) -> str | None:
    candidates = series.loc[after:]
    candidates = candidates[candidates >= -1e-8]
    return str(candidates.index[0].date()) if len(candidates) else None


def recovery_metrics(history: pd.DataFrame, anchor: pd.Timestamp) -> dict:
    underwater = history.profit < -1e-8
    loss_dates = history.index[underwater]
    last_loss = loss_dates[-1] if len(loss_dates) else None
    last_recovery = None
    if last_loss is not None and last_loss != history.index[-1]:
        last_recovery = history.index[history.index.get_loc(last_loss) + 1]
    longest_days, episode_start, longest_start, longest_end = 0, None, None, None
    for i, (date, below) in enumerate(underwater.items()):
        if below and episode_start is None:
            episode_start = date
        if episode_start is not None and (not below or i == len(history) - 1):
            days = (date - episode_start).days
            if days > longest_days:
                longest_days, longest_start = days, episode_start
                longest_end = None if below else date
            episode_start = None
    return {
        "first_dca_breakeven_after_2002_trough": first_nonnegative_after(history.profit, anchor),
        "last_dca_breakeven_in_sample": str(last_recovery.date()) if last_recovery is not None else None,
        "first_lump_breakeven_after_2002_trough": first_nonnegative_after(history.initial_dollar_value - 1, anchor),
        "last_lump_dollar_value": float(history.initial_dollar_value.iloc[-1]),
        "worst_dca_loss_fraction": float(history.profit_on_deposited.min()),
        "worst_dca_loss_date": str(history.profit_on_deposited.idxmin().date()),
        "max_synthetic_nav_drawdown": float(history.nav_drawdown.min()),
        "longest_underwater_years": longest_days / 365.2425,
        "longest_underwater_start": str(longest_start.date()) if longest_start is not None else None,
        "longest_underwater_end": str(longest_end.date()) if longest_end is not None else None,
    }


def window_metrics(history: pd.DataFrame) -> dict:
    return {"actual_start": str(history.index[0].date()), "actual_end": str(history.index[-1].date()),
            "contributed": float(history.deposited.iloc[-1]), "final_value": float(history.equity.iloc[-1]),
            "profit": float(history.profit.iloc[-1]), "roi": float(history.profit_on_deposited.iloc[-1]),
            "worst_loss_on_deposits": float(history.profit_on_deposited.min()),
            "lump_return": float(history.initial_dollar_value.iloc[-1] - 1),
            "max_nav_drawdown": float(history.nav_drawdown.min())}


def rolling_windows(paths: dict[int, pd.Series], index: pd.Series, costs: Costs) -> pd.DataFrame:
    records = []
    for years in (5, 10):
        starts = pd.date_range(index.index[0], index.index[-1] - pd.DateOffset(years=years), freq="MS")
        for start in starts:
            end = start + pd.DateOffset(years=years)
            benchmark = index.loc[start:end]
            elapsed = (benchmark.index[-1] - benchmark.index[0]).days / 365.2425
            annual_log_growth = np.log(benchmark.iloc[-1] / benchmark.iloc[0]) / elapsed
            volatility = benchmark.pct_change().std() * np.sqrt(252)
            last_two = benchmark.loc[benchmark.index[-1] - pd.DateOffset(years=2):]
            for leverage, path in paths.items():
                history = simulate_close_dca(path.loc[start:end])
                records.append({"scenario": costs.name, "years": years, "window_start": str(start.date()),
                                "leverage": leverage, "index_log_growth": annual_log_growth,
                                "index_volatility": volatility,
                                "index_last_two_year_return": last_two.iloc[-1] / last_two.iloc[0] - 1,
                                **window_metrics(history)})
    return pd.DataFrame(records)


def plot_recovery(histories: dict[int, pd.DataFrame], output: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), layout="constrained")
    for leverage, h in histories.items():
        early = h.loc[:"2015-03-27"]
        axes[0].plot(early.index, early.profit_on_deposited, color=COLORS[leverage], label=f"Synthetic {leverage}x")
        axes[1].plot(h.index, h.initial_dollar_value, color=COLORS[leverage])
    axes[0].axhline(0, color="#555555", linestyle="--", linewidth=1)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].set(ylabel="Profit / cumulative deposits", title="Start at the 2000-03-27 peak | $100/week | Illustrative costs")
    axes[0].legend(frameon=False, ncol=3)
    axes[1].axhline(1, color="#555555", linestyle="--", linewidth=1)
    axes[1].set(yscale="log", ylabel="Remaining value of the first $1 (log)", title="First deposit only: later contributions cannot repair this dollar")
    for ax in axes:
        ax.grid(axis="y", alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output / "dotcom_recovery.png", dpi=170)
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parent
    data = root / "data" / "dotcom_stress_2026-09-17"
    output = root / "output" / "dotcom_stress_2026-09-17"
    output.mkdir(parents=True, exist_ok=True)
    index = load_fred(data / "fred_NASDAQXNDX.csv", "NASDAQXNDX")
    rates = load_fred(data / "fred_DFF.csv", "DFF") / 100
    peak = index.loc["2000"].idxmax()
    trough = index.loc[peak:"2003-12-31"].idxmin()
    starts = {"2000_year_start": pd.Timestamp("2000-01-03"), "2000_peak": peak,
              "2002_trough": trough, "2007_year_start": pd.Timestamp("2007-01-03"),
              "2008_year_start": pd.Timestamp("2008-01-02"),
              "2010_year_start": pd.Timestamp("2010-01-04"),
              "2018_year_start": pd.Timestamp("2018-01-02"),
              "recent_five_years": pd.Timestamp("2021-09-16")}
    selected, recovery, rolling, calibration = [], [], [], []
    for costs in SCENARIOS:
        paths = synthesize(index, rates, costs)
        pd.DataFrame(paths).to_csv(output / f"synthetic_paths_{costs.name}.csv", index_label="date")
        for name, start in starts.items():
            for years in (5, 10):
                end = start + pd.DateOffset(years=years)
                if end > index.index[-1]:
                    continue
                for leverage, path in paths.items():
                    history = simulate_close_dca(path.loc[start:end])
                    selected.append({"scenario": costs.name, "start_label": name, "years": years,
                                     "leverage": leverage, **window_metrics(history)})
        histories = {}
        for leverage, path in paths.items():
            history = simulate_close_dca(path.loc[peak:])
            history.to_csv(output / f"peak_{costs.name}_{leverage}x.csv", index_label="date")
            histories[leverage] = history
            recovery.append({"scenario": costs.name, "leverage": leverage,
                             **recovery_metrics(history, trough)})
        if costs.name == "illustrative_cost":
            plot_recovery(histories, output)
        rolling.append(rolling_windows(paths, index, costs))
        for leverage, symbol in ((1, "QQQ"), (2, "QLD"), (3, "TQQQ")):
            actual = pd.read_csv(root / "data" / "five_year_dca_2026-09-17" /
                                 f"{symbol.lower()}_2010-01-01_2026-09-17.csv", index_col="date", parse_dates=True).adj_close
            common = actual.index.intersection(paths[leverage].index)
            modeled = paths[leverage].loc[common]
            actual = actual.loc[common]
            years = (common[-1] - common[0]).days / 365.2425
            calibration.append({"scenario": costs.name, "symbol": symbol, "start": str(common[0].date()),
                                "end": str(common[-1].date()),
                                "actual_cagr": (actual.iloc[-1] / actual.iloc[0]) ** (1 / years) - 1,
                                "model_cagr": (modeled.iloc[-1] / modeled.iloc[0]) ** (1 / years) - 1,
                                "daily_return_correlation": actual.pct_change().corr(modeled.pct_change())})
    selected = pd.DataFrame(selected)
    recovery = pd.DataFrame(recovery)
    rolling = pd.concat(rolling, ignore_index=True)
    calibration = pd.DataFrame(calibration)
    for name, frame in (("selected_windows", selected), ("recovery", recovery),
                        ("rolling_windows", rolling), ("model_vs_actual", calibration)):
        frame.to_csv(output / f"{name}.csv", index=False)

    aggregates = []
    for (scenario, years), group in rolling.groupby(["scenario", "years"]):
        table = group.pivot(index="window_start", columns="leverage", values="final_value")
        winners = table.idxmax(axis=1).value_counts()
        for leverage, records in group.groupby("leverage"):
            worst = records.loc[records.roi.idxmin()]
            aggregates.append({"scenario": scenario, "years": years, "leverage": leverage,
                               "windows": len(records), "winner_count": int(winners.get(leverage, 0)),
                               "end_below_cost_count": int((records.profit < -1e-8).sum()),
                               "worst_roi": worst.roi, "worst_start": worst.actual_start,
                               "worst_end": worst.actual_end, "worst_contributed": worst.contributed,
                               "worst_final_value": worst.final_value})
    aggregates = pd.DataFrame(aggregates)
    aggregates.to_csv(output / "window_summary.csv", index=False)
    conditions = []
    baseline = rolling[rolling.scenario.eq("illustrative_cost")]
    for years, group in baseline.groupby("years"):
        table = group.pivot(index="window_start", columns="leverage", values="final_value")
        base = group[group.leverage.eq(1)].set_index("window_start")
        for wins in (True, False):
            chosen = base.loc[table.idxmax(axis=1).eq(3).eq(wins)]
            conditions.append({"years": years, "three_x_wins": wins, "windows": len(chosen),
                               **{f"median_{c}": float(chosen[c].median()) for c in
                                  ("index_log_growth", "index_volatility", "index_last_two_year_return")}})
    pd.DataFrame(conditions).to_csv(output / "winning_window_conditions.csv", index=False)

    provenance = {"created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                  "index": "FRED NASDAQXNDX; Nasdaq-100 total-return index, not Nasdaq Composite",
                  "start": str(index.index[0].date()), "end": str(index.index[-1].date()),
                  "peak": str(peak.date()), "post_crash_trough": str(trough.date()),
                  "sources": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                              (data / "fred_NASDAQXNDX.csv", data / "fred_DFF.csv")}}
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    write_report(selected, recovery, aggregates, calibration, output, peak, trough, index)
    print("PEAK START FIVE AND TEN YEARS\n", selected[selected.start_label.eq("2000_peak")].to_string(index=False))
    print("\nRECOVERY\n", recovery.to_string(index=False))
    print("\nROLLING WINDOWS\n", aggregates.to_string(index=False))
    print("\nMODEL VALIDATION\n", calibration.to_string(index=False))


def write_report(selected, recovery, aggregates, calibration, output, peak, trough, index):
    lines = ["# Dot-com crisis: synthetic daily 1x / 2x / 3x Nasdaq-100 DCA", "",
             "IMPORTANT: counterfactual simulations, NOT actual historical QLD/TQQQ returns before inception.", "",
             f"Index data: {index.index[0].date()} to {index.index[-1].date()}. Stress start: {peak.date()}, the retrospectively selected 2000 closing peak. Post-crash trough anchor: {trough.date()}.", "",
             "## Fixed assumptions", "",
             "- $100 each ISO week's first trading day at the CLOSE, including the first partial week. No initial lump sum. This differs from the previous real-ETF test's opening executions; figures must not be spliced together.",
             "- Deterministic schedule: new money receives no earlier same-day gain. Fractional synthetic units; 5 bps purchase slippage; no taxes or final liquidation cost. No tactical switches, stops or hindsight trading signals.",
             "- Benchmark is the Nasdaq-100 TOTAL RETURN index (NASDAQXNDX), including dividends, not the Nasdaq Composite. FRED distributes Nasdaq data; any index back-history is not a tradable fund record.",
             "- Daily factor = 1 + leverage * daily_index_return - financing_cost - fund_fee. Compound each day; NEVER multiply a five-year cumulative return by 2 or 3.",
             "- Ideal: zero fund/financing costs, still with purchase slippage. Illustrative cost: annual fees 0.20% for 1x, 0.95% for 2x/3x; borrowed notional (L-1) charged historical effective federal funds rate + 0.50% spread. Higher financing: spread 1.50%. These are declared scenarios, not calibrated replicas or actual historical swap contracts.",
             "- Accrue financing on calendar days using prior-day rates, actual/365; include weekends and holidays. Daily rebalancing occurs at closes. Financing, rebalancing, and intraday exposures of actual ETFs may differ.",
             "- Assume the hypothetical fund survives, remains liquid, permits ongoing investment, and executes daily leverage throughout. Fund closure, derivatives counterparty failures, changing regulations, taxes, trading halts, and intraday path-dependent liquidation are NOT modeled. A nonpositive daily modeled NAV factor raises an error rather than resurrecting capital.",
             "- Breakeven is nominal account value >= cumulative cash contributed. It is NOT inflation-adjusted, risk-adjusted, or equivalent to outperforming 1x. No withdrawals or contribution interruptions are assumed.",
             "", "## Starting at the 2000 peak", "",
             "| Cost scenario | Years | Deposited | 1x final | 2x final | 3x final |", "|---|---:|---:|---:|---:|---:|"]
    for (scenario, years), group in selected[selected.start_label.eq("2000_peak")].groupby(["scenario", "years"], sort=False):
        values = group.set_index("leverage")
        lines.append(f"| {scenario} | {years} | ${group.iloc[0].contributed:,.0f} | ${values.at[1, 'final_value']:,.0f} | ${values.at[2, 'final_value']:,.0f} | ${values.at[3, 'final_value']:,.0f} |")
    lines += ["", "## Recovery definitions", "",
              "First recovery is searched AFTER the 2002 market trough, to avoid mistaking a brief early bounce for surviving the crisis. Last recovery is retrospective: the date after the final observed loss versus deposits, through the sample end, NOT a guarantee of future safety.", "",
              "| Scenario | Leverage | DCA first recovery after trough | Last observed DCA recovery | First dollar recovery after trough | First dollar value at sample end | Worst DCA loss / deposits | Max NAV drawdown |",
              "|---|---:|---|---|---|---:|---:|---:|"]
    for r in recovery.itertuples():
        lines.append(f"| {r.scenario} | {r.leverage}x | {r.first_dca_breakeven_after_2002_trough or 'Not recovered'} | {r.last_dca_breakeven_in_sample or 'Not recovered'} | {r.first_lump_breakeven_after_2002_trough or 'Not recovered'} | ${r.last_lump_dollar_value:.5f} | {r.worst_dca_loss_fraction:.2%} | {r.max_synthetic_nav_drawdown:.3%} |")
    lines += ["", "## Rolling monthly-start windows", "",
              "Five-year and ten-year windows overlap heavily. Counts describe the historical sample; they are not independent observations or future probabilities. The sample includes two major crises but not every possible future market path.", "",
              "| Scenario | Years | Leverage | Windows | Highest final value | Below deposits at end | Worst loss / deposits | Worst dates |",
              "|---|---:|---:|---:|---:|---:|---:|---|"]
    for r in aggregates.itertuples():
        lines.append(f"| {r.scenario} | {r.years} | {r.leverage}x | {r.windows} | {r.winner_count} | {r.end_below_cost_count} | {r.worst_roi:.2%} | {r.worst_start} to {r.worst_end} |")
    lines += ["", "## Model compared with real funds after 2010", "",
              "Not fitted to the fund results. Comparison is close-to-close with reinvested distributions, no contribution schedule; differences quantify model limitations and can materially affect long-horizon recovery dates.", "",
              "| Scenario | ETF | Actual CAGR | Model CAGR | Daily return correlation |", "|---|---|---:|---:|---:|"]
    for r in calibration.itertuples():
        lines.append(f"| {r.scenario} | {r.symbol} | {r.actual_cagr:.2%} | {r.model_cagr:.2%} | {r.daily_return_correlation:.5f} |")
    lines += ["", "## Interpretation", "",
              "3x is helped by strong sustained growth, lower volatility and financing costs, and a large recovery after early contributions acquired cheap units. Large late drawdowns hit a much larger accumulated balance; five or ten years is not a guaranteed recovery horizon.",
              "DCA account recovery can coexist with the first deposit remaining almost worthless. Later money bought at lower prices is earning the recovery; the original purchase has not magically healed.",
              "", "## Sources", "",
              "- [Nasdaq-100 total return via FRED](https://fred.stlouisfed.org/series/NASDAQXNDX)",
              "- [Historical effective federal funds rate](https://fred.stlouisfed.org/series/DFF)",
              "- [Nasdaq index description and back-history caution](https://indexes.nasdaq.com/Index/Overview/XNDX)",
              "- [TQQQ daily objective and risk](https://www.proshares.com/our-etfs/leveraged-and-inverse/tqqq)",
              "- [QLD daily objective and risk](https://www.proshares.com/our-etfs/leveraged-and-inverse/qld)", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
