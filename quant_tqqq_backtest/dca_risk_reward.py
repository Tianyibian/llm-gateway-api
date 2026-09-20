"""Contribution sizing, downside distributions, and paired DCA upside ratios."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from quant_tqqq_backtest.dotcom_stress import simulate_close_dca


ALLOCATIONS = {
    "100_QQQ": ({1: 1.0}, 100.),
    "100_QLD": ({2: 1.0}, 100.),
    "100_TQQQ": ({3: 1.0}, 100.),
    "50_TQQQ_only": ({3: 1.0}, 50.),
    "50_TQQQ_50_QQQ": ({3: .5, 1: .5}, 100.),
    "50_50_monthly_rebalance": ({3: .5, 1: .5}, 100.),
    "50_TQQQ_50_cash": ({3: .5, 0: .5}, 100.),
    "30_TQQQ_70_cash": ({3: .3, 0: .7}, 100.),
}


def combine_sleeves(histories: dict[int, pd.DataFrame], weights: dict[int, float], weekly: float) -> pd.DataFrame:
    if weekly <= 0 or any(not np.isfinite(w) or w < 0 for w in weights.values()) or not np.isclose(sum(weights.values()), 1):
        raise ValueError("Require positive budget and nonnegative weights summing to one.")
    if set(weights) - {0, 1, 2, 3}:
        raise ValueError("Unknown sleeve.")
    reference = histories[1]
    for history in histories.values():
        if not reference.index.equals(history.index) or not reference.deposit.equals(history.deposit):
            raise ValueError("Sleeves must have identical dates and external deposits.")
    scale = weekly / 100.
    equity = reference.deposited * weights.get(0, 0.)
    for leverage in (1, 2, 3):
        equity = equity + histories[leverage].equity * weights.get(leverage, 0.)
    result = pd.DataFrame({"equity": equity * scale, "deposited": reference.deposited * scale})
    result["profit"] = result.equity - result.deposited
    result["roi"] = result.equity / result.deposited - 1
    result["tqqq_market_weight"] = histories[3].equity * weights.get(3, 0.) * scale / result.equity
    return result


def simulate_monthly_mix(qqq: pd.Series, tqqq: pd.Series, weekly: float = 100., slippage_bps: float = 5.) -> pd.DataFrame:
    if not qqq.index.equals(tqqq.index):
        raise ValueError("Asset calendars must match.")
    template = simulate_close_dca(qqq, weekly, slippage_bps)
    if not np.isfinite(tqqq).all() or (tqqq <= 0).any():
        raise ValueError("Invalid TQQQ prices.")
    q_units, t_units, cash = 0., 0., 0.
    previous_month = None
    equity, weights, costs, cash_values = [], [], [], []
    slip = slippage_bps / 10000
    for date, qp, tp, deposit in zip(qqq.index, qqq.to_numpy(), tqqq.to_numpy(), template.deposit.to_numpy()):
        cash += deposit
        month = (date.year, date.month)
        qv, tv = q_units * qp, t_units * tp
        cost = 0.
        if month != previous_month:
            pretrade = qv + tv + cash
            # Solve the fully funded post-cost target without borrowing for fees.
            low, high = 0., pretrade
            for _ in range(45):
                net = (low + high) / 2
                cost = slip * (abs(.5 * net - qv) + abs(.5 * net - tv))
                if net + cost > pretrade:
                    high = net
                else:
                    low = net
            net = low
            cost = slip * (abs(.5 * net - qv) + abs(.5 * net - tv))
            q_units, t_units = .5 * net / qp, .5 * net / tp
            cash = max(0., pretrade - net - cost)
        elif cash > 0:
            budget = cash
            q_units += .5 * budget / (qp * (1 + slip))
            t_units += .5 * budget / (tp * (1 + slip))
            cost = budget * slip / (1 + slip)
            cash = 0.
        total = q_units * qp + t_units * tp + cash
        equity.append(total)
        weights.append(t_units * tp / total)
        costs.append(cost)
        cash_values.append(cash)
        previous_month = month
    result = pd.DataFrame({"equity": equity, "deposited": template.deposited,
                           "tqqq_market_weight": weights, "transaction_cost": costs, "cash": cash_values}, index=qqq.index)
    result["profit"] = result.equity - result.deposited
    result["roi"] = result.equity / result.deposited - 1
    return result


def main() -> None:
    root = Path(__file__).resolve().parent
    source = root / "output" / "dotcom_stress_2026-09-17"
    output = root / "output" / "dca_risk_reward_2026-09-17"
    output.mkdir(parents=True, exist_ok=True)
    original = pd.read_csv(source / "rolling_windows.csv")
    original = original[original.scenario.eq("illustrative_cost")].copy()
    paths = pd.read_csv(source / "synthetic_paths_illustrative_cost.csv", index_col="date", parse_dates=True)
    downside, ratios, endpoints = [], [], []
    for years, group in original.groupby("years"):
        table = group.pivot(index="window_start", columns="leverage", values="final_value")
        tqqq = group[group.leverage.eq(3)]
        losing = tqqq[tqqq.roi < 0]
        downside.append({"years": years, "windows": len(tqqq), "negative_windows": len(losing),
                         "loss_median_conditional": losing.roi.median(), "worst_end_loss": tqqq.roi.min(),
                         "worst_interim_loss": tqqq.worst_loss_on_deposits.min(),
                         "end_loss_over_30_count": int((tqqq.roi < -.3).sum())})
        for benchmark in (1, 2):
            paired = table[3] / table[benchmark]
            ratios.append({"years": years, "benchmark_leverage": benchmark, "minimum": paired.min(),
                           "p10": paired.quantile(.1), "median": paired.median(),
                           "p90": paired.quantile(.9), "maximum": paired.max(),
                           "half_budget_exceeds_full_benchmark": int((paired * .5 > 1).sum())})
        for start in table.index:
            end = pd.Timestamp(start) + pd.DateOffset(years=int(years))
            histories = {lev: simulate_close_dca(paths[str(lev)].loc[start:end]) for lev in (1, 2, 3)}
            for name, (weights, weekly) in ALLOCATIONS.items():
                if name == "50_50_monthly_rebalance":
                    combined = simulate_monthly_mix(paths["1"].loc[start:end], paths["3"].loc[start:end])
                else:
                    combined = combine_sleeves(histories, weights, weekly)
                last = combined.iloc[-1]
                endpoints.append({"years": years, "window_start": start,
                                  "actual_start": str(combined.index[0].date()),
                                  "actual_end": str(combined.index[-1].date()),
                                  "allocation": name, "weekly_budget": weekly,
                                  "contributed": last.deposited, "final_value": last.equity,
                                  "profit": last.profit, "roi": last.roi,
                                  "worst_interim_loss": combined.roi.min(),
                                  "terminal_tqqq_market_weight": last.tqqq_market_weight,
                                  "maximum_tqqq_market_weight": combined.tqqq_market_weight.max()})
    downside = pd.DataFrame(downside)
    ratios = pd.DataFrame(ratios)
    endpoints = pd.DataFrame(endpoints)
    summaries = []
    for (years, allocation), g in endpoints.groupby(["years", "allocation"]):
        worst = g.loc[g.roi.idxmin()]
        summaries.append({"years": years, "allocation": allocation, "windows": len(g),
                          "worst_end_roi": g.roi.min(), "median_end_roi": g.roi.median(),
                          "p10_end_roi": g.roi.quantile(.1), "p90_end_roi": g.roi.quantile(.9),
                          "best_end_roi": g.roi.max(), "worst_interim_loss": g.worst_interim_loss.min(),
                          "end_loss_over_30_count": int((g.roi < -.3).sum()),
                          "negative_end_count": int((g.roi < 0).sum()),
                          "worst_start": worst.actual_start, "worst_end": worst.actual_end,
                          "worst_contributed": worst.contributed, "worst_final_value": worst.final_value,
                          "maximum_tqqq_market_weight": g.maximum_tqqq_market_weight.max()})
    summary = pd.DataFrame(summaries)

    actual_examples = []
    for years in (5, 10):
        candidates = original[original.years.eq(years) & original.leverage.eq(3)]
        best = candidates.loc[candidates.roi.idxmax()]
        real_paths, histories = {}, {}
        for lev, symbol in ((1, "QQQ"), (2, "QLD"), (3, "TQQQ")):
            path = pd.read_csv(root / "data" / "five_year_dca_2026-09-17" /
                               f"{symbol.lower()}_2010-01-01_2026-09-17.csv", index_col="date", parse_dates=True).adj_close
            real_paths[lev] = path.loc[best.actual_start:best.actual_end]
            h = simulate_close_dca(real_paths[lev])
            histories[lev] = h
            actual_examples.append({"years": years, "start": best.actual_start, "end": best.actual_end,
                                    "symbol": symbol, "contributed": h.deposited.iloc[-1],
                                    "final_value": h.equity.iloc[-1], "profit": h.profit.iloc[-1]})
        for symbol, h in (("50_50_deposits", combine_sleeves(histories, {1: .5, 3: .5}, 100)),
                          ("50_50_monthly", simulate_monthly_mix(real_paths[1], real_paths[3]))):
            actual_examples.append({"years": years, "start": best.actual_start, "end": best.actual_end,
                                    "symbol": symbol, "contributed": h.deposited.iloc[-1],
                                    "final_value": h.equity.iloc[-1], "profit": h.profit.iloc[-1]})
    actual_examples = pd.DataFrame(actual_examples)
    allocation_ratios = []
    for years, g in endpoints.groupby("years"):
        paired = g.pivot(index="window_start", columns="allocation", values="final_value")
        for allocation in ALLOCATIONS:
            for benchmark in ("100_QQQ", "100_QLD"):
                ratio = paired[allocation] / paired[benchmark]
                allocation_ratios.append({"years": years, "allocation": allocation, "benchmark": benchmark,
                                          "median_wealth_ratio": ratio.median(), "minimum": ratio.min(),
                                          "maximum": ratio.max(), "win_count": int((ratio > 1 + 1e-10).sum()),
                                          "windows": len(ratio)})
    pd.DataFrame(allocation_ratios).to_csv(output / "allocation_paired_ratios.csv", index=False)
    for name, frame in (("downside", downside), ("paired_ratios", ratios), ("allocation_windows", endpoints),
                        ("allocation_summary", summary), ("actual_fund_upside_examples", actual_examples)):
        frame.to_csv(output / f"{name}.csv", index=False)
    plot(summary, output)
    report(downside, ratios, summary, actual_examples, output)
    print("DOWNSIDE\n", downside.to_string(index=False))
    print("\nPAIRED TERMINAL WEALTH RATIOS\n", ratios.to_string(index=False))
    print("\nALLOCATION SENSITIVITY\n", summary.to_string(index=False))
    print("\nREAL FUND UPSIDE EXAMPLES\n", actual_examples.to_string(index=False))


def plot(summary: pd.DataFrame, output: Path) -> None:
    names = ["100_QQQ", "100_QLD", "100_TQQQ", "50_TQQQ_50_QQQ", "50_50_monthly_rebalance"]
    labels = [r"\$100 QQQ", r"\$100 QLD", r"\$100 TQQQ",
              r"\$50 QQQ + \$50 TQQQ" + "\nNo rebalancing",
              r"\$50 QQQ + \$50 TQQQ" + "\nMonthly 50/50 reset"]
    colors = ["#2563a6", "#16836b", "#cd454c", "#85623a", "#555555"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained", sharey=True)
    for ax, years in zip(axes, (5, 10)):
        rows = summary[summary.years.eq(years)].set_index("allocation").loc[names]
        for i, (_, row) in enumerate(rows.iterrows()):
            ax.plot([1 + row.worst_end_roi, 1 + row.best_end_roi], [i, i], color=colors[i], alpha=.35, linewidth=2)
            ax.plot([1 + row.p10_end_roi, 1 + row.p90_end_roi], [i, i], color=colors[i], linewidth=7, solid_capstyle="butt")
            ax.plot(1 + row.median_end_roi, i, "o", color=colors[i], markersize=7, markeredgecolor="white")
        ax.axvline(1, color="#555555", linestyle="--", linewidth=1)
        ax.set_xscale("log")
        ticks = [.1, .3, .5, 1, 2, 3, 5, 10, 25]
        visible = [x for x in ticks if x >= (1 + rows.worst_end_roi.min()) * .7 and x <= (1 + rows.best_end_roi.max()) * 1.2]
        ax.set_xticks(visible, [f"{x:g}x" for x in visible])
        ax.set_title(f"{years}-year windows | n={int(rows.windows.iloc[0])}")
        ax.set_xlabel("Ending value / cumulative contributions (log)")
        ax.grid(axis="x", alpha=.15)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(range(len(labels)), labels)
    axes[0].invert_yaxis()
    fig.suptitle("Synthetic DCA, including costs | Thin: min-max; thick: 10th-90th percentile; dot: median", fontsize=12)
    fig.savefig(output / "risk_reward.png", dpi=170)
    plt.close(fig)


def report(downside, ratios, summary, actual, output):
    lines = ["# DCA downside, upside, and contribution sizing", "",
             "Uses the prior Nasdaq-100 synthetic daily-reset model, 1999-03-04 to 2026-09-16, illustrative costs. This is not actual pre-inception TQQQ/QLD history.",
             "All loss percentages below are relative to cumulative nominal contributions, NOT drawdowns from a previous portfolio peak.", "",
             "## Conditional downside", "",
             "Conditional median means only the losing windows; it is not the median outcome of all windows.", "",
             "| Years | Windows | Losing windows | Median loss among losers | Worst ending loss | Ending losses over 30% | Worst interim loss across windows |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for r in downside.itertuples():
        lines.append(f"| {r.years} | {r.windows} | {r.negative_windows} | {r.loss_median_conditional:.2%} | {r.worst_end_loss:.2%} | {r.end_loss_over_30_count} | {r.worst_interim_loss:.2%} |")
    lines += ["", "## Paired terminal wealth ratios", "",
              "Compare final account values within the SAME window with identical $100/week deposits. These are wealth ratios, not profit ratios or annualized returns.", "",
              "| Years | 3x relative to | Minimum | Median | 90th percentile | Maximum |", "|---|---|---:|---:|---:|---:|"]
    for r in ratios.itertuples():
        lines.append(f"| {r.years} | {r.benchmark_leverage}x | {r.minimum:.2f}x | {r.median:.2f}x | {r.p90:.2f}x | {r.maximum:.2f}x |")
    lines += ["", "## Allocation comparison", "",
              "QQQ/QLD/TQQQ labels refer to synthetic 1x/2x/3x here. Except the monthly-rebalance strategy, weights apply only to NEW CONTRIBUTIONS; market weights drift. The monthly strategy restores 50/50 at the first trading day's close each month, splits other weekly deposits 50/50, and charges 5 bps on all buys AND sells. Cash earns 0%, has no nominal loss, and is kept separate. Except 50_TQQQ_only ($50/week), all rows save/invest $100/week in total.", "",
              "| Years | Allocation | Worst end ROI | Median end ROI | Best end ROI | Worst interim loss | Worst ending contributed | Worst ending value |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in summary.itertuples():
        lines.append(f"| {r.years} | {r.allocation} | {r.worst_end_roi:.2%} | {r.median_end_roi:.2%} | {r.best_end_roi:.2%} | {r.worst_interim_loss:.2%} | ${r.worst_contributed:,.0f} | ${r.worst_final_value:,.0f} |")
    pairs = pd.read_csv(output / "allocation_paired_ratios.csv")
    pairs = pairs[pairs.allocation.isin(["50_TQQQ_50_QQQ", "50_50_monthly_rebalance"])]
    lines += ["", "## Mixed portfolios versus the benchmarks", "",
              "Median of matched-window final-wealth ratios, not a ratio of unpaired medians and not a forecast.", "",
              "| Years | Mix | Benchmark | Median wealth ratio | Mix higher (windows) | All windows |",
              "|---|---|---|---:|---:|---:|"]
    for r in pairs.itertuples():
        lines.append(f"| {r.years} | {r.allocation} | {r.benchmark} | {r.median_wealth_ratio:.3f}x | {r.win_count} | {r.windows} |")
    lines += ["", "## Strong upside examples, checked using real ETF data", "",
              "Dates selected retrospectively as the synthetic 3x's highest-return windows, then recalculated using REAL split/dividend-adjusted ETF prices at weekly closes. These are deliberately favorable examples, not representative forecasts. $100/week, 5 bps buying slippage, no taxes.", "",
              "| Years | Dates | Contributed | QQQ ending | QLD ending | TQQQ ending | 50/50 deposits ending | Monthly 50/50 ending |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for years, group in actual.groupby("years"):
        v = group.set_index("symbol").final_value
        r = group.iloc[0]
        lines.append(f"| {years} | {r.start} to {r.end} | ${r.contributed:,.0f} | ${v.QQQ:,.0f} | ${v.QLD:,.0f} | ${v.TQQQ:,.0f} | ${v['50_50_deposits']:,.0f} | ${v['50_50_monthly']:,.0f} |")
    lines += ["", "## Interpretation", "",
              "- Halving a pure TQQQ contribution halves dollars invested, terminal dollars, profits, and dollar losses. Its percentage return and percentage losses are unchanged.",
              "- Do not count the uninvested $50 as a protective cash sleeve unless it is actually retained and included in the same portfolio denominator.",
              "- A 50/50 contribution split is NOT a 50% portfolio-weight cap. Gains can make the leveraged sleeve dominate later. The separate monthly reset mitigates drift but does not bound losses. Tax costs of selling are omitted, so taxable-account results may be worse.",
              "- At a 50/50 market-value split the approximate daily index exposure is 0.5 * 1 + 0.5 * 3 = 2x. This is not half the risk of a Nasdaq allocation; compare it directly with QLD. Monthly 50/50 is not mechanically identical to a daily-reset 2x fund.",
              "- $30 TQQQ plus $70 cash is an illustrative accounting case, not a recommendation. With ring-fenced cash, no borrowing/transfers, and fund losses limited to invested capital, losing the entire TQQQ sleeve loses 30% of nominal cumulative contributions; portfolio peak drawdowns can still exceed 30%, and cash has inflation/credit risks outside this idealization.",
              "- All windows overlap; historical counts, quantiles, and maxima are not estimated probabilities or future loss bounds. Model assumes fund survival and continuous trading/contributions. Retirement suitability and current portfolio exposures are not established by this exercise.",
              "- Model funding and fees: 1x 0.20%/year; 2x and 3x 0.95%/year plus historical federal funds rate + 0.50% on additional borrowed exposure. Daily reset, dividends included; no tax, inflation, or withdrawal modeling.",
              "", "Sources and prior methodology:", "",
              "- [Nasdaq total return via FRED](https://fred.stlouisfed.org/series/NASDAQXNDX)",
              "- [Historical financing benchmark](https://fred.stlouisfed.org/series/DFF)",
              "- [TQQQ issuer daily objective and risks](https://www.proshares.com/our-etfs/leveraged-and-inverse/tqqq)",
              "- Real fund checks use the saved Yahoo adjusted-price data from the previous five-year comparison.", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
