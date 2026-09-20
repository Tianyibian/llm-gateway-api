import unittest

import numpy as np
import pandas as pd

from quant_tqqq_backtest.dca_five_year import aligned_window, simulate_weekly


class WeeklyDcaTest(unittest.TestCase):
    def prices(self, dates, opens, closes):
        return pd.DataFrame({"adj_open": opens, "adj_close": closes}, index=pd.to_datetime(dates))

    def test_flat_prices_deposits_are_not_returns(self):
        asset = self.prices(["2025-01-06", "2025-01-07", "2025-01-13"], [100.] * 3, [100.] * 3)
        history, metrics, trades = simulate_weekly(asset, slippage_bps=0)
        self.assertEqual(len(trades), 2)
        self.assertAlmostEqual(metrics["contributed"], 200)
        self.assertAlmostEqual(metrics["final_value"], 200)
        self.assertAlmostEqual(metrics["xirr"], 0)
        np.testing.assert_allclose(history.daily_return, 0)

    def test_price_scale_does_not_create_profit(self):
        asset = self.prices(["2025-01-03", "2025-01-06", "2025-01-13"], [100., 110., 80.], [105., 100., 120.])
        a, am, _ = simulate_weekly(asset)
        b, bm, _ = simulate_weekly(asset / 1000)
        np.testing.assert_allclose(a.equity, b.equity)
        self.assertAlmostEqual(am["xirr"], bm["xirr"])

    def test_deposit_cannot_mask_overnight_loss(self):
        asset = self.prices(["2025-01-03", "2025-01-06"], [100., 50.], [100., 50.])
        history, metrics, _ = simulate_weekly(asset, slippage_bps=0)
        self.assertAlmostEqual(history.equity.iloc[-1], 150)
        self.assertAlmostEqual(metrics["max_drawdown"], -.5)
        self.assertAlmostEqual(metrics["worst_loss_on_contributed"], -.25)

    def test_opening_deposit_and_two_return_segments(self):
        asset = self.prices(["2025-01-03", "2025-01-06"], [100., 110.], [100., 121.])
        history, _, _ = simulate_weekly(asset, slippage_bps=0)
        self.assertAlmostEqual(history.equity.iloc[-1], 231)
        self.assertAlmostEqual(history.daily_return.iloc[-1], .21)

    def test_holiday_week_uses_first_available_day(self):
        asset = self.prices(["2025-01-17", "2025-01-21", "2025-01-22", "2025-01-27"], [100.] * 4, [100.] * 4)
        _, metrics, trades = simulate_weekly(asset, slippage_bps=0)
        self.assertEqual(trades.execution_date.dt.strftime("%Y-%m-%d").tolist(), ["2025-01-17", "2025-01-21", "2025-01-27"])
        self.assertEqual(metrics["contributed"], 300)

    def test_first_cost_and_proportional_contributions(self):
        asset = self.prices(["2025-01-03", "2025-01-06"], [100., 100.], [100., 100.])
        history, metrics, _ = simulate_weekly(asset)
        _, larger, _ = simulate_weekly(asset, weekly=200)
        self.assertAlmostEqual(history.daily_return.iloc[0], 1 / 1.0005 - 1)
        self.assertAlmostEqual(metrics["final_value"], 200 / 1.0005)
        self.assertAlmostEqual(larger["final_value"], 2 * metrics["final_value"])
        self.assertAlmostEqual(larger["xirr"], metrics["xirr"])

    def test_missing_bars_are_not_silently_dropped(self):
        asset = self.prices(["2025-01-03", "2025-01-06", "2025-01-07"], [100.] * 3, [100.] * 3)
        with self.assertRaises(ValueError):
            aligned_window({"QQQ": asset, "QLD": asset.iloc[1:], "TQQQ": asset}, asset.index[0], asset.index[-1])

    def test_invalid_prices_are_rejected(self):
        asset = self.prices(["2025-01-03", "2025-01-06"], [100., 0.], [100., 100.])
        with self.assertRaises(ValueError):
            simulate_weekly(asset)


if __name__ == "__main__":
    unittest.main()
