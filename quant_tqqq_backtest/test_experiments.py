import unittest

import numpy as np
import pandas as pd

from quant_tqqq_backtest.experiments import metrics, simulate


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.date_range("2026-01-05", periods=4)
        self.assets = {"QQQ": pd.DataFrame({"adj_open": 100., "adj_close": 100.}, index=self.dates)}

    def test_all_cash_is_preserved(self):
        targets = pd.DataFrame({"QQQ": 0.}, index=self.dates)
        result = simulate(self.assets, targets)
        np.testing.assert_allclose(result.history.equity, 500.)
        self.assertTrue(result.trades.empty)

    def test_cost_accounting_and_initial_drawdown(self):
        targets = pd.DataFrame({"QQQ": [1., 0.]}, index=pd.to_datetime(["2026-01-04", "2026-01-06"]))
        result = simulate(self.assets, targets)
        after_buy = 500 / 1.0005
        self.assertAlmostEqual(result.history.equity.iloc[0], after_buy)
        self.assertAlmostEqual(result.history.equity.iloc[-1], after_buy * .9995)
        self.assertAlmostEqual(500 - result.history.equity.iloc[-1], result.trades.cost.sum())
        self.assertAlmostEqual(metrics(result)["max_drawdown"], after_buy * .9995 / 500 - 1)
        self.assertGreaterEqual(result.history.cash.min(), -1e-8)

    def test_same_day_signal_cannot_trade(self):
        targets = pd.DataFrame({"QQQ": [1., 0., 0., 0.]}, index=self.dates)
        result = simulate(self.assets, targets)
        self.assertEqual(result.history.QQQ.iloc[0], 0.)
        self.assertAlmostEqual(result.history.QQQ.iloc[1], 1.)
        self.assertEqual(result.history.QQQ.iloc[2], 0.)
        self.assertTrue((result.trades.signal_date < result.trades.execution_date).all())

    def test_rotation_between_assets_charges_both_sides(self):
        assets = {**self.assets, "QLD": self.assets["QQQ"].copy()}
        targets = pd.DataFrame({"QQQ": [1., 0.], "QLD": [0., 1.]},
                               index=pd.to_datetime(["2026-01-04", "2026-01-06"]))
        result = simulate(assets, targets)
        expected = 500 / 1.0005 * .9995 / 1.0005
        self.assertAlmostEqual(result.history.equity.iloc[-1], expected)
        self.assertAlmostEqual(500 - expected, result.trades.cost.sum())
        self.assertGreaterEqual(result.history.cash.min(), -1e-8)

    def test_future_information_cannot_change_prefix(self):
        targets = pd.DataFrame({"QQQ": [1., .5, 0., 1.]}, index=self.dates)
        full = simulate(self.assets, targets)
        short_assets = {s: prices.iloc[:3] for s, prices in self.assets.items()}
        prefix = simulate(short_assets, targets.iloc[:3])
        pd.testing.assert_frame_equal(full.history.iloc[:3], prefix.history)

    def test_rejects_borrowed_or_short_weights(self):
        for weight in (-.1, 1.1, np.nan):
            with self.assertRaises(ValueError):
                simulate(self.assets, pd.DataFrame({"QQQ": weight}, index=self.dates))


if __name__ == "__main__":
    unittest.main()
