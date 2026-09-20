import unittest

import numpy as np
import pandas as pd

from quant_tqqq_backtest.dotcom_stress import (
    Costs, SCENARIOS, financing_accrual, first_nonnegative_after,
    recovery_metrics, simulate_close_dca, synthesize,
)


class DotcomStressTest(unittest.TestCase):
    def rates(self):
        return pd.Series(.05, index=pd.date_range("1999-01-01", "2001-01-01"))

    def test_daily_reset_not_multiple_of_terminal_return(self):
        index = pd.Series([100., 110., 100.], index=pd.to_datetime(["2000-01-03", "2000-01-04", "2000-01-05"]))
        paths = synthesize(index, self.rates(), SCENARIOS[0])
        self.assertAlmostEqual(paths[1].iloc[-1], 100)
        self.assertAlmostEqual(paths[2].iloc[-1], 100 * 1.2 * (1 - 2 / 11))
        self.assertAlmostEqual(paths[3].iloc[-1], 100 * 1.3 * (1 - 3 / 11))

    def test_financing_accrues_weekend_and_uses_lagged_rates(self):
        dates = pd.to_datetime(["2000-01-07", "2000-01-10"])
        rates = self.rates()
        rates.loc["2000-01-10":] = .99
        accrued = financing_accrual(dates, rates)
        self.assertAlmostEqual(accrued.iloc[-1], .05 * 3 / 365)
        paths = synthesize(pd.Series(100., index=dates), rates, Costs("test", 0, .01, .005, True))
        self.assertAlmostEqual(paths[3].iloc[-1], 100 * (1 - (2 * .055 + .01) * 3 / 365))

    def test_close_deposit_does_not_receive_earlier_daily_return(self):
        path = pd.Series([100., 50.], index=pd.to_datetime(["2000-01-07", "2000-01-10"]))
        h = simulate_close_dca(path, slippage_bps=0)
        self.assertEqual(h.deposit.tolist(), [100, 100])
        self.assertAlmostEqual(h.equity.iloc[-1], 150)
        self.assertAlmostEqual(h.profit_on_deposited.iloc[-1], -.25)

    def test_one_x_no_cost_matches_benchmark(self):
        index = pd.Series([100., 110., 90.], index=pd.to_datetime(["2000-01-03", "2000-01-04", "2000-01-05"]))
        paths = synthesize(index, self.rates(), SCENARIOS[0])
        np.testing.assert_allclose(paths[1], index)

    def test_scaling_prices_or_contributions_does_not_change_roi(self):
        path = pd.Series([100., 50., 80.], index=pd.to_datetime(["2000-01-03", "2000-01-10", "2000-01-18"]))
        a = simulate_close_dca(path)
        b = simulate_close_dca(path * 1e-6, weekly=200)
        np.testing.assert_allclose(a.equity * 2, b.equity)
        np.testing.assert_allclose(a.profit_on_deposited, b.profit_on_deposited)

    def test_exhausted_nav_is_not_resurrected(self):
        index = pd.Series([100., 60., 100.], index=pd.to_datetime(["2000-01-03", "2000-01-04", "2000-01-05"]))
        with self.assertRaises(ValueError):
            synthesize(index, self.rates(), SCENARIOS[0])

    def test_recovery_ignores_early_bounce_and_reports_reloss(self):
        dates = pd.date_range("2000-01-01", periods=7)
        profit = pd.Series([-1., 1., -50., 1., -5., -1., 2.], index=dates)
        history = pd.DataFrame({"profit": profit, "profit_on_deposited": profit / 100,
                                "initial_dollar_value": .1, "nav_drawdown": -.9})
        metrics = recovery_metrics(history, dates[2])
        self.assertEqual(metrics["first_dca_breakeven_after_2002_trough"], "2000-01-04")
        self.assertEqual(metrics["last_dca_breakeven_in_sample"], "2000-01-07")
        self.assertIsNone(metrics["first_lump_breakeven_after_2002_trough"])
        self.assertIsNone(first_nonnegative_after(profit, dates[-1] + pd.Timedelta(days=1)))

    def test_stopping_deposits_freezes_cost_basis(self):
        path = pd.Series([100., 100., 100.], index=pd.to_datetime(["2000-01-03", "2000-01-10", "2000-01-18"]))
        h = simulate_close_dca(path, slippage_bps=0, stop_after=pd.Timestamp("2000-01-10"))
        self.assertEqual(h.deposited.tolist(), [100, 200, 200])


if __name__ == "__main__":
    unittest.main()
