import unittest

import numpy as np
import pandas as pd

from quant_tqqq_backtest.dip_dca import prior_close_signal, simulate, treasury_carry
from quant_tqqq_backtest.dotcom_stress import simulate_close_dca


class DipDcaTest(unittest.TestCase):
    def prices(self, dates, tqqq=None, bond=None):
        return pd.DataFrame({"qqq": 100., "qld": 100., "tqqq": tqqq if tqqq is not None else 100.,
                             "bond": bond if bond is not None else 100.}, index=pd.to_datetime(dates))

    def test_signal_uses_prior_session_and_includes_exact_one_percent(self):
        dates = pd.date_range("2024-01-01", periods=4)
        q = pd.Series([100., 99., 100., 95.], index=dates)
        self.assertEqual(prior_close_signal(q).tolist(), [False, False, True, False])

    def test_one_point_five_percent_threshold_is_stricter_and_lagged(self):
        dates = pd.date_range("2024-01-01", periods=6)
        q = pd.Series(100 * np.cumprod([1., .99, .986, .985, .98, 1.]), index=dates)
        signal = prior_close_signal(q, -.015)
        self.assertEqual(signal.tolist(), [False, False, False, False, True, True])
        self.assertTrue((~signal | prior_close_signal(q, -.01)).all())

    def test_invalid_threshold_rejected(self):
        q = pd.Series([100., 99.], index=pd.date_range("2024-01-01", periods=2))
        for threshold in [0, .015, -1, float("nan"), float("-inf")]:
            with self.assertRaises(ValueError):
                prior_close_signal(q, threshold)

    def test_dip_buys_at_next_open_not_signal_close(self):
        prices = self.prices(["2024-01-02", "2024-01-03", "2024-01-04"], [100., 70., 90.])
        q = pd.Series([100., 98., 99.], index=prices.index)
        opens = prices.copy()
        opens.loc[prices.index[-1], "tqqq"] = 80.
        h = simulate(opens, prices, prior_close_signal(q), "dip_cash", slippage_bps=0)
        np.testing.assert_allclose(h.tqqq_units, [0, 0, 1.25])
        self.assertAlmostEqual(h.equity.iloc[-1], 112.5)

    def test_no_dip_preserves_bond_position_and_includes_its_return(self):
        p = self.prices(["2024-01-02", "2024-01-03"], bond=[100., 101.])
        h = simulate(p, p, pd.Series(False, index=p.index), "dip_bond", slippage_bps=0)
        np.testing.assert_allclose(h.equity, [100, 101])
        self.assertEqual(h.tqqq_units.sum(), 0)

    def test_trigger_deploys_accumulated_contributions_and_never_sells_tqqq(self):
        p = self.prices(["2024-01-02", "2024-01-08", "2024-01-09", "2024-01-10", "2024-01-15"])
        h = simulate(p, p, pd.Series([False, False, True, False, False], index=p.index),
                     "dip_bond", slippage_bps=0)
        np.testing.assert_allclose(h.tqqq_units, [0, 0, 2, 2, 2])
        np.testing.assert_allclose(h.bond_value, [100, 200, 0, 0, 100])
        np.testing.assert_allclose(h.equity, [100, 200, 200, 200, 300])
        self.assertAlmostEqual(h.deployed_principal.sum(), 200)
        self.assertAlmostEqual(h.principal_wait_days.sum() / 200, 4)

    def test_both_legs_pay_transaction_cost_without_borrowing(self):
        p = self.prices(["2024-01-02", "2024-01-03"])
        h = simulate(p, p, pd.Series([False, True], index=p.index), "dip_bond", slippage_bps=100)
        expected = 100 / 1.01 * .99 / 1.01
        self.assertAlmostEqual(h.equity.iloc[-1], expected)
        self.assertAlmostEqual(h.equity.iloc[-1] + h.transaction_cost.sum(), 100)
        self.assertTrue(h.cash.ge(0).all())

    def test_close_execution_baseline_matches_previous_engine(self):
        p = self.prices(["2024-01-02", "2024-01-03", "2024-01-08"], [100., 80., 110.])
        h = simulate(p, p, pd.Series(False, index=p.index), "weekly_tqqq")
        previous = simulate_close_dca(p.tqqq)
        np.testing.assert_allclose(h.equity, previous.equity)
        np.testing.assert_allclose(h.roi, previous.profit_on_deposited)

    def test_equal_mix_and_scale_invariance(self):
        p = self.prices(["2024-01-02", "2024-01-03", "2024-01-08"], [100., 80., 110.])
        sig = pd.Series([False, True, True], index=p.index)
        h = simulate(p, p, sig, "weekly_half_half")
        q = simulate(p, p, sig, "weekly_qqq")
        t = simulate(p, p, sig, "weekly_tqqq")
        np.testing.assert_allclose(h.equity, (q.equity + t.equity) / 2)
        a = simulate(p, p, sig, "dip_bond")
        b = simulate(p * .00001, p * .00001, sig, "dip_bond", weekly=50)
        np.testing.assert_allclose(a.equity / 2, b.equity)
        np.testing.assert_allclose(a.roi, b.roi)

    def test_treasury_carry_is_lagged_and_accrues_weekends(self):
        dates = pd.to_datetime(["2024-01-05", "2024-01-08"])
        rates = pd.Series(.05, index=pd.date_range("2023-12-01", "2024-01-10"))
        rates.loc["2024-01-05":] = .9
        p = treasury_carry(dates, rates, annual_fee=0)
        self.assertAlmostEqual(p.iloc[-1] / p.iloc[0], np.exp(.05 * 3 / 365))

    def test_missing_signal_and_invalid_prices_rejected(self):
        p = self.prices(["2024-01-02", "2024-01-03"])
        with self.assertRaises(ValueError):
            simulate(p, p, pd.Series(False, index=p.index[:1]), "dip_cash")
        p.iloc[0, 0] = 0
        with self.assertRaises(ValueError):
            simulate(p, p, pd.Series(False, index=p.index), "dip_cash")


if __name__ == "__main__":
    unittest.main()
