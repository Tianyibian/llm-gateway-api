import unittest

import numpy as np
import pandas as pd

from quant_tqqq_backtest.dca_risk_reward import combine_sleeves, simulate_monthly_mix
from quant_tqqq_backtest.dotcom_stress import simulate_close_dca


class DcaSizingTest(unittest.TestCase):
    def histories(self):
        dates = pd.to_datetime(["2000-01-03", "2000-01-10", "2000-01-18"])
        return {1: simulate_close_dca(pd.Series([100., 90., 80.], index=dates)),
                2: simulate_close_dca(pd.Series([100., 80., 60.], index=dates)),
                3: simulate_close_dca(pd.Series([100., 70., 30.], index=dates))}

    def test_halving_contributions_halves_dollars_not_percent(self):
        h = self.histories()
        full = combine_sleeves(h, {3: 1.}, 100)
        half = combine_sleeves(h, {3: 1.}, 50)
        np.testing.assert_allclose(full.equity / 2, half.equity)
        np.testing.assert_allclose(full.profit / 2, half.profit)
        np.testing.assert_allclose(full.roi, half.roi)

    def test_cash_sleeve_dilutes_principal_loss(self):
        h = self.histories()
        full = combine_sleeves(h, {3: 1.}, 100)
        mixed = combine_sleeves(h, {3: .5, 0: .5}, 100)
        np.testing.assert_allclose(mixed.roi, full.roi / 2)

    def test_mixed_etf_roi_is_weighted_not_independent_minima(self):
        h = self.histories()
        mixed = combine_sleeves(h, {3: .5, 1: .5}, 100)
        np.testing.assert_allclose(mixed.roi, (h[3].profit_on_deposited + h[1].profit_on_deposited) / 2)
        self.assertLess(mixed.tqqq_market_weight.iloc[-1], .5)

    def test_invalid_budget_and_weights_rejected(self):
        h = self.histories()
        for weights, weekly in [({3: .7, 0: .5}, 100), ({3: -1, 0: 2}, 100), ({3: 1}, 0)]:
            with self.assertRaises(ValueError):
                combine_sleeves(h, weights, weekly)

    def test_monthly_mix_flat_prices_and_deposits(self):
        dates = pd.to_datetime(["2000-01-03", "2000-01-10", "2000-02-01"])
        prices = pd.Series(100., index=dates)
        h = simulate_monthly_mix(prices, prices, slippage_bps=0)
        np.testing.assert_allclose(h.equity, [100, 200, 300])
        np.testing.assert_allclose(h.tqqq_market_weight, .5)
        self.assertTrue((h.cash >= 0).all())

    def test_monthly_mix_resets_after_drift_and_preserves_value(self):
        dates = pd.to_datetime(["2000-01-03", "2000-01-04", "2000-02-01"])
        q = pd.Series(100., index=dates)
        t = pd.Series([100., 200., 200.], index=dates)
        h = simulate_monthly_mix(q, t, slippage_bps=0)
        np.testing.assert_allclose(h.equity, [100, 150, 250])
        self.assertAlmostEqual(h.tqqq_market_weight.iloc[1], 2 / 3)
        self.assertAlmostEqual(h.tqqq_market_weight.iloc[2], .5)

    def test_monthly_mix_is_self_financing_with_costs(self):
        dates = pd.to_datetime(["2000-01-03", "2000-01-04", "2000-02-01"])
        q = pd.Series(100., index=dates)
        t = pd.Series([100., 400., 400.], index=dates)
        h = simulate_monthly_mix(q, t, slippage_bps=25)
        self.assertTrue((h.cash >= 0).all())
        self.assertAlmostEqual(h.equity.iloc[0] + h.transaction_cost.iloc[0], 100)
        self.assertAlmostEqual(h.equity.iloc[-1] + h.transaction_cost.iloc[-1], h.equity.iloc[-2] + 100)

    def test_monthly_mix_price_normalization_invariance(self):
        dates = pd.to_datetime(["2000-01-03", "2000-01-04", "2000-02-01"])
        q = pd.Series([100., 110., 90.], index=dates)
        t = pd.Series([100., 130., 60.], index=dates)
        a = simulate_monthly_mix(q, t)
        b = simulate_monthly_mix(q * 5, t / 1000)
        np.testing.assert_allclose(a.equity, b.equity)


if __name__ == "__main__":
    unittest.main()
