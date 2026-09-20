import unittest

import pandas as pd

from quant_tqqq_backtest.smart_dca import (
    contribution_adjusted_returns, regime_with_buffer, simulate_mix,
    trend_target, volatility_target,
)


class SmartDcaRulesTest(unittest.TestCase):
    def test_first_entry_uses_prior_signal(self):
        dates = pd.date_range("2026-01-05", periods=2)
        prices = pd.DataFrame({"adj_open": [100., 100.], "adj_close": [100., 100.]}, index=dates)
        target = pd.Series([0., 1., 1.], index=pd.date_range("2026-01-04", periods=3))
        result = simulate_mix(prices, prices, target, 500., 0., 0.)
        self.assertEqual(result.tqqq_weight.tolist(), [0., 1.])

    def test_opening_deposit_does_not_dilute_overnight_return(self):
        dates = pd.to_datetime(["2026-01-09", "2026-01-12"])
        prices = pd.DataFrame({"adj_open": [100., 110.], "adj_close": [100., 121.]}, index=dates)
        target = pd.Series(0., index=dates)
        result = simulate_mix(prices, prices, target, 500., 100., 0.)
        returns = contribution_adjusted_returns(result)
        self.assertAlmostEqual(returns.iloc[0], 0.)
        self.assertAlmostEqual(returns.iloc[1], .21)
        self.assertAlmostEqual(result.equity.iloc[-1], 836.)

    def test_first_day_cost_is_in_return(self):
        dates = pd.date_range("2026-01-05", periods=2)
        prices = pd.DataFrame({"adj_open": 100., "adj_close": 100.}, index=dates)
        result = simulate_mix(prices, prices, pd.Series(0., index=dates), 500., 0., 5.)
        self.assertAlmostEqual(contribution_adjusted_returns(result).iloc[0], 1 / 1.0005 - 1)

    def test_trend_target_turns_leverage_off_in_bear_regime(self):
        signal = pd.DataFrame(
            {"regime": ["bull", "bear", "bull"]},
            index=pd.date_range("2026-01-01", periods=3),
        )
        self.assertEqual(trend_target(signal, 0.30).tolist(), [0.30, 0.0, 0.30])

    def test_volatility_target_is_capped_and_disabled_in_bear(self):
        signal = pd.DataFrame(
            {
                "regime": ["bull", "bull", "bear"],
                "realized_vol": [0.10, 0.25, 0.10],
            },
            index=pd.date_range("2026-01-01", periods=3),
        )
        self.assertEqual(volatility_target(signal).round(4).tolist(), [0.30, 0.10, 0.0])

    def test_regime_buffer_persists(self):
        close = pd.Series(
            [100.0, 100.0, 105.0, 101.0, 95.0, 97.5],
            index=pd.date_range("2026-01-01", periods=6),
        )
        regimes = regime_with_buffer(close, sma_days=2, bull_multiple=1.02, bear_multiple=0.98)[
            "regime"
        ].tolist()
        self.assertEqual(regimes[2], "bull")
        self.assertEqual(regimes[3], "bull")
        self.assertEqual(regimes[4], "bear")
        self.assertEqual(regimes[5], "bear")


if __name__ == "__main__":
    unittest.main()
