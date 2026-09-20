import unittest

import pandas as pd

from quant_tqqq_backtest.backtest import (
    StrategyConfig,
    compute_actions,
    compute_regime,
    simulate_strategy,
)


class StrategyRulesTest(unittest.TestCase):
    def test_buffer_preserves_regime(self):
        config = StrategyConfig(sma_days=2, bull_multiple=1.04, bear_multiple=0.97)
        dates = pd.date_range("2024-01-01", periods=6, freq="D")
        close = pd.Series([100.0, 100.0, 110.0, 108.0, 90.0, 92.0], index=dates)
        regime = compute_regime(close, config)["regime"].tolist()
        self.assertEqual(regime[2], "bull")
        self.assertEqual(regime[3], "bull")
        self.assertEqual(regime[4], "bear")
        self.assertEqual(regime[5], "bear")

    def test_transition_and_dip_actions(self):
        config = StrategyConfig()
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        signal = pd.DataFrame(
            {
                "regime": [None, "bear", "bull", "bull", "bear"],
                "daily_return": [0.0, 0.0, 0.01, -0.011, -0.04],
            },
            index=dates,
        )
        actions = compute_actions(signal, config).tolist()
        self.assertEqual(actions, ["hold", "hold", "buy", "buy", "sell"])

    def test_signal_executes_on_next_trading_day(self):
        config = StrategyConfig(slippage_bps=0)
        signal_dates = pd.date_range("2024-01-01", periods=3, freq="D")
        signal = pd.DataFrame(
            {
                "regime": ["bear", "bull", "bull"],
                "daily_return": [0.0, 0.02, 0.0],
            },
            index=signal_dates,
        )
        actions = pd.Series(["hold", "buy", "hold"], index=signal_dates)
        asset_dates = pd.date_range("2024-01-02", periods=3, freq="D")
        asset = pd.DataFrame(
            {"adj_open": [10.0, 11.0, 12.0], "adj_close": [10.5, 11.5, 12.5]},
            index=asset_dates,
        )
        result = simulate_strategy(signal, actions, asset, config, 100.0, 0.0, False)
        self.assertEqual(result.trades.iloc[0]["execution_date"], pd.Timestamp("2024-01-03"))
        self.assertEqual(result.trades.iloc[0]["price"], 11.0)


if __name__ == "__main__":
    unittest.main()
