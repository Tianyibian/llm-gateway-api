# TQQQ regime strategy backtest

This directory contains a reproducible exploratory backtest of the QQQ/TQQQ
regime strategy discussed in the associated Codex task.

Run:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache python3 quant_tqqq_backtest/backtest.py --refresh
python3 -m unittest quant_tqqq_backtest.test_backtest
```

The script downloads daily Yahoo Finance chart data into `data/` and writes a
Markdown report, trade logs, and an equity chart into `output/`.

Yahoo Finance data is convenient for exploratory research but is not an
execution-grade or guaranteed data source. Validate the strategy against a
second source before using it for live trading.
