"""Quick re-run of ARIMA forecasts in trader-friendly metrics.

Not part of the project library; a one-off audit script. Loads cached data,
re-runs the walk-forward backtest on each frequency, and reports:
- total trades, wins, win rate
- gross cumulative log-return if you took every forecast's sign as a position
- gross cumulative log-return of buy-and-hold over the same window
- mean per-trade return
- breakdown of long vs short trades
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.auto_arima import grid_search
from src.data import add_returns, load_bars
from src.models import ARIMAForecaster, walk_forward


def trading_metrics(results: pd.DataFrame, label: str) -> None:
    f = results["forecast"].to_numpy()
    a = results["actual"].to_numpy()
    opinion = f != 0.0
    n = int(opinion.sum())

    signal = np.sign(f[opinion])
    realized = a[opinion]
    per_trade = signal * realized  # log-return earned per trade
    wins = (per_trade > 0).sum()
    losses = (per_trade < 0).sum()
    flat = (per_trade == 0).sum()

    long_mask = signal > 0
    short_mask = signal < 0
    long_wins = ((per_trade > 0) & long_mask).sum()
    short_wins = ((per_trade > 0) & short_mask).sum()

    gross_log_pnl = per_trade.sum()
    buy_hold_log_pnl = realized.sum()  # equivalent: always go long

    print(f"\n=== {label} ===")
    print(f"Total trades:          {n:,}")
    print(f"  long  signals:       {long_mask.sum():,}")
    print(f"  short signals:       {short_mask.sum():,}")
    print(f"Wins:                  {wins:,}   ({wins / n * 100:.2f}%)")
    print(f"  long wins:           {long_wins:,} / {long_mask.sum():,}  "
          f"({long_wins / max(long_mask.sum(), 1) * 100:.2f}%)")
    print(f"  short wins:          {short_wins:,} / {short_mask.sum():,}  "
          f"({short_wins / max(short_mask.sum(), 1) * 100:.2f}%)")
    print(f"Losses:                {losses:,}   ({losses / n * 100:.2f}%)")
    print(f"Flat (per_trade==0):   {flat:,}")
    print(f"Mean per-trade log-return:  {per_trade.mean() * 10000:+.3f} bps")
    print(f"Median per-trade log-return:{np.median(per_trade) * 10000:+.3f} bps")
    print(f"Gross cumulative log-return (sum):  {gross_log_pnl * 100:+.3f} %")
    print(f"  Buy-and-hold log-return (compare):{buy_hold_log_pnl * 100:+.3f} %")
    print(f"  Annualization factor used downstream: applies, but this is GROSS, no costs.")

    # Standard error of win rate at n trades
    se = np.sqrt(0.5 * 0.5 / n) * 100
    print(f"Standard error of 50% win-rate at n={n}: ~{se:.2f}pp")
    print(f"So a 'real' edge needs roughly >50% + 2*SE = >{50 + 2 * se:.1f}% to be 2-sigma.")


# ---------------------------------------------------------------------------
# 1. INTRADAY: NIFTY 15m, 60d
# ---------------------------------------------------------------------------
df_intra = add_returns(load_bars("NIFTY", "15m", "60d"))
y_intra = df_intra["logret"]
split_i = int(len(y_intra) * 0.8)
best_i, _ = grid_search(y_intra.iloc[:split_i], max_p=3, max_q=3, d=0, trend="n")
arima_intra = walk_forward(
    y_intra, ARIMAForecaster(order=best_i.order, trend="n"), train_size=split_i
)
trading_metrics(arima_intra, f"INTRADAY 15m  ARIMA{best_i.order}  (test bars={len(arima_intra)})")

# ---------------------------------------------------------------------------
# 2. DAILY: NIFTY 1d, 15y
# ---------------------------------------------------------------------------
df_daily = add_returns(load_bars("NIFTY", "1d", "15y"))
y_daily = df_daily["logret"]
split_d = int(len(y_daily) * 0.8)
best_d, _ = grid_search(y_daily.iloc[:split_d], max_p=3, max_q=3, d=0, trend="n")
arima_daily = walk_forward(
    y_daily, ARIMAForecaster(order=best_d.order, trend="n"), train_size=split_d
)
trading_metrics(arima_daily, f"DAILY  1d  ARIMA{best_d.order}  (test days={len(arima_daily)})")
