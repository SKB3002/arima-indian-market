"""Pick a random liquid NSE stock, paper-trade ARIMA + GJR-GARCH on 15m bars.

Standalone runner. Uses everything from src/ but ships the experiment as one
script so it's easy to re-run with a different stock.

Costs are *intraday equity* tier (MIS) which is more expensive than NIFTY
futures because STT on sell side is 25 bps for cash equity vs 1 bps for
futures. This is the realistic cost ceiling for single-stock paper trading.
"""

from __future__ import annotations

import random
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.auto_arima import grid_search
from src.backtest import NSECostModel
from src.data import add_returns, load_bars
from src.diagnostics import check_stationarity, ljung_box, normality
from src.models import ARIMAForecaster
from src.paper_trade import (
    PaperTrader,
    make_combined_rule,
    make_voltarget_long_rule,
    rule_arima_directional,
)
from src.vol_models import GARCHForecaster

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
pd.set_option("display.precision", 6)

# Pool of liquid NSE large-caps (NIFTY 50 constituents, mostly).
# Excluding ones we've already used so this is genuinely fresh.
LIQUID_STOCKS = [
    "INFY.NS", "ITC.NS", "ICICIBANK.NS", "SBIN.NS", "LT.NS",
    "BAJFINANCE.NS", "AXISBANK.NS", "KOTAKBANK.NS", "HINDUNILVR.NS",
    "BHARTIARTL.NS", "MARUTI.NS", "ASIANPAINT.NS", "WIPRO.NS",
    "SUNPHARMA.NS", "TATAMOTORS.NS", "ADANIENT.NS", "TITAN.NS",
    "HCLTECH.NS", "ULTRACEMCO.NS", "POWERGRID.NS", "NTPC.NS",
    "COALINDIA.NS", "JSWSTEEL.NS", "TATASTEEL.NS", "GRASIM.NS",
    "BAJAJFINSV.NS", "HEROMOTOCO.NS", "EICHERMOT.NS",
]

# Seed by today's date for reproducible-within-the-day randomness; the
# user can change/remove this seed for true random.
random.seed(int(pd.Timestamp.now().strftime("%Y%m%d")))
PICK = random.choice(LIQUID_STOCKS)
print(f"=== Picked stock: {PICK} ===\n")


# ---------------------------------------------------------------------------
# 1. Pull data
# ---------------------------------------------------------------------------
df = add_returns(load_bars(PICK, interval="15m", period="60d", use_cache=False))
y = df["logret"]
close = df["close"]
print(f"Bars: {len(y):,}")
print(f"Range: {y.index.min()} -> {y.index.max()}")
print(f"Trading days: {y.index.normalize().nunique()}")
print(f"Most recent close: {close.iloc[-1]:.2f}")
print(f"Most recent 5 bars:")
print(df[["close", "logret", "volume"]].tail(5).to_string())

# ---------------------------------------------------------------------------
# 2. EDA — does this stock look anything like NIFTY?
# ---------------------------------------------------------------------------
print("\n=== EDA ===")
print(f"Mean log-ret per bar: {y.mean() * 10000:+.3f} bps")
print(f"Std log-ret per bar:  {y.std() * 100:.4f}%   (annualized ≈ {y.std() * np.sqrt(252 * 25) * 100:.1f}%)")
print(f"Min bar: {y.min() * 100:+.2f}%   Max bar: {y.max() * 100:+.2f}%")
print(f"\nStationarity:")
print(f"  Close prices:   {check_stationarity(close, regression='ct')}")
print(f"  Log returns:    {check_stationarity(y, regression='c')}")
print(f"  Squared rets:   {check_stationarity(y ** 2, regression='c')}")

lb = ljung_box(y, lags=20)
lb_sq = ljung_box(y ** 2, lags=20)
print(f"\nLjung-Box (returns)  p @ lag 10/20: {lb.loc[10, 'lb_pvalue']:.4f} / {lb.loc[20, 'lb_pvalue']:.4f}")
print(f"Ljung-Box (sq rets)  p @ lag 10/20: {lb_sq.loc[10, 'lb_pvalue']:.4g} / {lb_sq.loc[20, 'lb_pvalue']:.4g}")

n = normality(y)
print(f"Normality: kurt_excess={n['excess_kurtosis']:.2f}  skew={n['skew']:.2f}  JB-p={n['pvalue']:.4f}")

# ---------------------------------------------------------------------------
# 3. Split + fit models
# ---------------------------------------------------------------------------
days = pd.Index(sorted(y.index.normalize().unique()))
PAPER_DAYS = 12
cutoff = days[-PAPER_DAYS]
train_mask = y.index < cutoff
paper_mask = y.index >= cutoff
y_train, y_paper = y[train_mask], y[paper_mask]
close_paper = close[paper_mask]
print(f"\n=== Split ===")
print(f"Train: {y_train.index.min()} -> {y_train.index.max()}  ({len(y_train):,} bars)")
print(f"Paper: {y_paper.index.min()} -> {y_paper.index.max()}  ({len(y_paper):,} bars, {y_paper.index.normalize().nunique()} days)")

best, _ = grid_search(y_train, max_p=3, max_q=3, d=0, trend="n")
print(f"\nARIMA picked: {best}")

arima = ARIMAForecaster(order=best.order, trend="n"); arima.fit(y_train)
gjr = GARCHForecaster(asymmetric=True, dist="t", refit_every=None); gjr.fit(y_train)
print(f"GJR-GARCH params: mu={gjr._mu:.4f}  omega={gjr._omega:.5f}  "
      f"alpha={gjr._alpha:.4f}  gamma={gjr._gamma:.4f}  beta={gjr._beta:.4f}")

# ---------------------------------------------------------------------------
# 4. Paper-trade loop
# ---------------------------------------------------------------------------
cost = NSECostModel.intraday_equity()
TARGET_VOL = 0.0040  # 0.4% per 15m bar (stocks are more volatile than index)
PPY = 6300.0
print(f"\nCost (single-stock intraday MIS): round-trip {cost.round_trip_bps:.2f} bps")
print(f"Target per-bar vol: {TARGET_VOL * 100:.2f}% (≈ {TARGET_VOL * np.sqrt(PPY) * 100:.1f}% ann.)")

traders = {
    "arima_directional":  PaperTrader(rule_arima_directional, cost, name="arima_directional"),
    "gjr_voltarget_long": PaperTrader(make_voltarget_long_rule(TARGET_VOL), cost, name="gjr_voltarget_long"),
    "combined":           PaperTrader(make_combined_rule(TARGET_VOL), cost, name="combined"),
}

prev_r = arima.forecast_next(); prev_v = gjr.forecast_next_var()
for ts, c, r in zip(y_paper.index, close_paper.to_numpy(), y_paper.to_numpy()):
    arima.update(float(r)); gjr.update(float(r))
    nr = arima.forecast_next(); nv = gjr.forecast_next_var()
    for t in traders.values():
        t.observe_bar(ts=ts, close=float(c), realized_logret=float(r),
                      forecast_ret_for_this_bar=prev_r, forecast_var_for_this_bar=prev_v,
                      forecast_ret_for_next_bar=nr, forecast_var_for_next_bar=nv)
    prev_r, prev_v = nr, nv

# ---------------------------------------------------------------------------
# 5. Report
# ---------------------------------------------------------------------------
# Buy-and-hold benchmark
bh_eq = float((1 + y_paper).cumprod().iloc[-1])
bh_ret = bh_eq - 1.0
bh_sd = float(y_paper.std(ddof=1))
bh_mean = float(y_paper.mean())
bh_sharpe = bh_mean / bh_sd * np.sqrt(PPY) if bh_sd > 0 else float("nan")
bh_dd = float(((1 + y_paper).cumprod().to_numpy()
               / np.maximum.accumulate((1 + y_paper).cumprod().to_numpy()) - 1).min())

print(f"\n=== Buy-and-hold over paper window ===")
print(f"Total return: {bh_ret * 100:+.2f}%   Sharpe: {bh_sharpe:.2f}   Max DD: {bh_dd * 100:.2f}%")

summary = pd.DataFrame([t.summary(PPY) for t in traders.values()]).set_index("name")
cols = ["n_bars", "n_trades_with_pos", "n_position_changes", "win_rate",
        "total_return", "ann_vol", "sharpe", "sortino", "max_drawdown",
        "total_cost", "final_equity"]
print(f"\n=== Strategy results ===")
print(summary[cols].to_string())

# ---------------------------------------------------------------------------
# 6. Today's bars (truly fresh)
# ---------------------------------------------------------------------------
last_day = traders["arima_directional"].blotter_df().index.normalize().max()
print(f"\n=== Per-bar log on {last_day.date()} (most recent trading day) ===")
for name, trader in traders.items():
    bl = trader.blotter_df()
    today = bl[bl.index.normalize() == last_day]
    if len(today) == 0:
        continue
    print(f"\n--- {name} ---")
    print(today[["close", "realized_logret", "forecast_ret", "target_position",
                 "delta_pos", "cost", "gross_pnl", "net_pnl", "equity", "note"]].to_string())
