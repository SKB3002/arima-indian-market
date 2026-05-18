# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       format_name: percent
# ---

# %% [markdown]
# # 06 — Intraday paper trading on fresh NIFTY 15m bars
#
# Until now every backtest has been vectorized — fast, but the per-bar
# decision-making has been hidden. This notebook runs the simulator
# **bar-by-bar** the way it would happen in production:
#
# 1. A new 15-min bar arrives.
# 2. We realize the P&L on whatever position we were holding.
# 3. We run ARIMA + GJR-GARCH on history-up-to-now and generate a forecast
#    for the *next* bar.
# 4. We update our target position based on the rule (directional,
#    vol-target, combined), paying the NSE round-trip cost on any change.
# 5. We log the trade.
#
# We refresh data each run so the test window includes the **most recent
# trading day's bars** as truly never-seen-before out-of-sample.
#
# Three rules tested:
# - `arima_directional`  : position = sign(ARIMA forecast)
# - `gjr_voltarget_long` : always long, sized = target_vol / forecast_sigma
# - `combined`           : ARIMA sign × GARCH-scaled size

# %%
from __future__ import annotations

import sys
import warnings
from pathlib import Path

ROOT = Path.cwd()
if (ROOT / "src").is_dir():
    sys.path.insert(0, str(ROOT))
elif (ROOT.parent / "src").is_dir():
    sys.path.insert(0, str(ROOT.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.auto_arima import grid_search
from src.backtest import NSECostModel
from src.data import add_returns, load_bars
from src.models import ARIMAForecaster
from src.paper_trade import (
    PaperTrader,
    make_combined_rule,
    make_voltarget_long_rule,
    rule_arima_directional,
)
from src.vol_models import GARCHForecaster

warnings.filterwarnings("ignore")
pd.set_option("display.width", 160)
pd.set_option("display.max_rows", 100)
pd.set_option("display.precision", 6)
plt.rcParams["figure.figsize"] = (12, 4)

# %% [markdown]
# ## 1. Pull the latest NIFTY 15m data (force-refresh)

# %%
df = add_returns(load_bars("NIFTY", interval="15m", period="60d", use_cache=False))
y = df["logret"]
close = df["close"]
print(f"Bars: {len(y):,}")
print(f"Range: {y.index.min()} to {y.index.max()}")
print(f"Trading days: {y.index.normalize().nunique()}")
print("\nLatest 8 bars:")
print(df.tail(8)[["close", "logret"]])

# %% [markdown]
# ## 2. Define training / paper-trade split
#
# We pick a cutoff date so that "training" includes everything up to that day,
# and "paper trading" runs across all subsequent bars — including any new
# bars from the current trading day.

# %%
# Use roughly the last 12 trading days as the paper-trade window
all_days = pd.Index(sorted(y.index.normalize().unique()))
paper_days = all_days[-12:]
cutoff_ts = paper_days[0]
train_mask = y.index < cutoff_ts
paper_mask = y.index >= cutoff_ts

print(f"Cutoff timestamp:   {cutoff_ts}")
print(f"Training bars:      {train_mask.sum():,}  ({y.index[train_mask].min()} -> {y.index[train_mask].max()})")
print(f"Paper-trade bars:   {paper_mask.sum():,}  ({y.index[paper_mask].min()} -> {y.index[paper_mask].max()})")
print(f"Trading days in paper window: {y.index[paper_mask].normalize().nunique()}")

y_train = y[train_mask]
y_paper = y[paper_mask]
close_paper = close[paper_mask]

# %% [markdown]
# ## 3. Fit initial models on training data only

# %%
best_order, _ = grid_search(y_train, max_p=3, max_q=3, d=0, trend="n")
print(f"ARIMA order chosen on train: {best_order}")

arima = ARIMAForecaster(order=best_order.order, trend="n")
arima.fit(y_train)

gjr = GARCHForecaster(asymmetric=True, dist="t", refit_every=None)
gjr.fit(y_train)
print(
    f"GJR-GARCH params: mu={gjr._mu:.4f}  omega={gjr._omega:.6f}  "
    f"alpha={gjr._alpha:.4f}  gamma={gjr._gamma:.4f}  beta={gjr._beta:.4f}"
)

# %% [markdown]
# ## 4. Configure the paper traders
#
# Cost model: NIFTY futures tier (~18 bps round-trip). Target 0.25% per 15m bar
# vol for the vol-targeting rules — that's about 12.6% annualized
# (sqrt(6300) ≈ 79.4 ; 0.25% × 79.4 ≈ 19.8%/yr; we land lower in practice).
# Adjust if you want lower/higher exposure.

# %%
cost = NSECostModel.intraday_futures()
TARGET_VOL_PER_BAR = 0.0025   # 0.25% per 15-min bar
PERIODS_PER_YEAR_15M = 252.0 * 25.0  # 6300

print(f"Round-trip cost: {cost.round_trip_bps:.2f} bps")
print(f"Target per-bar vol: {TARGET_VOL_PER_BAR * 100:.3f}% (≈ {TARGET_VOL_PER_BAR * np.sqrt(PERIODS_PER_YEAR_15M) * 100:.1f}% annualized)")

traders = {
    "arima_directional":  PaperTrader(rule_arima_directional, cost, name="arima_directional"),
    "gjr_voltarget_long": PaperTrader(
        make_voltarget_long_rule(TARGET_VOL_PER_BAR), cost, name="gjr_voltarget_long"
    ),
    "combined":           PaperTrader(
        make_combined_rule(TARGET_VOL_PER_BAR), cost, name="combined"
    ),
}

# %% [markdown]
# ## 5. Replay the paper-trade loop bar by bar
#
# This is what the simulator would do live. At each bar:
# 1. ARIMA produces a forecast for THIS bar (made at the prior bar's close)
# 2. GJR-GARCH does the same for variance
# 3. We realize P&L on whatever position we held into THIS bar
# 4. We update models with the actual observation
# 5. We produce NEW forecasts (for the NEXT bar) and reposition

# %%
# We carry forecasts across iterations: the *next-bar* forecast made at
# step t becomes the *this-bar* forecast that drives step t+1's realisation.
prev_ret_fc = arima.forecast_next()
prev_var_fc = gjr.forecast_next_var()

paper_bars = list(zip(y_paper.index, close_paper.to_numpy(), y_paper.to_numpy()))

for ts, c, r in paper_bars:
    # Update models with the observation that JUST realized
    arima.update(float(r))
    gjr.update(float(r))
    # New forecasts for the bar that will close NEXT
    next_ret_fc = arima.forecast_next()
    next_var_fc = gjr.forecast_next_var()
    # Have each trader observe and act
    for t in traders.values():
        t.observe_bar(
            ts=ts,
            close=float(c),
            realized_logret=float(r),
            forecast_ret_for_this_bar=prev_ret_fc,
            forecast_var_for_this_bar=prev_var_fc,
            forecast_ret_for_next_bar=next_ret_fc,
            forecast_var_for_next_bar=next_var_fc,
        )
    prev_ret_fc, prev_var_fc = next_ret_fc, next_var_fc

print(f"Replay complete. Processed {len(paper_bars):,} bars.")

# %% [markdown]
# ## 6. Per-trader summary table

# %%
summary = pd.DataFrame([t.summary(PERIODS_PER_YEAR_15M) for t in traders.values()]).set_index("name")
print(summary)

# %% [markdown]
# ## 7. Trade blotter — most recent 20 bars (each strategy)
#
# This is the per-bar decision log you would log to disk in a live system.

# %%
for name, trader in traders.items():
    print(f"\n=== {name} — last 20 bars ===")
    bl = trader.blotter_df()
    bl_show = bl[["close", "realized_logret", "forecast_ret",
                  "prev_position", "target_position", "delta_pos",
                  "cost", "gross_pnl", "net_pnl", "equity", "note"]].tail(20)
    print(bl_show.to_string())

# %% [markdown]
# ## 8. Highlight TODAY's bars only (the truly fresh out-of-sample slice)
#
# yfinance was refreshed at notebook run; the last trading date in the
# blotter is today (or last open day). We zoom in.

# %%
last_day = traders["arima_directional"].blotter_df().index.normalize().max()
print(f"Most recent trading day in the paper-trade window: {last_day.date()}")

for name, trader in traders.items():
    bl = trader.blotter_df()
    today_bl = bl[bl.index.normalize() == last_day]
    if len(today_bl) == 0:
        continue
    print(f"\n--- {name} on {last_day.date()} ---")
    print(today_bl[["close", "realized_logret", "target_position",
                    "delta_pos", "cost", "gross_pnl", "net_pnl", "equity", "note"]].to_string())

# %% [markdown]
# ## 9. Equity curves over the paper-trade window

# %%
fig, ax = plt.subplots(figsize=(12, 5))
buy_hold_eq = (1.0 + y_paper).cumprod()
ax.plot(buy_hold_eq.index, buy_hold_eq.values, lw=1.2, color="black", label="buy_and_hold (gross)")
for name, trader in traders.items():
    bl = trader.blotter_df()
    ax.plot(bl.index, bl["equity"], lw=1.0, label=name)
ax.set_title(f"15m paper-trade equity over {paper_mask.sum()} bars "
             f"({y_paper.index.min().date()} -> {y_paper.index.max().date()})")
ax.legend(loc="upper left")
ax.axhline(1.0, color="gray", lw=0.4)
plt.tight_layout()

# %% [markdown]
# ## 10. Take-aways
#
# Read the summary table top-down:
#
# - **win_rate** is per-bar hit rate (sign(position) × actual_return > 0,
#   counting only bars where we held a non-zero position).
# - **n_position_changes** drives **total_cost** — at 18 bps round-trip and
#   intraday flipping, the directional strategy pays an enormous cost bill.
# - **sharpe** annualised at √6300 — intraday vol is much smaller per bar so
#   small per-bar mean returns multiply into big Sharpe numbers, both directions.
# - The **today's bars** section is the most honest forward test we can do
#   without paying for a real-time feed.
#
# If `gjr_voltarget_long` has higher Sharpe than `buy_and_hold` and a smaller
# max drawdown, the vol-targeting story has merit at intraday too. If
# `arima_directional` looks anything other than terrible, double-check —
# it's almost certainly the cost model that's letting it look ok.
