# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       format_name: percent
# ---

# %% [markdown]
# # 05 — Strategy backtest: ARIMA directional vs GARCH vol-targeting
#
# Three strategies on 15 years of daily NIFTY, side-by-side against buy-and-hold:
#
# 1. **ARIMA-direction**: position = sign(forecast). The strawman from NB02/04.
#    We expect it to look like buy-and-hold gross (drift bleed-through) and
#    materially worse net of costs.
# 2. **GJR-GARCH vol-targeting (long-only)**: position = target_vol /
#    forecast_sigma, capped at max_leverage, direction always +1. Doesn't try
#    to predict direction — it scales index exposure inversely to forecast
#    risk. This is the strategy where the QLIKE win matters.
# 3. **Combined**: ARIMA sign × GARCH-scaled size. Sanity check on whether the
#    weak directional signal does anything useful when sized properly.
#
# Cost assumptions are NIFTY-futures-tier (Zerodha-style discount brokerage):
# ~9 bps per leg / ~18 bps round-trip. Easy to swap to delivery-equity costs
# for sensitivity testing.

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
from src.backtest import (
    NSECostModel,
    backtest_directional,
    backtest_voltarget,
    buy_and_hold,
    strategy_metrics,
)
from src.data import add_returns, load_bars
from src.models import ARIMAForecaster, walk_forward
from src.vol_models import GARCHForecaster, walk_forward_vol

warnings.filterwarnings("ignore")
pd.set_option("display.width", 140)
pd.set_option("display.precision", 4)
plt.rcParams["figure.figsize"] = (12, 4)

# %% [markdown]
# ## 1. Load data and split

# %%
df = add_returns(load_bars("NIFTY", interval="1d", period="15y"))
y = df["logret"]
split = int(len(y) * 0.8)
test_start = y.index[split]
print(f"Total bars:  {len(y):,}")
print(f"Train:       {y.index[0].date()} -> {y.index[split-1].date()}  ({split:,})")
print(f"Test:        {test_start.date()} -> {y.index[-1].date()}  ({len(y) - split:,})")

# %% [markdown]
# ## 2. Generate forecasts (return signal + vol signal) over the test window

# %%
# Auto-pick ARIMA order on training set, then walk forward
y_train = y.iloc[:split]
best_order, _ = grid_search(y_train, max_p=3, max_q=3, d=0, trend="n")
print(f"ARIMA order chosen on train: {best_order}")

arima_forecasts = walk_forward(
    y, ARIMAForecaster(order=best_order.order, trend="n"), train_size=split
)
print(f"ARIMA forecasts produced:    {len(arima_forecasts):,}")

# GJR-GARCH walk-forward with annual refit
gjr_results = walk_forward_vol(
    y, GARCHForecaster(asymmetric=True, dist="t", refit_every=252), train_size=split
)
print(f"GARCH forecasts produced:    {len(gjr_results):,}")

# Align: both forecasters output on the same test slice.
assert (arima_forecasts.index == gjr_results.index).all()

# %% [markdown]
# ## 3. Cost model
#
# We use NIFTY-futures tier (intraday or rollover). Most realistic for an
# index strategy. Run the same backtests with `delivery_equity()` later to see
# how much more cost would hurt a cash-equity implementation.

# %%
cost = NSECostModel.intraday_futures()
print(f"Per-leg cost:    buy={cost.buy_leg_bps:.2f} bps   sell={cost.sell_leg_bps:.2f} bps")
print(f"Round-trip cost: {cost.round_trip_bps:.2f} bps")

# %% [markdown]
# ## 4. Strategy A — ARIMA directional

# %%
bt_arima = backtest_directional(
    forecasts=arima_forecasts["forecast"],
    actual_returns=arima_forecasts["actual"],
    cost=cost,
)
m_arima = strategy_metrics(bt_arima, periods_per_year=252.0)
print({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in m_arima.items()})

# %% [markdown]
# ## 5. Strategy B — GJR-GARCH vol-targeting (always long)
#
# Target 0.7% per day ≈ 11% annualized. NIFTY's realized vol over the sample
# is ~16.5%; target_vol < realized means the strategy will run lower exposure
# on average. The interesting question: does it scale down enough during
# stress to *improve* risk-adjusted return vs buy-and-hold?

# %%
TARGET_DAILY_VOL = 0.007  # 0.7% per day ≈ 11% annualized

bt_voltarget = backtest_voltarget(
    vol_forecasts=gjr_results["forecast_var"],
    actual_returns=gjr_results["actual_ret"],
    cost=cost,
    target_vol_per_bar=TARGET_DAILY_VOL,
    max_leverage=2.0,
    direction=None,  # always long
)
m_voltarget = strategy_metrics(bt_voltarget, periods_per_year=252.0)
print({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in m_voltarget.items()})

# %% [markdown]
# ## 6. Strategy C — Combined (ARIMA sign × GARCH-scaled size)

# %%
direction = pd.Series(
    np.sign(arima_forecasts["forecast"].to_numpy()),
    index=arima_forecasts.index,
)
bt_combined = backtest_voltarget(
    vol_forecasts=gjr_results["forecast_var"],
    actual_returns=gjr_results["actual_ret"],
    cost=cost,
    target_vol_per_bar=TARGET_DAILY_VOL,
    max_leverage=2.0,
    direction=direction,
)
m_combined = strategy_metrics(bt_combined, periods_per_year=252.0)
print({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in m_combined.items()})

# %% [markdown]
# ## 7. Benchmark — buy-and-hold

# %%
bt_bh = buy_and_hold(arima_forecasts["actual"])
m_bh = strategy_metrics(bt_bh, periods_per_year=252.0)
print({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in m_bh.items()})

# %% [markdown]
# ## 8. Side-by-side comparison

# %%
def pct(x: float) -> str:
    return f"{x * 100:+.2f}%" if np.isfinite(x) else "n/a"


table = pd.DataFrame(
    {
        "buy_and_hold":      m_bh,
        "arima_directional": m_arima,
        "gjr_voltarget_long": m_voltarget,
        "arima_sign_x_garch_size": m_combined,
    }
).T

# Pretty-print the key columns
display_cols = ["n", "n_trades", "win_rate", "cagr", "ann_vol",
                "sharpe", "sortino", "max_drawdown",
                "total_return", "turnover", "total_cost"]
print(table[display_cols])

# %% [markdown]
# ## 9. Equity curves

# %%
fig, ax = plt.subplots(figsize=(12, 5))
for label, bt in [
    ("buy_and_hold", bt_bh),
    ("arima_directional", bt_arima),
    ("gjr_voltarget_long", bt_voltarget),
    ("arima_sign_x_garch_size", bt_combined),
]:
    ax.plot(bt.index, bt["equity"], lw=1.1, label=label)
ax.set_title("Strategy equity curves — daily NIFTY test window")
ax.set_ylabel("equity (1 + cumulative net return)")
ax.legend(loc="upper left")
ax.axhline(1.0, color="black", lw=0.4)
plt.tight_layout()

# %% [markdown]
# ## 10. Drawdown curves
#
# Same equity curves expressed as drawdown from running peak. A vol-targeting
# strategy that does its job should show a *shallower* worst drawdown than
# buy-and-hold, even if total return is similar.

# %%
fig, ax = plt.subplots(figsize=(12, 4))
for label, bt in [
    ("buy_and_hold", bt_bh),
    ("gjr_voltarget_long", bt_voltarget),
    ("arima_directional", bt_arima),
]:
    eq = bt["equity"].to_numpy()
    dd = eq / np.maximum.accumulate(eq) - 1.0
    ax.plot(bt.index, dd * 100, lw=1.0, label=label)
ax.set_title("Drawdown from running peak (%)")
ax.legend(loc="lower left")
plt.tight_layout()

# %% [markdown]
# ## 11. Cost sensitivity — what if we used delivery equity instead of futures?

# %%
cost_delivery = NSECostModel.delivery_equity()
print(f"Delivery round-trip cost: {cost_delivery.round_trip_bps:.2f} bps")

bt_arima_del = backtest_directional(
    arima_forecasts["forecast"], arima_forecasts["actual"], cost=cost_delivery
)
bt_voltarget_del = backtest_voltarget(
    gjr_results["forecast_var"], gjr_results["actual_ret"],
    cost=cost_delivery, target_vol_per_bar=TARGET_DAILY_VOL, max_leverage=2.0,
)

print("\nWith delivery-equity costs:")
sens = pd.DataFrame({
    "arima_directional": strategy_metrics(bt_arima_del),
    "gjr_voltarget_long": strategy_metrics(bt_voltarget_del),
}).T[display_cols]
print(sens)

# %% [markdown]
# ## 12. Interpretation guide
#
# Read the table top-down on a few axes:
#
# - **CAGR**: total return per year, compounded. The bottom line.
# - **Ann vol**: realized strategy volatility, annualized. Vol-targeting should
#   land near our target of ~11%.
# - **Sharpe**: (return - risk_free) / vol, annualized. The risk-adjusted
#   judge. > 0.5 starts to be interesting; > 1.0 is genuinely good.
# - **Max drawdown**: worst peak-to-trough loss. The pain we'd have endured.
#   For NIFTY buy-and-hold over this window, this is essentially COVID-2020.
# - **Turnover** (sum of |Δposition|): proxy for cost burden. Directional
#   flips every bar = turnover ≈ 2 * (number of sign flips). Vol-targeting
#   has continuous small adjustments → much smaller per-bar turnover.
# - **Total cost** (fraction of NAV paid in fees over the test window).
#
# **What to expect / interpret**
#
# 1. *ARIMA directional* should have CAGR roughly matching buy-and-hold gross
#    (drift bleed-through) but with massive `total_cost` from constant
#    sign-flipping. Net CAGR should be visibly worse.
# 2. *GJR vol-targeting long-only* should have lower realized vol than buy-
#    and-hold by design. If vol-targeting works, its Sharpe is *higher* even
#    though its CAGR may be lower — the win is risk-adjusted, not absolute.
# 3. *Combined ARIMA-sign × GARCH-size* — the most generous test of the
#    return-direction model, because the size is intelligent. If this still
#    loses to vol-targeting-long-only, the directional signal is conclusively
#    not adding value.
#
# This is the project's "does any of this make money" moment. The answer is
# rarely "yes, gobs of it" — but a positive risk-adjusted result for the
# vol-targeting variant would be a real win, and any negative result is
# itself a publishable finding ("classical ARIMA-GARCH on Indian intraday
# index data does not generate net-of-cost alpha").
