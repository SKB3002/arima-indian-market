# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       format_name: percent
# ---

# %% [markdown]
# # 02 — Baselines and a first ARIMA
#
# Notebook 01 told us, before fitting anything, that:
#
# - log-returns are stationary,
# - Ljung-Box can't reject independence at any lag up to 20,
# - returns are extremely fat-tailed (excess kurtosis ~44).
#
# So our prior is that ARIMA on returns will *not* beat a naïve zero forecast.
# We will now show that rigorously rather than assume it. The walk-forward
# harness is the spine for everything downstream (SARIMA, ARIMAX, ARIMA-GARCH).

# %%
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path.cwd()
if (ROOT / "src").is_dir():
    sys.path.insert(0, str(ROOT))
elif (ROOT.parent / "src").is_dir():
    sys.path.insert(0, str(ROOT.parent))

import numpy as np
import pandas as pd

from src.auto_arima import grid_search
from src.data import add_returns, load_bars
from src.models import (
    ARIMAForecaster,
    EWMAForecaster,
    NaiveForecaster,
    diebold_mariano,
    score_forecasts,
    walk_forward,
)

pd.set_option("display.width", 120)
pd.set_option("display.precision", 6)

# %% [markdown]
# ## 1. Data + train/test split
#
# 60 days of NIFTY 15m bars → ~1450 bars. We hold out the last 20% as the
# walk-forward test set; the first 80% feeds order selection and the initial
# model fit.

# %%
df = add_returns(load_bars("NIFTY", interval="15m", period="60d"))
y = df["logret"]

split = int(len(y) * 0.8)
y_train, y_test = y.iloc[:split], y.iloc[split:]
print(f"Train bars: {len(y_train):,}   ({y_train.index.min()} -> {y_train.index.max()})")
print(f"Test  bars: {len(y_test):,}    ({y_test.index.min()} -> {y_test.index.max()})")

# %% [markdown]
# ## 2. AIC grid search for (p, q)
#
# We restrict to ARIMA(p, 0, q) with no trend since the series is mean-zero and
# stationary by construction (already log-differenced).

# %%
best_order, grid = grid_search(y_train, max_p=3, max_q=3, d=0, trend="n")
print(grid.head(10))
print(f"\nBest by AIC: {best_order}")

# %% [markdown]
# **Reading the grid.** When AIC is essentially flat across orders, the model
# is just adding parameters without finding structure — exactly what we'd
# expect from a white-noise series. Differences of <2 in AIC are usually
# considered non-significant evidence in favor of the lower-AIC model.

# %% [markdown]
# ## 3. Walk-forward backtest — three models

# %%
naive_results = walk_forward(y, NaiveForecaster(), train_size=split)
ewma_results = walk_forward(y, EWMAForecaster(halflife=25.0), train_size=split)
arima_results = walk_forward(
    y, ARIMAForecaster(order=best_order.order, trend="n"), train_size=split
)

# %% [markdown]
# ## 4. Score table

# %%
table = pd.DataFrame(
    {
        "naive":  score_forecasts(naive_results),
        "ewma":   score_forecasts(ewma_results),
        f"arima{best_order.order}": score_forecasts(arima_results),
    }
).T
print(table)
table

# %% [markdown]
# ## 5. Diebold-Mariano vs the naïve baseline
#
# The DM stat measures the mean of (loss_model − loss_naive). A *negative* mean
# diff means the model has lower loss than naive. The p-value is two-sided
# and uses the standard normal approximation.

# %%
dm_ewma = diebold_mariano(ewma_results, naive_results, loss="se")
dm_arima = diebold_mariano(arima_results, naive_results, loss="se")
print("EWMA  vs naive (SE loss):", dm_ewma)
print("ARIMA vs naive (SE loss):", dm_arima)

# %% [markdown]
# ## 6. Diagnostic plot — forecast vs realised
#
# A useful sanity check: ARIMA forecasts on near-noise should be tiny in
# magnitude and almost flat. If you see anything that looks like a "signal,"
# it's almost certainly overfitting.

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(11, 4))
arima_results["actual"].plot(ax=ax, lw=0.5, color="gray", label="actual")
arima_results["forecast"].plot(ax=ax, lw=1.0, color="firebrick", label="ARIMA forecast")
ax.axhline(0, color="black", lw=0.5)
ax.set_title("ARIMA one-step forecast vs realised log returns (test set)")
ax.legend()
plt.tight_layout()

# %% [markdown]
# ## Takeaways
#
# - If RMSE for ARIMA is within ~0.5% of naïve, and directional accuracy is
#   ~50%, our EDA prior is confirmed: pure ARIMA on 15m log-returns is not a
#   forecasting tool, it's an existence proof of the efficient-market
#   hypothesis at this frequency.
# - This is the *correct* result to find on liquid index returns. The point of
#   this notebook isn't to win; it's to establish a defensible "ARIMA can't do
#   it" benchmark, against which SARIMA / ARIMAX / ARIMA-GARCH will be
#   compared in later notebooks.
# - Next: deseasonalize (or SARIMA with s=25) and re-run. Then add exogenous
#   regressors. Then model *volatility* with ARIMA-GARCH, where we expect
#   real predictive performance.
