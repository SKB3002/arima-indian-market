# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       format_name: percent
# ---

# %% [markdown]
# # 01 — EDA: NIFTY 50 at 15-minute bars
#
# **Goal of this notebook.** Before we model anything, audit the data against
# every assumption an ARIMA model is going to make:
#
# 1. Stationarity (ADF + KPSS, on both prices and log-returns)
# 2. Linear autocorrelation (ACF / PACF on returns — likely near zero)
# 3. Nonlinear autocorrelation (ACF on **squared** returns — where volatility
#    clustering hides)
# 4. Intraday seasonality (the U-shape: open + close volatility spikes)
# 5. Distribution shape (heavy tails, JB test)
# 6. Volatility clustering (visual)
#
# Each section ends with a one-line verdict on whether classical ARIMA is
# defensible for that piece of the signal.

# %%
from __future__ import annotations

import sys
from pathlib import Path

# Make `src` importable when running this file from anywhere
ROOT = Path.cwd()
if (ROOT / "src").is_dir():
    sys.path.insert(0, str(ROOT))
elif (ROOT.parent / "src").is_dir():
    sys.path.insert(0, str(ROOT.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from src.data import add_returns, load_bars
from src.diagnostics import (
    check_stationarity,
    intraday_seasonality,
    ljung_box,
    normality,
)

pd.set_option("display.width", 120)
plt.rcParams["figure.figsize"] = (11, 4)

# %% [markdown]
# ## 1. Load NIFTY 50 — 15-minute bars (last ~60 days)

# %%
df = load_bars("NIFTY", interval="15m", period="60d")
df = add_returns(df, price_col="close")
print(f"Bars: {len(df):,}")
print(f"Range: {df.index.min()}  to  {df.index.max()}")
print(f"Trading days: {df.index.normalize().nunique()}")
df.head()

# %% [markdown]
# ## 2. Price and returns — eyeball the series

# %%
fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
df["close"].plot(ax=axes[0], lw=0.8, title="NIFTY 50 close (15m bars)")
axes[0].set_ylabel("price")
df["logret"].plot(ax=axes[1], lw=0.6, color="darkorange", title="Log returns")
axes[1].axhline(0, color="black", lw=0.5)
axes[1].set_ylabel("log return")
plt.tight_layout()

# %% [markdown]
# ## 3. Stationarity — ADF + KPSS together
#
# We run both because they have opposite null hypotheses:
#
# - **ADF** null = unit root (non-stationary). Small p → stationary.
# - **KPSS** null = stationary. Large p → stationary.
#
# Agreement between the two is the verdict you can actually trust.

# %%
print("Close prices:   ", check_stationarity(df["close"], regression="ct"))
print("Log returns:    ", check_stationarity(df["logret"], regression="c"))
print("Squared returns:", check_stationarity(df["logret"] ** 2, regression="c"))

# %% [markdown]
# **Expected.** Close is non-stationary (trends + level). Log returns are
# stationary in the mean — this justifies the "I=1" in ARIMA(p,1,q) on the
# price, or equivalently ARMA(p,q) on log-returns. Squared returns are also
# stationary but, as we'll see in §5, are *very* autocorrelated.

# %% [markdown]
# ## 4. Linear autocorrelation — ACF / PACF of returns
#
# If markets were even mildly inefficient at 15m, we'd see significant spikes
# in the return ACF. On a liquid index we expect near-noise, perhaps a small
# negative lag-1 (bid-ask bounce on the constituents).

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
plot_acf(df["logret"], lags=40, ax=axes[0], title="ACF — log returns")
plot_pacf(df["logret"], lags=40, ax=axes[1], method="ywm", title="PACF — log returns")
plt.tight_layout()

# %%
lb = ljung_box(df["logret"], lags=20)
print("Ljung-Box on returns (H0: no autocorrelation up to lag k):")
print(lb.tail(5))
print(f"\nAny lag with p < 0.05? {(lb['lb_pvalue'] < 0.05).any()}")

# %% [markdown]
# **Verdict for return forecasting.** If Ljung-Box can't reject independence,
# an ARIMA(p,0,q) on log-returns has essentially no signal to fit. We'll still
# fit it — but expect it to lose to a naïve zero-return baseline.

# %% [markdown]
# ## 5. Nonlinear autocorrelation — ACF of squared returns
#
# This is where volatility lives. Even when r_t looks like noise, r_t² is
# typically very predictable: big moves cluster.

# %%
fig, ax = plt.subplots(figsize=(11, 3.5))
plot_acf(df["logret"] ** 2, lags=80, ax=ax, title="ACF — squared log returns (volatility memory)")
plt.tight_layout()

# %%
lb_sq = ljung_box(df["logret"] ** 2, lags=20)
print("Ljung-Box on squared returns:")
print(lb_sq.tail(5))

# %% [markdown]
# **Verdict for volatility forecasting.** Long, slow decay in the ACF of r²
# is the canonical signature of ARCH/GARCH effects — i.e. ARIMA-GARCH has
# something real to chew on, even when plain ARIMA does not.

# %% [markdown]
# ## 6. Intraday seasonality — the U-shape
#
# Indian equities show heightened volatility at the open (09:15) and close
# (15:15 bar), and a quiet trough around 12:00–13:30. ARIMA is shift-invariant
# and cannot model time-of-day; we'll either deseasonalize first or move to
# SARIMA with seasonal period s = 25 (bars per day).

# %%
season = intraday_seasonality(df["logret"])
fig, ax = plt.subplots(figsize=(11, 3.5))
season["mean"].plot(kind="bar", ax=ax, color="steelblue")
ax.set_title("Mean |log return| by time of day (15m bars)")
ax.set_ylabel("avg |return|")
ax.set_xlabel("IST")
plt.tight_layout()
season

# %% [markdown]
# ## 7. Distribution shape — fat tails check
#
# Returns are famously non-Gaussian. JB tests for normality; we expect it to
# reject overwhelmingly. The excess kurtosis number is the one that matters
# for risk modelling.

# %%
norm = normality(df["logret"])
for k, v in norm.items():
    print(f"  {k:18s} {v:.4f}")

# %%
from scipy import stats

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
df["logret"].hist(bins=80, ax=axes[0], color="steelblue", density=True, alpha=0.7)
x = np.linspace(df["logret"].min(), df["logret"].max(), 200)
axes[0].plot(x, stats.norm.pdf(x, df["logret"].mean(), df["logret"].std()),
             "r-", lw=1.5, label="Normal fit")
axes[0].set_title("Return distribution vs Normal")
axes[0].legend()
stats.probplot(df["logret"].dropna(), dist="norm", plot=axes[1])
axes[1].set_title("QQ plot vs Normal (departure at tails = fat tails)")
plt.tight_layout()

# %% [markdown]
# ## 8. Volatility clustering — visual
#
# Plot |returns|. If quiet periods cluster with quiet, and storms cluster with
# storms, the GARCH story is already telling itself.

# %%
fig, ax = plt.subplots(figsize=(11, 3.5))
df["logret"].abs().plot(ax=ax, lw=0.5, color="firebrick")
ax.set_title("|log return| — clustering of high-vol regimes")
plt.tight_layout()

# %% [markdown]
# ## Takeaways heading into modelling
#
# - Log-returns are stationary → ARIMA(p, 0, q) on returns is the right form
#   (equivalent to ARIMA(p, 1, q) on prices).
# - Linear autocorrelation in returns is weak → expect ARIMA-on-returns to
#   barely outperform naïve. **Document the failure rigorously**; it's the
#   honest finding.
# - Strong autocorrelation in *squared* returns → ARIMA-GARCH is the move for
#   volatility forecasting, and that's where tradable edge probably lives.
# - Strong intraday U-shape → either deseasonalize first or use SARIMA with
#   seasonal period s = 25 (bars/day) for 15m data.
# - Heavy tails, high kurtosis → switch the GARCH innovation distribution
#   from Normal to Student-t when we get there.
