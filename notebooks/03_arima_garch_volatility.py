# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       format_name: percent
# ---

# %% [markdown]
# # 03 — Volatility forecasting with GARCH(1,1) and GJR-GARCH
#
# Notebook 02 confirmed empirically what notebook 01 predicted: ARIMA on 15m
# NIFTY returns cannot beat a zero forecast. So we pivot the question:
#
# > Forget direction. Can we forecast **magnitude** — how much the market
# > will move in the next 15 minutes?
#
# This is what GARCH was built for. The conditional variance recursion
#
#     sigma2_{t+1} = omega + alpha * a_t^2 + beta * sigma2_t
#
# exploits volatility clustering: a big move today raises the forecast for
# tomorrow's volatility.
#
# Notebook 01 *also* showed that on NIFTY, the Ljung-Box p-value on squared
# returns was ~1.0 — meaning even **non-linear** autocorrelation looks weak on
# the index. So we run two experiments here:
#
# 1. **NIFTY 15m** — the broad index, where vol clustering is averaged across 50 stocks
# 2. **RELIANCE 15m** — a single liquid stock, where idiosyncratic vol clustering survives
#
# If GARCH adds nothing on either, we have a publishable finding. If it helps
# on RELIANCE but not NIFTY, that's also publishable — aggregation washes out
# the GARCH signal.

# %%
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path.cwd()
if (ROOT / "src").is_dir():
    sys.path.insert(0, str(ROOT))
elif (ROOT.parent / "src").is_dir():
    sys.path.insert(0, str(ROOT.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import add_returns, load_bars
from src.vol_models import (
    ConstantVolForecaster,
    EWMAVolForecaster,
    GARCHForecaster,
    score_vol,
    walk_forward_vol,
)

pd.set_option("display.width", 130)
pd.set_option("display.precision", 6)
plt.rcParams["figure.figsize"] = (11, 4)


# %% [markdown]
# ## 1. Helper to run the full vol experiment on any series

# %%
def run_vol_experiment(symbol: str, label: str, interval: str = "15m", period: str = "60d") -> dict:
    df = add_returns(load_bars(symbol, interval=interval, period=period))
    y = df["logret"]
    split = int(len(y) * 0.8)
    print(f"\n=== {label} ({symbol}, {interval}) ===")
    print(f"  bars: {len(y):,}  |  train {split:,}  |  test {len(y) - split:,}")

    models = [
        ConstantVolForecaster(),
        EWMAVolForecaster(lam=0.97),
        GARCHForecaster(asymmetric=False, dist="t"),
        GARCHForecaster(asymmetric=True, dist="t"),  # GJR
    ]

    rows: dict[str, pd.DataFrame] = {}
    scores: dict[str, dict] = {}
    for m in models:
        res = walk_forward_vol(y, m, train_size=split)
        rows[m.name] = res
        scores[m.name] = score_vol(res)

    return {"label": label, "y": y, "split": split, "results": rows, "scores": scores}


# %% [markdown]
# ## 2. Experiment 1 — NIFTY 50

# %%
nifty = run_vol_experiment("NIFTY", "NIFTY 50")
nifty_table = pd.DataFrame(nifty["scores"]).T
print(nifty_table)
nifty_table

# %% [markdown]
# ## 3. Experiment 2 — RELIANCE (single liquid stock)

# %%
reliance = run_vol_experiment("RELIANCE", "RELIANCE Industries")
rel_table = pd.DataFrame(reliance["scores"]).T
print(rel_table)
rel_table

# %% [markdown]
# ## 4. Visualize: forecast vol vs realized |returns|
#
# A picture is worth ten metrics. We overlay each model's forecast standard
# deviation against the realized |return|. A good vol forecast hugs the
# *upper envelope* of |r| — it correctly anticipates the noisy spikes.

# %%
def plot_vol_overlay(exp: dict, models_to_show: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    any_result = next(iter(exp["results"].values()))
    realized = np.sqrt(any_result["actual_sq"])
    ax.plot(realized.index, realized.values, color="lightgray", lw=0.5, label="|realized return|")

    palette = {"constant_var": "black", "ewma_vol": "tab:blue",
               "garch_t": "tab:orange", "gjr_garch_t": "tab:red"}
    for m in models_to_show:
        v = exp["results"][m]["forecast_var"]
        ax.plot(v.index, np.sqrt(v.values), lw=1.0, alpha=0.85,
                color=palette.get(m, None), label=m)
    ax.set_title(f"{exp['label']}: forecast vol vs realized |return| (15m bars)")
    ax.set_ylabel("std dev / |return|")
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()


plot_vol_overlay(nifty, ["constant_var", "ewma_vol", "garch_t", "gjr_garch_t"])
plot_vol_overlay(reliance, ["constant_var", "ewma_vol", "garch_t", "gjr_garch_t"])

# %% [markdown]
# ## 5. Head-to-head: QLIKE delta vs the constant-vol baseline
#
# Positive delta = baseline beat the model. Negative = model beat baseline.
# QLIKE is the right loss for vol comparison (Patton 2011); MSE on r² is
# dominated by squared-return outliers and very noisy.

# %%
def qlike_deltas(exp: dict) -> pd.DataFrame:
    base = exp["scores"]["constant_var"]["qlike"]
    rows = []
    for name, sc in exp["scores"].items():
        rows.append({"model": name, "qlike": sc["qlike"], "delta_vs_const": sc["qlike"] - base})
    return pd.DataFrame(rows).set_index("model").sort_values("qlike")


print("\nNIFTY 50:")
print(qlike_deltas(nifty))
print("\nRELIANCE:")
print(qlike_deltas(reliance))

# %% [markdown]
# ## 6. Interpretation guide
#
# - If **GARCH/GJR** has clearly lower QLIKE than constant vol on RELIANCE
#   but not NIFTY, the story writes itself: **idiosyncratic vol clustering is
#   real on individual stocks but is washed out by cross-sectional
#   aggregation in the index.** This is a meaningful empirical finding and
#   the kind of result that justifies the whole "deep dive" framing.
# - If EWMA roughly matches GARCH, you've discovered why RiskMetrics shipped
#   in 1996 with just lambda=0.94 — the marginal value of full GARCH
#   parameter estimation over a well-tuned EWMA is often small.
# - If **constant vol** beats everything on NIFTY 60-day window, the
#   conclusion is honest: this sample doesn't have enough vol regime variation
#   to make adaptive models pay. Re-run on a longer (e.g., 1-year) daily
#   bar sample to expose the regimes (vol of vol is much larger in daily
#   data spanning multiple events).
#
# This notebook is the end of the "ARIMA-family on prices/returns" arc. The
# next natural moves are:
#
# - **04**: SARIMA / intraday-deseasonalized returns — back to direction
#   forecasting, but accounting for the U-shape. Does removing seasonality
#   reveal a residual signal that plain ARIMA missed?
# - **05**: ARIMAX with India VIX and USDINR as exogenous regressors.
# - **06**: Strategy backtest — translate the volatility forecast into a
#   simple position-sizing rule, score on realistic NSE costs (STT, slippage),
#   and ask: did *any* of this earn a paisa?
