"""Multi-stock sweep with ARMAX-GJR-GARCH-t (own-lag + India VIX + NIFTY +
time-of-day + Parkinson range).

Same 15 stocks, same paper window, same cost model as multistock_sweep.py.
Only the forecasting model changes: exogenous regressors enter the mean
equation. We score:

  - directional_accuracy   : sign(mean forecast) == sign(realised return)
  - vol-target P&L         : same rule as before, using GARCH variance
  - combined P&L           : ARMAX sign x GARCH-scaled size
  - buy-and-hold benchmark

The interesting comparison is per-stock: does adding broad-market + fear
state improve directional hit rate above 50% in a way that survives costs?
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

from src.backtest import NSECostModel
from src.data import add_returns, load_bars
from src.features import build_features
from src.paper_trade import (
    PaperTrader,
    make_combined_rule,
    make_voltarget_long_rule,
    rule_arima_directional,
)
from src.vol_models import JointARMAXGARCHForecaster

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)
pd.set_option("display.precision", 4)

SAMPLE_N = 15
PAPER_DAYS = 12
TARGET_VOL = 0.0040
PPY = 6300.0
SEED = 20260518

LIQUID_STOCKS = [
    "INFY.NS", "ITC.NS", "ICICIBANK.NS", "SBIN.NS", "LT.NS",
    "BAJFINANCE.NS", "AXISBANK.NS", "KOTAKBANK.NS", "HINDUNILVR.NS",
    "BHARTIARTL.NS", "MARUTI.NS", "ASIANPAINT.NS", "WIPRO.NS",
    "SUNPHARMA.NS", "TATAMOTORS.NS", "ADANIENT.NS", "TITAN.NS",
    "HCLTECH.NS", "ULTRACEMCO.NS", "POWERGRID.NS", "NTPC.NS",
    "COALINDIA.NS", "JSWSTEEL.NS", "TATASTEEL.NS", "GRASIM.NS",
    "BAJAJFINSV.NS", "HEROMOTOCO.NS", "EICHERMOT.NS",
]

random.seed(SEED)
SAMPLE = random.sample(LIQUID_STOCKS, k=SAMPLE_N)
print(f"=== ARMAX-GJR-GARCH-t sweep ({SAMPLE_N} stocks) ===")
print(f"Exogenous regressors: nifty_lag, vix_lag, vix_chg_lag, park_var_lag, tod_sin, tod_cos")
print(f"Sample: {SAMPLE}\n")

# Pre-fetch NIFTY + VIX once to populate cache
_ = load_bars("NIFTY", interval="15m", period="60d", use_cache=False)
_ = load_bars("INDIAVIX", interval="15m", period="60d", use_cache=False)


def run_one(ticker: str):
    try:
        ohlc = load_bars(ticker, interval="15m", period="60d", use_cache=True)
    except Exception as exc:
        return None, f"data: {exc}"

    feats = build_features(ohlc, interval="15m", period="60d")
    df = add_returns(ohlc)
    common = df.index.intersection(feats.index)
    if len(common) < 500:
        return None, f"alignment thin: {len(common)} common bars"

    y = df.loc[common, "logret"]
    X = feats.loc[common]
    close = df.loc[common, "close"]

    days = pd.Index(sorted(y.index.normalize().unique()))
    if len(days) < PAPER_DAYS + 5:
        return None, f"only {len(days)} days"
    cutoff = days[-PAPER_DAYS]
    train_mask = y.index < cutoff
    paper_mask = y.index >= cutoff

    y_train, y_paper = y[train_mask], y[paper_mask]
    X_train, X_paper = X[train_mask], X[paper_mask]
    close_paper = close[paper_mask]

    if len(y_train) < 300 or len(y_paper) < 30:
        return None, f"thin: train={len(y_train)} paper={len(y_paper)}"

    try:
        model = JointARMAXGARCHForecaster(asymmetric=True, dist="t")
        model.fit(y_train, X_train)
    except Exception as exc:
        return None, f"fit: {exc}"

    cost = NSECostModel.intraday_equity()
    traders = {
        "armax_directional":   PaperTrader(rule_arima_directional, cost, name="armax_directional"),
        "garch_voltarget":     PaperTrader(make_voltarget_long_rule(TARGET_VOL), cost, name="garch_voltarget"),
        "combined":            PaperTrader(make_combined_rule(TARGET_VOL), cost, name="combined"),
    }

    # Seed: the very first paper bar's forecast uses the last training observation
    # for the AR lag, and X_paper.iloc[0] for the exogenous row.
    model.set_next_x(X_paper.iloc[0])
    prev_r = model.forecast_next_ret()
    prev_v = model.forecast_next_var()

    # Track directional accuracy alongside the strategy backtests
    correct = 0
    counted = 0

    for i, (ts, c, r) in enumerate(zip(y_paper.index, close_paper.to_numpy(), y_paper.to_numpy())):
        # The forecast we made BEFORE this bar (prev_r) is being judged now
        if prev_r != 0:
            counted += 1
            if np.sign(prev_r) == np.sign(r):
                correct += 1

        # Update model with the just-realised return and the X row that DROVE that forecast
        x_for_this_bar = X_paper.iloc[i].to_numpy()
        model.update(float(r), x_for_this_bar)

        # Set up next bar's X regressors (if any), then forecast
        if i + 1 < len(X_paper):
            model.set_next_x(X_paper.iloc[i + 1])
            nr = model.forecast_next_ret()
            nv = model.forecast_next_var()
        else:
            nr, nv = 0.0, model.forecast_next_var()

        for t in traders.values():
            t.observe_bar(
                ts=ts, close=float(c), realized_logret=float(r),
                forecast_ret_for_this_bar=prev_r, forecast_var_for_this_bar=prev_v,
                forecast_ret_for_next_bar=nr, forecast_var_for_next_bar=nv,
            )
        prev_r, prev_v = nr, nv

    # Buy-and-hold
    bh_eq = (1 + y_paper).cumprod()
    bh_total = float(bh_eq.iloc[-1] - 1.0)
    bh_sd = float(y_paper.std(ddof=1))
    bh_sharpe = float(y_paper.mean()) / bh_sd * np.sqrt(PPY) if bh_sd > 0 else float("nan")
    bh_dd = float((bh_eq.to_numpy() / np.maximum.accumulate(bh_eq.to_numpy()) - 1).min())

    row = {
        "ticker": ticker,
        "phi_AR1": model._phi,
        "nu_t": model._nu,
        "dir_hit_rate": (correct / counted) if counted > 0 else float("nan"),
        "n_directional_calls": counted,
        "bh_total": bh_total,
        "bh_sharpe": bh_sharpe,
        "bh_max_dd": bh_dd,
    }
    # Print fitted exog coefficients for the first few stocks (limit clutter)
    coefs = dict(zip(model._x_cols, model._beta_x))
    for name, t in traders.items():
        s = t.summary(PPY)
        row[f"{name}_total"]  = s["total_return"]
        row[f"{name}_sharpe"] = s["sharpe"]
        row[f"{name}_dd"]     = s["max_drawdown"]
        row[f"{name}_cost"]   = s["total_cost"]
        row[f"{name}_flips"]  = s["n_position_changes"]
    return row, coefs


# Compare to previous sweep results (loaded as constants for clarity)
PRIOR_OWNLAG_VT_TOTAL = {
    "GRASIM.NS": 0.0599, "HINDUNILVR.NS": -0.0706, "BAJAJFINSV.NS": -0.0703,
    "AXISBANK.NS": -0.0554, "LT.NS": -0.0393, "JSWSTEEL.NS": -0.0090,
    "NTPC.NS": -0.0428, "POWERGRID.NS": -0.1247, "MARUTI.NS": -0.0553,
    "ITC.NS": -0.1103, "SBIN.NS": -0.0813, "TITAN.NS": -0.0579,
    "ASIANPAINT.NS": 0.0505, "EICHERMOT.NS": -0.0293, "SUNPHARMA.NS": 0.0198,
}

results = []
all_coefs = {}
for i, ticker in enumerate(SAMPLE, 1):
    print(f"[{i:>2}/{SAMPLE_N}] {ticker:>15s}  ...", end=" ", flush=True)
    out = run_one(ticker)
    if out is None or out[0] is None:
        err = out[1] if out else "?"
        print(f"FAILED ({err})")
        continue
    row, coefs = out
    all_coefs[ticker] = coefs
    results.append(row)
    vt_new = row["garch_voltarget_total"]
    bh = row["bh_total"]
    hit = row["dir_hit_rate"]
    delta_vs_ownlag = (vt_new - PRIOR_OWNLAG_VT_TOTAL.get(ticker, vt_new))
    print(f"hit={hit*100:.1f}%  BH {bh*100:+.2f}%  VT_ARMAX {vt_new*100:+.2f}%  "
          f"Δvs own-lag {delta_vs_ownlag*100:+.2f}pp")

if not results:
    print("No successful runs.")
    sys.exit(1)

df = pd.DataFrame(results).set_index("ticker")

# Per-stock printout
print(f"\n=== Per-stock results, ARMAX-GJR-GARCH-t ===")
show = df[[
    "phi_AR1", "nu_t", "dir_hit_rate",
    "bh_total", "garch_voltarget_total", "armax_directional_total",
    "bh_sharpe", "garch_voltarget_sharpe",
    "garch_voltarget_flips", "garch_voltarget_cost",
    "armax_directional_flips", "armax_directional_cost",
]].copy()
for col in ["bh_total", "garch_voltarget_total", "armax_directional_total",
            "garch_voltarget_cost", "armax_directional_cost"]:
    show[col] = show[col] * 100
show["dir_hit_rate"] = show["dir_hit_rate"] * 100
print(show.round(3).to_string())

# Aggregate vs B&H
vt = df["garch_voltarget_total"]
bh = df["bh_total"]
spread = vt - bh
print(f"\n=== ARMAX-GARCH vol-target vs buy-and-hold ===")
print(f"  Beat B&H:                {(spread > 0).sum()}/{len(df)}  ({(spread > 0).mean()*100:.1f}%)")
print(f"  Mean spread (VT - BH):   {spread.mean()*100:+.3f}pp")
print(f"  Median spread:           {spread.median()*100:+.3f}pp")
print(f"  Best:                    {spread.max()*100:+.3f}pp ({spread.idxmax()})")
print(f"  Worst:                   {spread.min()*100:+.3f}pp ({spread.idxmin()})")

# Aggregate directional accuracy improvement
print(f"\n=== ARMAX directional hit rate ===")
print(f"  Mean hit rate:           {df['dir_hit_rate'].mean()*100:.2f}%   (n=15 stocks)")
print(f"  Stocks > 50%:            {(df['dir_hit_rate'] > 0.5).sum()}/{len(df)}")
print(f"  Stocks > 52%:            {(df['dir_hit_rate'] > 0.52).sum()}/{len(df)}")
print(f"  Best hit rate:           {df['dir_hit_rate'].max()*100:.2f}% ({df['dir_hit_rate'].idxmax()})")

# ARMAX directional P&L
ad = df["armax_directional_total"]
print(f"\n=== ARMAX directional P&L (net of 42bps r/t cost) ===")
print(f"  Profitable:              {(ad > 0).sum()}/{len(df)}")
print(f"  Beat B&H:                {(ad > df['bh_total']).sum()}/{len(df)}")
print(f"  Mean return:             {ad.mean()*100:+.2f}%")
print(f"  Mean cost paid:          {df['armax_directional_cost'].mean()*100:.2f}% of NAV")

# Direct comparison vs own-lag joint model
prior = pd.Series(PRIOR_OWNLAG_VT_TOTAL)
common = df.index.intersection(prior.index)
delta = df.loc[common, "garch_voltarget_total"] - prior.loc[common]
print(f"\n=== ARMAX vs own-lag joint model — per-stock vol-target delta ===")
print(f"  ARMAX better on:         {(delta > 0).sum()}/{len(common)}")
print(f"  Mean improvement:        {delta.mean()*100:+.3f}pp")
print(f"  Best improvement:        {delta.max()*100:+.3f}pp ({delta.idxmax()})")
print(f"  Worst regression:        {delta.min()*100:+.3f}pp ({delta.idxmin()})")

# Show fitted coefficients on a couple of representative stocks
print(f"\n=== Fitted exogenous coefficients (first 3 stocks) ===")
for ticker in list(all_coefs.keys())[:3]:
    print(f"\n  {ticker}:")
    for k, v in all_coefs[ticker].items():
        print(f"     {k:14s}  {v:+.4f}")
