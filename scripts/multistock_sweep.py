"""Multi-stock sweep: joint AR(1)-GJR-GARCH-t paper-trade across N random NSE
large-caps. Aggregates whether GJR vol-targeting actually beats buy-and-hold
more often than chance on individual stocks.

For each stock:
    - pull 60d of 15m bars (yfinance free tier)
    - split: last PAPER_DAYS as the paper-trade window, rest as training
    - fit joint AR(1)-GJR-GARCH(1,1)-t on training
    - run paper-trade with three strategies:
        * arima_directional   (sign of mean-equation forecast)
        * gjr_voltarget_long  (sized inversely to forecast vol, always long)
        * combined            (sign x vol-scaled size)
    - score against buy-and-hold (same window)

At the end: aggregate win rate of vol-target vs B&H, return spreads,
Sharpe deltas, total cost burn per strategy.
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
from src.paper_trade import (
    PaperTrader,
    make_combined_rule,
    make_voltarget_long_rule,
    rule_arima_directional,
)
from src.vol_models import JointARGARCHForecaster

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
pd.set_option("display.precision", 4)

# ----- Config -----------------------------------------------------------------
SAMPLE_N = 15           # how many stocks to test
PAPER_DAYS = 12         # paper-trade window length (most recent N trading days)
TARGET_VOL = 0.0040     # per-bar target vol for vol-targeting rules
PPY = 6300.0            # 15m bars per trading year
SEED = 20260518         # deterministic pick for reproducibility this session
# ------------------------------------------------------------------------------

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
sample = random.sample(LIQUID_STOCKS, k=SAMPLE_N)
print(f"=== Multi-stock sweep ===")
print(f"Sampled {SAMPLE_N} of {len(LIQUID_STOCKS)} stocks: {sample}\n")

cost = NSECostModel.intraday_equity()
results: list[dict] = []
failures: list[tuple[str, str]] = []


def run_one(ticker: str) -> dict | None:
    """Pull data, fit joint model, paper-trade, return summary row."""
    try:
        df = add_returns(load_bars(ticker, interval="15m", period="60d", use_cache=False))
    except Exception as exc:
        failures.append((ticker, f"data: {exc}"))
        return None

    y = df["logret"]
    close = df["close"]
    days = pd.Index(sorted(y.index.normalize().unique()))
    if len(days) < PAPER_DAYS + 5:
        failures.append((ticker, f"only {len(days)} days available"))
        return None

    cutoff = days[-PAPER_DAYS]
    y_train = y[y.index < cutoff]
    y_paper = y[y.index >= cutoff]
    close_paper = close[y.index >= cutoff]
    if len(y_train) < 300 or len(y_paper) < 30:
        failures.append((ticker, f"thin: train={len(y_train)} paper={len(y_paper)}"))
        return None

    try:
        joint = JointARGARCHForecaster(asymmetric=True, dist="t", refit_every=None)
        joint.fit(y_train)
    except Exception as exc:
        failures.append((ticker, f"fit: {exc}"))
        return None

    traders = {
        "arima_directional":  PaperTrader(rule_arima_directional, cost, name="arima_directional"),
        "gjr_voltarget_long": PaperTrader(make_voltarget_long_rule(TARGET_VOL), cost, name="gjr_voltarget_long"),
        "combined":           PaperTrader(make_combined_rule(TARGET_VOL), cost, name="combined"),
    }

    prev_r = joint.forecast_next_ret()
    prev_v = joint.forecast_next_var()
    for ts, c, r in zip(y_paper.index, close_paper.to_numpy(), y_paper.to_numpy()):
        joint.update(float(r))
        nr = joint.forecast_next_ret()
        nv = joint.forecast_next_var()
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
        "bars_paper": len(y_paper),
        "phi_AR1": joint._phi,
        "nu_t": joint._nu,
        "bh_total":   bh_total,
        "bh_sharpe":  bh_sharpe,
        "bh_max_dd":  bh_dd,
    }
    for name, t in traders.items():
        s = t.summary(PPY)
        row[f"{name}_total"]  = s["total_return"]
        row[f"{name}_sharpe"] = s["sharpe"]
        row[f"{name}_dd"]     = s["max_drawdown"]
        row[f"{name}_cost"]   = s["total_cost"]
        row[f"{name}_flips"]  = s["n_position_changes"]
    return row


for i, ticker in enumerate(sample, 1):
    print(f"[{i:>2}/{SAMPLE_N}] {ticker:>15s}  ...", end=" ", flush=True)
    row = run_one(ticker)
    if row is None:
        print("FAILED")
        continue
    results.append(row)
    vt = row["gjr_voltarget_long_total"]
    bh = row["bh_total"]
    flag = "✓" if vt > bh else "·"
    print(f"BH {bh*100:+.2f}%  VT {vt*100:+.2f}%  {flag}")

if failures:
    print(f"\nFailures: {failures}")

if not results:
    print("\nNo successful runs.")
    sys.exit(1)

df = pd.DataFrame(results).set_index("ticker")

# ---------------------------------------------------------------------------
# Per-stock table
# ---------------------------------------------------------------------------
print(f"\n=== Per-stock results ({len(df)} stocks) ===")
per_stock = df[[
    "phi_AR1", "nu_t",
    "bh_total", "gjr_voltarget_long_total", "arima_directional_total",
    "bh_sharpe", "gjr_voltarget_long_sharpe",
    "gjr_voltarget_long_flips", "gjr_voltarget_long_cost",
    "arima_directional_flips", "arima_directional_cost",
]].copy()
# Convert returns/costs to %
for col in ["bh_total", "gjr_voltarget_long_total", "arima_directional_total",
            "gjr_voltarget_long_cost", "arima_directional_cost"]:
    per_stock[col] = per_stock[col] * 100
per_stock = per_stock.round(3)
print(per_stock.to_string())

# ---------------------------------------------------------------------------
# Aggregate: vol-target vs B&H
# ---------------------------------------------------------------------------
vt_ret = df["gjr_voltarget_long_total"]
bh_ret = df["bh_total"]
spread = vt_ret - bh_ret
vt_wins = int((spread > 0).sum())
sharpe_spread = df["gjr_voltarget_long_sharpe"] - df["bh_sharpe"]

print(f"\n=== Aggregate: GJR vol-target vs buy-and-hold ===")
print(f"  N stocks:                      {len(df)}")
print(f"  Vol-target beat B&H on return: {vt_wins}/{len(df)} ({vt_wins/len(df)*100:.1f}%)")
print(f"  Mean spread (VT - BH):         {spread.mean()*100:+.3f}pp")
print(f"  Median spread (VT - BH):       {spread.median()*100:+.3f}pp")
print(f"  Best spread:                   {spread.max()*100:+.3f}pp ({spread.idxmax()})")
print(f"  Worst spread:                  {spread.min()*100:+.3f}pp ({spread.idxmin()})")
print(f"  Mean Sharpe spread:            {sharpe_spread.mean():+.3f}")
print(f"  Median Sharpe spread:          {sharpe_spread.median():+.3f}")

# One-sample t-test: is mean spread different from zero?
from scipy import stats
t_stat, p_value = stats.ttest_1samp(spread.to_numpy(), 0.0)
print(f"  t-test mean spread vs 0:       t={t_stat:.3f}  p={p_value:.4f}")

# ---------------------------------------------------------------------------
# Aggregate: ARIMA directional carnage
# ---------------------------------------------------------------------------
ad_ret = df["arima_directional_total"]
ad_cost = df["arima_directional_cost"]
print(f"\n=== Aggregate: ARIMA directional (single-stock costs) ===")
print(f"  Profitable (>0):               {int((ad_ret > 0).sum())}/{len(df)}")
print(f"  Beat B&H:                      {int((ad_ret > df['bh_total']).sum())}/{len(df)}")
print(f"  Mean return:                   {ad_ret.mean()*100:+.2f}%")
print(f"  Median return:                 {ad_ret.median()*100:+.2f}%")
print(f"  Mean cost paid:                {ad_cost.mean()*100:.2f}% of NAV in {PAPER_DAYS} days")
print(f"  Worst case:                    {ad_ret.min()*100:+.2f}%  ({ad_ret.idxmin()})")
