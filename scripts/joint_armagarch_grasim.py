"""Paper-trade joint AR(1)-GJR-GARCH(1,1)-t on GRASIM and compare to the
separate-models baseline from the previous script.

Same data window. Same cost model (single-stock intraday MIS = 42 bps r/t).
Same vol target. Only the model differs:

    SEPARATE  =  ARIMA  for direction  +  GJR-GARCH  for variance,
                 each model fitted on its own likelihood
    JOINT     =  one AR(1)-GJR-GARCH(1,1)-t fitted by joint MLE,
                 emitting BOTH the direction signal and the variance forecast

Three strategies on the joint model, three on the separate-models baseline,
plus buy-and-hold. Print everything side-by-side.
"""

from __future__ import annotations

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
from src.models import ARIMAForecaster
from src.paper_trade import (
    PaperTrader,
    make_combined_rule,
    make_voltarget_long_rule,
    rule_arima_directional,
)
from src.vol_models import GARCHForecaster, JointARGARCHForecaster

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)
pd.set_option("display.precision", 6)

TICKER = "GRASIM.NS"
PAPER_DAYS = 12
TARGET_VOL = 0.0040
PPY = 6300.0


# ---------------------------------------------------------------------------
# Data + split (use cached parquet from previous run for identical comparison)
# ---------------------------------------------------------------------------
df = add_returns(load_bars(TICKER, interval="15m", period="60d", use_cache=True))
y = df["logret"]
close = df["close"]
days = pd.Index(sorted(y.index.normalize().unique()))
cutoff = days[-PAPER_DAYS]
y_train = y[y.index < cutoff]
y_paper = y[y.index >= cutoff]
close_paper = close[y.index >= cutoff]
print(f"=== {TICKER} ===")
print(f"Train: {y_train.index.min()} -> {y_train.index.max()}  ({len(y_train):,} bars)")
print(f"Paper: {y_paper.index.min()} -> {y_paper.index.max()}  ({len(y_paper):,} bars, "
      f"{y_paper.index.normalize().nunique()} days)")


# ---------------------------------------------------------------------------
# Fit separate-models baseline (same as previous run)
# ---------------------------------------------------------------------------
best, _ = grid_search(y_train, max_p=3, max_q=3, d=0, trend="n")
arima_sep = ARIMAForecaster(order=best.order, trend="n"); arima_sep.fit(y_train)
gjr_sep = GARCHForecaster(asymmetric=True, dist="t", refit_every=None); gjr_sep.fit(y_train)
print(f"\n[SEPARATE] ARIMA picked: {best}")
print(f"[SEPARATE] GJR-GARCH params:  mu={gjr_sep._mu:.4f}  omega={gjr_sep._omega:.5f}  "
      f"alpha={gjr_sep._alpha:.4f}  gamma={gjr_sep._gamma:.4f}  beta={gjr_sep._beta:.4f}")


# ---------------------------------------------------------------------------
# Fit JOINT AR(1)-GJR-GARCH-t model
# ---------------------------------------------------------------------------
joint = JointARGARCHForecaster(asymmetric=True, dist="t", refit_every=None)
joint.fit(y_train)
print(f"\n[JOINT] AR(1)-GJR-GARCH-t params:")
print(f"   Mean eq:   mu={joint._mu / joint.rescale:+.6e}   phi(AR1)={joint._phi:+.4f}")
print(f"   Var eq:    omega={joint._omega:.5f}  alpha={joint._alpha:.4f}  "
      f"gamma={joint._gamma:.4f}  beta={joint._beta:.4f}")
print(f"   Tail dof:  nu={joint._nu:.2f}   (lower = fatter tails; <4 means infinite kurtosis)")


# ---------------------------------------------------------------------------
# Paper-trade loop: six strategies + buy-and-hold
# ---------------------------------------------------------------------------
cost = NSECostModel.intraday_equity()

traders = {
    "SEP_arima_directional":   PaperTrader(rule_arima_directional, cost, name="SEP_arima_directional"),
    "SEP_gjr_voltarget_long":  PaperTrader(make_voltarget_long_rule(TARGET_VOL), cost, name="SEP_gjr_voltarget_long"),
    "SEP_combined":            PaperTrader(make_combined_rule(TARGET_VOL), cost, name="SEP_combined"),
    "JOINT_arima_directional": PaperTrader(rule_arima_directional, cost, name="JOINT_arima_directional"),
    "JOINT_gjr_voltarget_long":PaperTrader(make_voltarget_long_rule(TARGET_VOL), cost, name="JOINT_gjr_voltarget_long"),
    "JOINT_combined":          PaperTrader(make_combined_rule(TARGET_VOL), cost, name="JOINT_combined"),
}

# Seed forecasts (made AT the last training bar, FOR the first paper bar)
prev_r_sep = arima_sep.forecast_next()
prev_v_sep = gjr_sep.forecast_next_var()
prev_r_joint = joint.forecast_next_ret()
prev_v_joint = joint.forecast_next_var()

for ts, c, r in zip(y_paper.index, close_paper.to_numpy(), y_paper.to_numpy()):
    # Update both model families with the realized return
    arima_sep.update(float(r))
    gjr_sep.update(float(r))
    joint.update(float(r))

    # Generate next-bar forecasts from each
    nr_sep = arima_sep.forecast_next();      nv_sep = gjr_sep.forecast_next_var()
    nr_joint = joint.forecast_next_ret();    nv_joint = joint.forecast_next_var()

    # Drive each trader with the appropriate forecast pair
    for name, t in traders.items():
        is_joint = name.startswith("JOINT_")
        pr = prev_r_joint if is_joint else prev_r_sep
        pv = prev_v_joint if is_joint else prev_v_sep
        nr = nr_joint if is_joint else nr_sep
        nv = nv_joint if is_joint else nv_sep
        t.observe_bar(
            ts=ts, close=float(c), realized_logret=float(r),
            forecast_ret_for_this_bar=pr, forecast_var_for_this_bar=pv,
            forecast_ret_for_next_bar=nr, forecast_var_for_next_bar=nv,
        )
    prev_r_sep, prev_v_sep = nr_sep, nv_sep
    prev_r_joint, prev_v_joint = nr_joint, nv_joint


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
bh_eq = float((1 + y_paper).cumprod().iloc[-1])
bh_ret = bh_eq - 1.0
bh_sd = float(y_paper.std(ddof=1))
bh_sharpe = float(y_paper.mean()) / bh_sd * np.sqrt(PPY) if bh_sd > 0 else float("nan")
bh_dd = float(((1 + y_paper).cumprod().to_numpy()
               / np.maximum.accumulate((1 + y_paper).cumprod().to_numpy()) - 1).min())
print(f"\n=== Buy-and-hold ===  total={bh_ret*100:+.2f}%  sharpe={bh_sharpe:.2f}  max_dd={bh_dd*100:.2f}%")

summary = pd.DataFrame([t.summary(PPY) for t in traders.values()]).set_index("name")
cols = ["n_position_changes", "win_rate", "total_return", "ann_vol",
        "sharpe", "sortino", "max_drawdown", "total_cost", "final_equity"]
print(f"\n=== Strategy results ===")
print(summary[cols].to_string())

# Spread on the strategies that differ most between SEP and JOINT
print(f"\n=== JOINT minus SEP, per strategy ===")
for kind in ["arima_directional", "gjr_voltarget_long", "combined"]:
    j = summary.loc[f"JOINT_{kind}"]
    s = summary.loc[f"SEP_{kind}"]
    delta_ret = (j["total_return"] - s["total_return"]) * 100
    delta_sharpe = j["sharpe"] - s["sharpe"]
    print(f"  {kind:22s}  Δret={delta_ret:+.3f}pp   Δsharpe={delta_sharpe:+.3f}")

# Per-bar log for today (the truly fresh out-of-sample slice), JOINT only
last_day = traders["JOINT_arima_directional"].blotter_df().index.normalize().max()
print(f"\n=== JOINT model — per-bar log on {last_day.date()} ===")
for name in ["JOINT_arima_directional", "JOINT_gjr_voltarget_long", "JOINT_combined"]:
    bl = traders[name].blotter_df()
    today = bl[bl.index.normalize() == last_day]
    if len(today) == 0:
        continue
    print(f"\n--- {name} ---")
    print(today[["close", "realized_logret", "forecast_ret", "target_position",
                 "delta_pos", "cost", "gross_pnl", "net_pnl", "equity"]].to_string())
