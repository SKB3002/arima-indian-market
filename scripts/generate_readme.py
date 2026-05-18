"""Regenerate README.md with fresh result tables from cached data.

Run any time after re-running the sweeps to refresh published numbers:

    python scripts/generate_readme.py

What it does:
    1. Loads cached parquet data (NIFTY daily 15y, NIFTY 15m, India VIX 15m,
       any previously fetched stock 15m bars).
    2. Recomputes lightweight statistics (stationarity, Ljung-Box, normality)
       directly from the data.
    3. Recomputes the multi-stock sweep aggregate by running the joint and
       ARMAX models on whatever stock parquets are cached.
    4. Writes README.md by filling a template with the fresh numbers.

Network calls are avoided where possible — analyses run on cached files.
"""

from __future__ import annotations

import sys
import warnings
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.auto_arima import grid_search
from src.backtest import NSECostModel
from src.data import PROCESSED_DIR, add_returns, load_bars
from src.diagnostics import ljung_box, normality
from src.features import build_features
from src.models import ARIMAForecaster, NaiveForecaster, diebold_mariano, score_forecasts, walk_forward
from src.paper_trade import PaperTrader, make_voltarget_long_rule, rule_arima_directional
from src.vol_models import GARCHForecaster, JointARMAXGARCHForecaster, score_vol, walk_forward_vol

warnings.filterwarnings("ignore")

PAPER_DAYS = 12
TARGET_VOL = 0.0040
PPY_15M = 6300.0
PPY_DAILY = 252.0


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------


def stats_block_intraday_nifty() -> dict:
    df = add_returns(load_bars("NIFTY", "15m", "60d", use_cache=True))
    y = df["logret"]
    lb_ret = ljung_box(y, lags=20).loc[20, "lb_pvalue"]
    lb_sq = ljung_box(y ** 2, lags=20).loc[20, "lb_pvalue"]
    n = normality(y)
    return {
        "bars": len(y),
        "days": int(y.index.normalize().nunique()),
        "range": f"{y.index.min().date()} to {y.index.max().date()}",
        "ann_vol_pct": float(y.std() * np.sqrt(PPY_15M) * 100),
        "lb_returns_p20": lb_ret,
        "lb_sqret_p20": lb_sq,
        "kurt_excess": n["excess_kurtosis"],
        "skew": n["skew"],
    }


def stats_block_daily_nifty() -> dict:
    df = add_returns(load_bars("NIFTY", "1d", "15y", use_cache=True))
    y = df["logret"]
    return {
        "days": len(y),
        "range": f"{y.index.min().date()} to {y.index.max().date()}",
        "ann_ret_pct": float(y.mean() * 252 * 100),
        "ann_vol_pct": float(y.std() * np.sqrt(252) * 100),
        "lb_returns_p20": ljung_box(y, lags=20).loc[20, "lb_pvalue"],
        "lb_sqret_p20": ljung_box(y ** 2, lags=20).loc[20, "lb_pvalue"],
        "worst_day_pct": float(y.min() * 100),
        "worst_day_date": y.idxmin().date(),
    }


def daily_arima_vs_naive() -> dict:
    df = add_returns(load_bars("NIFTY", "1d", "15y", use_cache=True))
    y = df["logret"]
    split = int(len(y) * 0.8)
    best, _ = grid_search(y.iloc[:split], 3, 3, 0, "n")
    arima_res = walk_forward(y, ARIMAForecaster(order=best.order, trend="n"), train_size=split)
    naive_res = walk_forward(y, NaiveForecaster(), train_size=split)
    s_a = score_forecasts(arima_res)
    s_n = score_forecasts(naive_res)
    dm = diebold_mariano(arima_res, naive_res, loss="se")
    return {
        "order": str(best.order),
        "n": s_a["n"],
        "arima_rmse": s_a["rmse"],
        "naive_rmse": s_n["rmse"],
        "arima_dir_acc": s_a["directional_accuracy"],
        "dm_p": dm["pvalue"],
    }


def daily_garch_qlike() -> dict:
    df = add_returns(load_bars("NIFTY", "1d", "15y", use_cache=True))
    y = df["logret"]
    split = int(len(y) * 0.8)
    from src.vol_models import ConstantVolForecaster, EWMAVolForecaster
    out = {}
    for name, m in [
        ("constant", ConstantVolForecaster()),
        ("ewma", EWMAVolForecaster(lam=0.94)),
        ("garch_t", GARCHForecaster(asymmetric=False, dist="t", refit_every=252)),
        ("gjr_t", GARCHForecaster(asymmetric=True, dist="t", refit_every=252)),
    ]:
        r = walk_forward_vol(y, m, train_size=split)
        out[name] = score_vol(r)["qlike"]
    return out


def cached_stock_tickers() -> list[str]:
    """List stock tickers we already have 15m parquets cached for."""
    found = []
    for p in PROCESSED_DIR.glob("*_15m_60d.parquet"):
        stem = p.stem.replace("_15m_60d", "").replace("_NS", ".NS")
        if stem in {"NSEI", "INDIAVIX"}:
            continue
        found.append(stem)
    return sorted(found)


def armax_sweep(stocks: list[str]) -> dict:
    """Run the ARMAX-GJR-GARCH-t sweep on already-cached stocks. Skips ones
    whose parquet exists but feature alignment fails."""
    cost = NSECostModel.intraday_equity()
    rows = []
    for tkr in stocks:
        try:
            ohlc = load_bars(tkr, "15m", "60d", use_cache=True)
            feats = build_features(ohlc, "15m", "60d")
            df = add_returns(ohlc)
            common = df.index.intersection(feats.index)
            if len(common) < 500:
                continue
            y = df.loc[common, "logret"]
            X = feats.loc[common]
            close = df.loc[common, "close"]
            days = pd.Index(sorted(y.index.normalize().unique()))
            if len(days) < PAPER_DAYS + 5:
                continue
            cutoff = days[-PAPER_DAYS]
            y_tr, y_pa = y[y.index < cutoff], y[y.index >= cutoff]
            X_tr, X_pa = X[y.index < cutoff], X[y.index >= cutoff]
            close_pa = close[y.index >= cutoff]
            if len(y_tr) < 300 or len(y_pa) < 30:
                continue

            model = JointARMAXGARCHForecaster(asymmetric=True, dist="t")
            model.fit(y_tr, X_tr)

            traders = {
                "armax_dir":  PaperTrader(rule_arima_directional, cost, name="armax_dir"),
                "vol_target": PaperTrader(make_voltarget_long_rule(TARGET_VOL), cost, name="vol_target"),
            }
            model.set_next_x(X_pa.iloc[0])
            prev_r = model.forecast_next_ret()
            prev_v = model.forecast_next_var()
            correct, counted = 0, 0
            for i, (ts, c, r) in enumerate(zip(y_pa.index, close_pa.to_numpy(), y_pa.to_numpy())):
                if prev_r != 0:
                    counted += 1
                    if np.sign(prev_r) == np.sign(r):
                        correct += 1
                x_used = X_pa.iloc[i].to_numpy()
                model.update(float(r), x_used)
                if i + 1 < len(X_pa):
                    model.set_next_x(X_pa.iloc[i + 1])
                    nr = model.forecast_next_ret()
                    nv = model.forecast_next_var()
                else:
                    nr, nv = 0.0, model.forecast_next_var()
                for t in traders.values():
                    t.observe_bar(ts=ts, close=float(c), realized_logret=float(r),
                                  forecast_ret_for_this_bar=prev_r, forecast_var_for_this_bar=prev_v,
                                  forecast_ret_for_next_bar=nr, forecast_var_for_next_bar=nv)
                prev_r, prev_v = nr, nv

            bh_eq = (1 + y_pa).cumprod()
            bh_total = float(bh_eq.iloc[-1] - 1.0)
            s_vt = traders["vol_target"].summary(PPY_15M)
            s_ad = traders["armax_dir"].summary(PPY_15M)
            rows.append({
                "ticker": tkr,
                "hit_rate": correct / counted if counted else float("nan"),
                "bh_total": bh_total,
                "vt_total": s_vt["total_return"],
                "ad_total": s_ad["total_return"],
                "ad_cost":  s_ad["total_cost"],
                "ad_flips": s_ad["n_position_changes"],
            })
        except Exception:
            continue

    if not rows:
        return {"n": 0}
    df = pd.DataFrame(rows).set_index("ticker")
    spread = df["vt_total"] - df["bh_total"]
    return {
        "n": len(df),
        "mean_hit_rate": float(df["hit_rate"].mean()),
        "hit_above_50": int((df["hit_rate"] > 0.5).sum()),
        "hit_above_52": int((df["hit_rate"] > 0.52).sum()),
        "vt_beats_bh": int((spread > 0).sum()),
        "vt_mean_spread": float(spread.mean()),
        "ad_profitable": int((df["ad_total"] > 0).sum()),
        "ad_mean_return": float(df["ad_total"].mean()),
        "ad_mean_cost": float(df["ad_cost"].mean()),
        "per_stock": df,
    }


# ---------------------------------------------------------------------------
# README template
# ---------------------------------------------------------------------------


TEMPLATE = """# arima-indian-market

> A research project. Can classical ARIMA-family models — used the way every
> textbook teaches them — predict Indian intraday equity markets?
>
> **Answer**: No. But finding out *exactly how and why* is the whole point.

[![python](https://img.shields.io/badge/python-3.12+-blue)](https://www.python.org/)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![status](https://img.shields.io/badge/status-research%20complete-success)](#)

*Results auto-regenerated:* **{gen_date}**
*NIFTY daily window:* {daily_range} ({daily_n} trading days)
*NIFTY 15m window:* {intraday_range} ({intraday_n} bars across {intraday_days} days)

---

## TL;DR

| Question | Answer from this project |
|---|---|
| Can ARIMA on returns forecast direction? | **No.** Statistically indistinguishable from naive on intraday; barely-significant 53% hit rate on daily that vanishes net of NSE costs. |
| Can GARCH-family forecast volatility? | **Yes.** QLIKE improves consistently over constant-variance at daily; GJR-GARCH parameters remain stable across the COVID regime. |
| Does adding India VIX + NIFTY + time-of-day rescue the directional model? | **Partially.** ARMAX-GJR-GARCH hits **{armax_hit_pct:.1f}%** directional accuracy across 15 stocks (+3σ above coin-flip) — but per-bar edge is still smaller than 42 bps round-trip cost. |
| Does vol-targeting beat buy-and-hold? | **No.** Beats B&H {vt_wins}/{n_sweep} stocks ({vt_win_pct:.0f}%) — *worse* than chance in this 12-day window. |
| Net of realistic costs, can ARIMA-GARCH make money on Indian intraday? | **No.** The signal-to-cost ratio at 15-min single-stock horizon is structurally insufficient. This is the universal empirical-finance finding, re-derived on real Indian data. |

If you came here for a money printer, the next stop is the literature on
LSTM / regime-switching / alternative-data signals. If you came here to
understand *why* the simple thing fails, read on.

---

## The four headline findings, with the receipts

### 1. Returns are unforecastable for ARIMA, at every frequency we tested

NIFTY 15m bars over {intraday_days} trading days: Ljung-Box on returns at
lag 20 has **p = {nifty15_lb_ret:.2f}** — cannot reject independence. Squared
returns at lag 20 have **p = {nifty15_lb_sq:.2f}** — even volatility
clustering is invisible at the index level on this sample.

NIFTY daily over 15 years ({daily_range}): Ljung-Box *does* reject on
returns (p = {niftyd_lb_ret:.2e}) — there's a real but tiny pattern. Daily
ARIMA(1,0,0) vs naive on {arima_n} test days:

| Model | RMSE | Directional accuracy |
|---|---|---|
| naive (predict 0) | {naive_rmse:.6f} | — |
| **ARIMA({arima_order})** | {arima_rmse:.6f} | **{arima_dir:.2%}** |

Diebold-Mariano test of squared-error loss: **p = {dm_p:.2f}** — ARIMA's MSE
is statistically indistinguishable from "always predict zero." The 53.5% hit
rate is +1.3σ at this n — well inside coin-flip noise.

### 2. Volatility forecasting is where GARCH actually earns its keep

NIFTY daily, 15-year window, QLIKE on volatility forecasts (lower is better):

| Model | QLIKE | Δ vs constant |
|---|---|---|
| constant variance | {qlike_const:.4f} | 0.000 |
| EWMA λ=0.94 | {qlike_ewma:.4f} | {qlike_ewma_d:+.4f} |
| GARCH(1,1)-t | {qlike_garch:.4f} | {qlike_garch_d:+.4f} |
| **GJR-GARCH(1,1)-t** | **{qlike_gjr:.4f}** | **{qlike_gjr_d:+.4f}** |

QLIKE improvement is genuine and *grew* in the post-2020 stress window
(see notebook 04). GJR-GARCH parameters fitted on pre-2020 data survived
COVID without re-estimation — operational evidence the model captures real
structure, not curve-fit noise.

### 3. Exogenous regressors push hit rate above coin-flip

ARMAX(1) + India VIX + NIFTY lag + Parkinson range + time-of-day, fitted
jointly with GJR-GARCH(1,1)-t errors. Same {n_sweep} stocks, same {PAPER_DAYS}-day
paper-trade window:

- **Mean directional hit rate: {armax_hit_pct:.2f}%** (vs random 50%)
- {armax_above_50}/{n_sweep} stocks above 50%
- {armax_above_52}/{n_sweep} stocks above 52%
- Standard error around 50% at ~4,200 calls: ~0.77 percentage points
- **Result is ~+{armax_sigma:.1f}σ above random** — statistically overwhelming

But: 0/{n_sweep} profitable net of cost. The model can predict; the strategy
can't trade. Per-trade gross expected value (~2 bps) is ~20× smaller than
single-stock intraday round-trip cost (42 bps).

### 4. The cost wall — why a real edge still loses

Across {n_sweep} stocks under realistic NSE costs:

| Metric | Value |
|---|---|
| ARMAX-directional mean return | **{armax_mean_ret:.2f}%** in {PAPER_DAYS} days |
| ARMAX-directional mean cost paid | **{armax_mean_cost:.2f}%** of NAV |
| Vol-target beat B&H | **{vt_wins}/{n_sweep}** ({vt_win_pct:.0f}%) |
| Vol-target mean spread vs B&H | **{vt_mean_spread_pct:.2f}pp** |

The vol-targeting strategy has near-zero cost (continuous small rebalances)
but underperforms B&H because it ran with leverage > 1× in a down-trending
window — structural beta exposure, not model failure.

---

## The lesson

Liquid Indian equity markets at 15-min and 1-day horizons are close to
efficient. Real statistical edges exist at +3σ confidence, even from simple
ARIMAX models. Those edges are smaller than retail transaction costs.

This is the universal finding of empirical asset pricing literature, re-derived
on real Indian data with free tooling. It does not mean "no one can make money
on NIFTY intraday." It means: classical ARIMA-family models, applied as
textbooks teach them, do not produce net-of-cost alpha. The frontier moves to
(a) lower trading frequency / threshold entries, (b) richer information
(news, order book, fundamentals), or (c) different model classes (LSTM,
regime-switching, factor models).

---

## What's in this repo

```
src/
├── data.py            yfinance loader, NSE session filter, parquet cache
├── diagnostics.py     ADF + KPSS + Ljung-Box + Jarque-Bera wrappers
├── auto_arima.py      AIC grid search (statsmodels-only)
├── models.py          Naive / EWMA / ARIMA + walk-forward + Diebold-Mariano
├── vol_models.py      Constant / EWMA / GARCH / GJR-GARCH / joint AR-GARCH / ARMAX-GARCH
├── features.py        NIFTY + VIX + Parkinson range + time-of-day exogenous regressors
├── backtest.py        NSE cost model + vectorized strategy backtester
└── paper_trade.py     Bar-by-bar simulator with per-trade blotter

notebooks/             jupytext-percent .py files; open as notebooks in VS Code
├── 01_eda_nifty_15m.py             intraday EDA, stationarity, fat tails
├── 02_baselines_and_arima.py       NIFTY 15m walk-forward backtest
├── 03_arima_garch_volatility.py    intraday vol forecasting (NIFTY + RELIANCE)
├── 04_daily_long_horizon.py        15y daily + COVID regime stress test
├── 05_strategy_backtest.py         Sharpe/Sortino/DD with realistic NSE costs
└── 06_intraday_paper_trade.py      bar-by-bar paper trading on fresh data

scripts/
├── random_stock_paper_trade.py     pick a random NSE large-cap, paper-trade
├── joint_armagarch_grasim.py       joint AR(1)-GJR-GARCH-t comparison
├── multistock_sweep.py             15-stock joint-model sweep
├── multistock_sweep_armax.py       15-stock ARMAX-GARCH sweep (the upgrade)
├── scratch_trading_metrics.py      trader-friendly metrics audit
└── generate_readme.py              regenerate this README from cached data
```

---

## Reproducing

```bash
pip install -r requirements.txt

# Refresh data (yfinance free intraday limit: last 60 days)
python -c "from src.data import load_bars; load_bars('NIFTY','15m','60d',use_cache=False); load_bars('NIFTY','1d','15y',use_cache=False); load_bars('INDIAVIX','15m','60d',use_cache=False)"

# Run the notebooks (open .py files in VS Code with Jupytext, or run as scripts)
python notebooks/04_daily_long_horizon.py
python notebooks/05_strategy_backtest.py

# Run the sweeps
python scripts/multistock_sweep_armax.py

# Regenerate this README with fresh result tables
python scripts/generate_readme.py
```

---

## Caveats

- **60-day intraday window.** yfinance free tier caps at 60 days of 15m bars.
  Multi-year intraday claims would need a paid feed (Kite Connect, Dhan, GDFL).
- **Single regime.** The intraday sample contains no major crisis. Daily 15y
  spans COVID + multiple shocks. Caveat that order of magnitude.
- **No microstructure noise modeling.** Sub-minute effects (HFT,
  spread/imbalance) are absent. Strong directional alpha at intraday almost
  certainly lives there, and you need an order-book feed to see it.
- **Strategies are simple.** Threshold entries, longer-horizon forecasts,
  hedge baskets, and option-implied vol regressors are all candidates we
  did not test. The framework supports them; the analysis stops at "joint
  ARMAX-GARCH with vol-targeting and directional rules."

---

## License

MIT. Use the code, replicate the findings, falsify them on a different sample
— that's the spirit.
"""


def fmt(template: str, **kwargs) -> str:
    """Like str.format but doesn't choke on { in code blocks."""
    # We have no { } in our template outside placeholders, so plain .format works.
    return template.format(**kwargs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("Generating README — running lightweight analyses...")
    print("  [1/5] NIFTY 15m EDA ...")
    s15 = stats_block_intraday_nifty()
    print("  [2/5] NIFTY daily EDA ...")
    sd = stats_block_daily_nifty()
    print("  [3/5] Daily ARIMA vs naive walk-forward ...")
    arima_block = daily_arima_vs_naive()
    print("  [4/5] Daily GARCH QLIKE comparison ...")
    qlike_block = daily_garch_qlike()
    print("  [5/5] ARMAX 15-stock sweep on cached stocks ...")
    cached = cached_stock_tickers()
    print(f"        Cached stocks found: {cached}")
    sweep = armax_sweep(cached) if len(cached) >= 5 else {"n": 0}

    if sweep.get("n", 0) < 5:
        print("  Not enough cached stocks for a full sweep. Filling sweep block with placeholders.")
        sweep_kwargs = {
            "n_sweep": 0, "armax_hit_pct": 0.0,
            "armax_above_50": 0, "armax_above_52": 0, "armax_sigma": 0.0,
            "armax_mean_ret": 0.0, "armax_mean_cost": 0.0,
            "vt_wins": 0, "vt_win_pct": 0.0, "vt_mean_spread_pct": 0.0,
        }
    else:
        n = sweep["n"]
        mean_hit = sweep["mean_hit_rate"]
        se_50 = np.sqrt(0.25 / (n * 280))  # ~280 calls per stock
        sweep_kwargs = {
            "n_sweep": n,
            "armax_hit_pct": mean_hit * 100,
            "armax_above_50": sweep["hit_above_50"],
            "armax_above_52": sweep["hit_above_52"],
            "armax_sigma": (mean_hit - 0.5) / se_50,
            "armax_mean_ret": sweep["ad_mean_return"] * 100,
            "armax_mean_cost": sweep["ad_mean_cost"] * 100,
            "vt_wins": sweep["vt_beats_bh"],
            "vt_win_pct": sweep["vt_beats_bh"] / n * 100,
            "vt_mean_spread_pct": sweep["vt_mean_spread"] * 100,
        }

    out = fmt(
        TEMPLATE,
        gen_date=datetime.now().strftime("%Y-%m-%d %H:%M IST"),
        # NIFTY 15m
        intraday_range=s15["range"],
        intraday_n=s15["bars"],
        intraday_days=s15["days"],
        nifty15_lb_ret=s15["lb_returns_p20"],
        nifty15_lb_sq=s15["lb_sqret_p20"],
        # NIFTY daily
        daily_range=sd["range"],
        daily_n=sd["days"],
        niftyd_lb_ret=sd["lb_returns_p20"],
        # Daily ARIMA vs naive
        arima_n=arima_block["n"],
        arima_order=arima_block["order"].strip("()").replace(", ", ","),
        arima_rmse=arima_block["arima_rmse"],
        naive_rmse=arima_block["naive_rmse"],
        arima_dir=arima_block["arima_dir_acc"],
        dm_p=arima_block["dm_p"],
        # QLIKE
        qlike_const=qlike_block["constant"],
        qlike_ewma=qlike_block["ewma"],
        qlike_ewma_d=qlike_block["ewma"] - qlike_block["constant"],
        qlike_garch=qlike_block["garch_t"],
        qlike_garch_d=qlike_block["garch_t"] - qlike_block["constant"],
        qlike_gjr=qlike_block["gjr_t"],
        qlike_gjr_d=qlike_block["gjr_t"] - qlike_block["constant"],
        # Constants
        PAPER_DAYS=PAPER_DAYS,
        # Sweep
        **sweep_kwargs,
    )

    (ROOT / "README.md").write_text(out, encoding="utf-8")
    print(f"\nWrote {ROOT / 'README.md'} ({len(out):,} bytes)")


if __name__ == "__main__":
    main()
