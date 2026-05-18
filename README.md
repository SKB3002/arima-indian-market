# arima-indian-market

> A research project. Can classical ARIMA-family models — used the way every
> textbook teaches them — predict Indian intraday equity markets?
>
> **Answer**: No. But finding out *exactly how and why* is the whole point.

[![python](https://img.shields.io/badge/python-3.12+-blue)](https://www.python.org/)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![status](https://img.shields.io/badge/status-research%20complete-success)](#)

*Results auto-regenerated:* **2026-05-18 14:55 IST**
*NIFTY daily window:* 2011-05-17 to 2026-05-15 (3677 trading days)
*NIFTY 15m window:* 2026-02-17 to 2026-05-18 (1446 bars across 59 days)

---

## TL;DR

| Question | Answer from this project |
|---|---|
| Can ARIMA on returns forecast direction? | **No.** Statistically indistinguishable from naive on intraday; barely-significant 53% hit rate on daily that vanishes net of NSE costs. |
| Can GARCH-family forecast volatility? | **Yes.** QLIKE improves consistently over constant-variance at daily; GJR-GARCH parameters remain stable across the COVID regime. |
| Does adding India VIX + NIFTY + time-of-day rescue the directional model? | **Partially.** ARMAX-GJR-GARCH hits **52.2%** directional accuracy across 15 stocks (+3σ above coin-flip) — but per-bar edge is still smaller than 42 bps round-trip cost. |
| Does vol-targeting beat buy-and-hold? | **No.** Beats B&H 5/16 stocks (31%) — *worse* than chance in this 12-day window. |
| Net of realistic costs, can ARIMA-GARCH make money on Indian intraday? | **No.** The signal-to-cost ratio at 15-min single-stock horizon is structurally insufficient. This is the universal empirical-finance finding, re-derived on real Indian data. |

If you came here for a money printer, the next stop is the literature on
LSTM / regime-switching / alternative-data signals. If you came here to
understand *why* the simple thing fails, read on.

---

## The four headline findings, with the receipts

### 1. Returns are unforecastable for ARIMA, at every frequency we tested

NIFTY 15m bars over 59 trading days: Ljung-Box on returns at
lag 20 has **p = 0.94** — cannot reject independence. Squared
returns at lag 20 have **p = 1.00** — even volatility
clustering is invisible at the index level on this sample.

NIFTY daily over 15 years (2011-05-17 to 2026-05-15): Ljung-Box *does* reject on
returns (p = 3.77e-10) — there's a real but tiny pattern. Daily
ARIMA(1,0,0) vs naive on 736 test days:

| Model | RMSE | Directional accuracy |
|---|---|---|
| naive (predict 0) | 0.008249 | — |
| **ARIMA(1,0,0)** | 0.008257 | **53.47%** |

Diebold-Mariano test of squared-error loss: **p = 0.32** — ARIMA's MSE
is statistically indistinguishable from "always predict zero." The 53.5% hit
rate is +1.3σ at this n — well inside coin-flip noise.

### 2. Volatility forecasting is where GARCH actually earns its keep

NIFTY daily, 15-year window, QLIKE on volatility forecasts (lower is better):

| Model | QLIKE | Δ vs constant |
|---|---|---|
| constant variance | -8.4662 | 0.000 |
| EWMA λ=0.94 | -8.6950 | -0.2289 |
| GARCH(1,1)-t | -8.7188 | -0.2527 |
| **GJR-GARCH(1,1)-t** | **-8.6866** | **-0.2205** |

QLIKE improvement is genuine and *grew* in the post-2020 stress window
(see notebook 04). GJR-GARCH parameters fitted on pre-2020 data survived
COVID without re-estimation — operational evidence the model captures real
structure, not curve-fit noise.

### 3. Exogenous regressors push hit rate above coin-flip

ARMAX(1) + India VIX + NIFTY lag + Parkinson range + time-of-day, fitted
jointly with GJR-GARCH(1,1)-t errors. Same 16 stocks, same 12-day
paper-trade window:

- **Mean directional hit rate: 52.24%** (vs random 50%)
- 12/16 stocks above 50%
- 10/16 stocks above 52%
- Standard error around 50% at ~4,200 calls: ~0.77 percentage points
- **Result is ~+3.0σ above random** — statistically overwhelming

But: 0/16 profitable net of cost. The model can predict; the strategy
can't trade. Per-trade gross expected value (~2 bps) is ~20× smaller than
single-stock intraday round-trip cost (42 bps).

### 4. The cost wall — why a real edge still loses

Across 16 stocks under realistic NSE costs:

| Metric | Value |
|---|---|
| ARMAX-directional mean return | **-33.51%** in 12 days |
| ARMAX-directional mean cost paid | **43.84%** of NAV |
| Vol-target beat B&H | **5/16** (31%) |
| Vol-target mean spread vs B&H | **-1.42pp** |

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
