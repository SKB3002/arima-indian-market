"""Forecasting models + walk-forward backtest harness.

Three modelling primitives live here:

- ``NaiveForecaster``     - next return = 0. The minimum bar any real model
                            must clear.
- ``EWMAForecaster``      - exponentially weighted mean of past returns.
                            Captures "drift" if any exists.
- ``ARIMAForecaster``     - statsmodels ARIMA fitted once on the training set,
                            then state-updated with each new observation via
                            ``ARIMAResults.append``. This is the standard
                            no-look-ahead walk-forward pattern: model parameters
                            are fixed on training data, but the model's internal
                            state absorbs each new observation as it arrives.

The walk-forward loop produces a DataFrame of one-step-ahead forecasts vs
realised returns, which ``score_forecasts`` then turns into a metrics table.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA


class Forecaster(Protocol):
    name: str

    def fit(self, train: pd.Series) -> "Forecaster": ...
    def forecast_next(self) -> float: ...
    def update(self, observed: float) -> None: ...


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


@dataclass
class NaiveForecaster:
    """Always predicts zero. The natural baseline for stationary mean-zero returns."""

    name: str = "naive_zero"

    def fit(self, train: pd.Series) -> "NaiveForecaster":
        return self

    def forecast_next(self) -> float:
        return 0.0

    def update(self, observed: float) -> None:
        return None


@dataclass
class EWMAForecaster:
    """Forecast = exponentially weighted mean of past returns.

    ``halflife`` is in *bars*. For 15m data, halflife=25 is ~one trading day.
    """

    halflife: float = 25.0
    name: str = "ewma"
    _alpha: float = 0.0
    _state: float = 0.0
    _seen: int = 0

    def fit(self, train: pd.Series) -> "EWMAForecaster":
        # alpha = 1 - exp(-ln(2)/halflife)
        self._alpha = 1.0 - np.exp(-np.log(2.0) / self.halflife)
        # Initialise state with the training mean.
        self._state = float(train.dropna().mean())
        self._seen = int(train.dropna().size)
        return self

    def forecast_next(self) -> float:
        return self._state

    def update(self, observed: float) -> None:
        self._state = self._alpha * observed + (1 - self._alpha) * self._state
        self._seen += 1


# ---------------------------------------------------------------------------
# ARIMA
# ---------------------------------------------------------------------------


class ARIMAForecaster:
    """ARIMA fitted once on training data, then state-updated as new bars arrive.

    Parameters
    ----------
    order : (p, d, q)
        Fixed ARIMA order. Use ``auto_arima.grid_search`` to pick this on the
        training window first.
    trend : str
        statsmodels trend spec. ``'n'`` is right for zero-mean returns.
    refit_every : int | None
        If set, fully re-estimates parameters every N bars instead of only
        updating state. Costlier but adapts to regime drift. ``None`` = never
        refit (pure state update; fastest and the textbook walk-forward).
    """

    def __init__(
        self,
        order: tuple[int, int, int] = (1, 0, 1),
        trend: str = "n",
        refit_every: int | None = None,
    ) -> None:
        self.order = order
        self.trend = trend
        self.refit_every = refit_every
        self.name = f"arima{order}"
        self._results = None
        self._bars_since_refit = 0
        self._train_template: pd.Series | None = None

    def fit(self, train: pd.Series) -> "ARIMAForecaster":
        self._train_template = train.dropna().copy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            warnings.simplefilter("ignore", UserWarning)
            self._results = ARIMA(
                self._train_template, order=self.order, trend=self.trend
            ).fit(method_kwargs={"warn_convergence": False})
        self._bars_since_refit = 0
        return self

    def forecast_next(self) -> float:
        if self._results is None:
            raise RuntimeError("Call fit() before forecast_next().")
        return float(self._results.forecast(steps=1).iloc[0])

    def update(self, observed: float) -> None:
        if self._results is None:
            raise RuntimeError("Call fit() before update().")
        # statsmodels needs the new observation as a 1-element Series with the
        # *next* index value. We can't always know the real timestamp here, so
        # we extend with an integer-position-style index — append accepts that
        # if we drop the index from the original. To keep things robust across
        # pandas versions, refit periodically using accumulated observations.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            warnings.simplefilter("ignore", UserWarning)
            self._results = self._results.append([observed], refit=False)
        self._bars_since_refit += 1
        if self.refit_every and self._bars_since_refit >= self.refit_every:
            # Pull the full augmented history back out and re-estimate params.
            full = pd.Series(self._results.model.endog)
            self._results = ARIMA(full, order=self.order, trend=self.trend).fit(
                method_kwargs={"warn_convergence": False}
            )
            self._bars_since_refit = 0


# ---------------------------------------------------------------------------
# Walk-forward harness
# ---------------------------------------------------------------------------


def walk_forward(
    series: pd.Series,
    model: Forecaster,
    train_size: int,
) -> pd.DataFrame:
    """Run a one-step-ahead walk-forward backtest.

    Splits ``series`` at index ``train_size``. For each test point t:
        1. produce a forecast for t (using only info up to t-1),
        2. record (forecast, actual),
        3. feed the actual value into the model's state.

    Returns a DataFrame indexed like ``series.iloc[train_size:]`` with columns
    ``forecast`` and ``actual``.
    """
    s = series.dropna()
    if train_size >= len(s):
        raise ValueError("train_size must be smaller than series length.")

    train = s.iloc[:train_size]
    test = s.iloc[train_size:]
    model.fit(train)

    forecasts = np.empty(len(test))
    actuals = test.to_numpy()
    for i, observed in enumerate(actuals):
        forecasts[i] = model.forecast_next()
        model.update(float(observed))

    return pd.DataFrame(
        {"forecast": forecasts, "actual": actuals},
        index=test.index,
    )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_forecasts(results: pd.DataFrame) -> dict:
    """Compute RMSE, MAE, directional accuracy, and hit-rate stats.

    Directional accuracy treats ``forecast == 0`` as 'no opinion' and excludes
    those bars from the hit-rate denominator (otherwise the naïve-zero baseline
    gets nonsense scoring).
    """
    f = results["forecast"].to_numpy()
    a = results["actual"].to_numpy()
    err = a - f

    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))

    opinion = f != 0.0
    n_opinion = int(opinion.sum())
    if n_opinion == 0:
        dir_acc = float("nan")
    else:
        dir_acc = float(np.mean(np.sign(f[opinion]) == np.sign(a[opinion])))

    return {
        "n": int(len(results)),
        "rmse": rmse,
        "mae": mae,
        "directional_accuracy": dir_acc,
        "n_opinion_bars": n_opinion,
    }


def diebold_mariano(
    results_a: pd.DataFrame,
    results_b: pd.DataFrame,
    loss: str = "se",
) -> dict:
    """Diebold-Mariano test: is model A's loss significantly lower than B's?

    H0: equal predictive accuracy. Two-sided p-value.

    Caveat: classical DM assumes non-nested models. For ARIMA vs naïve (nested),
    treat the p-value as suggestive rather than definitive — but the loss
    differential's sign and size still tell you what you need.
    """
    a = results_a.copy()
    b = results_b.copy()
    common = a.index.intersection(b.index)
    a = a.loc[common]
    b = b.loc[common]

    err_a = (a["actual"] - a["forecast"]).to_numpy()
    err_b = (b["actual"] - b["forecast"]).to_numpy()

    if loss == "se":
        loss_a, loss_b = err_a**2, err_b**2
    elif loss == "ae":
        loss_a, loss_b = np.abs(err_a), np.abs(err_b)
    else:
        raise ValueError("loss must be 'se' or 'ae'")

    d = loss_a - loss_b
    n = d.size
    mean_d = float(np.mean(d))
    # HAC-lite: lag-0 variance only. Sufficient for one-step forecasts.
    var_d = float(np.var(d, ddof=1))
    dm_stat = mean_d / np.sqrt(var_d / n) if var_d > 0 else float("nan")
    # Two-sided p-value from standard normal approximation.
    from scipy.stats import norm

    pvalue = float(2 * (1 - norm.cdf(abs(dm_stat)))) if np.isfinite(dm_stat) else float("nan")
    return {
        "mean_loss_diff_A_minus_B": mean_d,
        "dm_stat": float(dm_stat),
        "pvalue": pvalue,
        "n": int(n),
    }
