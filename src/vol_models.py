"""Volatility forecasting: constant, EWMA (RiskMetrics), and GARCH(1,1) / GJR.

Design choice — *manual state recursion for GARCH*. The `arch` package gives
us parameter estimates (omega, alpha, beta, optionally gamma for GJR, plus the
conditional mean mu and a Student-t dof nu). Rather than calling
``arch_model.forecast(...)`` at every walk-forward step (slow + awkward with
appended data), we maintain the conditional-variance recursion ourselves:

    GARCH(1,1):    sigma2_{t+1} = omega + alpha * a_t^2 + beta * sigma2_t
    GJR(1,1,1):    sigma2_{t+1} = omega + (alpha + gamma * 1[a_t<0]) * a_t^2
                                 + beta * sigma2_t

where a_t = r_t - mu is the demeaned return. This gives the same forecasts arch
would, but lets us update state in O(1) per bar with no refit. Parameters are
re-estimated on a configurable cadence (``refit_every``); between refits, only
state is updated.

This is the standard practitioner pattern and is pedagogically clearer than
treating ``arch`` as a black box.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
from arch import arch_model


class VolForecaster(Protocol):
    name: str

    def fit(self, train: pd.Series) -> "VolForecaster": ...
    def forecast_next_var(self) -> float: ...
    def update(self, observed_return: float) -> None: ...


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


@dataclass
class ConstantVolForecaster:
    """Forecast = training-set sample variance, held constant. The simplest baseline."""

    name: str = "constant_var"
    _var: float = 0.0

    def fit(self, train: pd.Series) -> "ConstantVolForecaster":
        self._var = float(train.dropna().var(ddof=1))
        return self

    def forecast_next_var(self) -> float:
        return self._var

    def update(self, observed_return: float) -> None:
        return None


@dataclass
class EWMAVolForecaster:
    """RiskMetrics-style EWMA on squared returns.

    sigma2_{t+1} = lambda * sigma2_t + (1 - lambda) * r_t^2

    JP Morgan's 1996 paper used lambda=0.94 for daily data. At 15m bars, the
    equivalent decay (matching half-life in calendar time) is much closer to 1.
    Default here is 0.97, comparable to ~33-bar half-life (~1.3 trading days).
    """

    lam: float = 0.97
    name: str = "ewma_vol"
    _var: float = 0.0
    _mean: float = 0.0

    def fit(self, train: pd.Series) -> "EWMAVolForecaster":
        s = train.dropna()
        self._mean = float(s.mean())
        self._var = float(((s - self._mean) ** 2).mean())
        return self

    def forecast_next_var(self) -> float:
        return self._var

    def update(self, observed_return: float) -> None:
        a = observed_return - self._mean
        self._var = self.lam * self._var + (1 - self.lam) * a * a


# ---------------------------------------------------------------------------
# GARCH family
# ---------------------------------------------------------------------------


class GARCHForecaster:
    """GARCH(1,1) (default) or GJR-GARCH(1,1,1) with manual state recursion.

    Parameters
    ----------
    asymmetric : bool
        If True, fit GJR-GARCH (a.k.a. TARCH) which adds a gamma term capturing
        the leverage effect (negative shocks raise vol more than positive).
    dist : str
        Innovation distribution passed to ``arch_model``. ``"t"`` (Student-t)
        is the right call when EDA shows fat tails (kurtosis >> 3). Use
        ``"normal"`` only as a stress test.
    refit_every : int | None
        Refit parameters every N bars. ``None`` = fit once on the training set
        and only update state thereafter (fastest, fine for short test windows).
    rescale : float
        Multiplier applied to returns before passing to arch_model and undone
        on the forecast. ``arch`` warns/converges poorly when input magnitudes
        are tiny (intraday log-returns ~1e-3); scaling to ~percent helps. We
        store the factor and undo it consistently.
    """

    def __init__(
        self,
        asymmetric: bool = False,
        dist: str = "t",
        refit_every: int | None = None,
        rescale: float = 100.0,
    ) -> None:
        self.asymmetric = asymmetric
        self.dist = dist
        self.refit_every = refit_every
        self.rescale = rescale
        self.name = ("gjr_garch" if asymmetric else "garch") + f"_{dist}"

        self._mu = 0.0
        self._omega = 0.0
        self._alpha = 0.0
        self._beta = 0.0
        self._gamma = 0.0  # used only when asymmetric
        self._sigma2 = 0.0  # current conditional variance (rescaled units)
        self._bars_since_refit = 0
        self._history: list[float] = []  # rescaled returns, for periodic refits

    def _fit_params(self, returns_rescaled: pd.Series) -> None:
        kw = dict(mean="Constant", vol="GARCH", p=1, q=1, dist=self.dist, rescale=False)
        if self.asymmetric:
            kw["o"] = 1  # GJR / TARCH

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = arch_model(returns_rescaled, **kw).fit(disp="off", show_warning=False)

        p = res.params
        self._mu = float(p.get("mu", 0.0))
        self._omega = float(p["omega"])
        self._alpha = float(p["alpha[1]"])
        self._beta = float(p["beta[1]"])
        self._gamma = float(p["gamma[1]"]) if self.asymmetric else 0.0
        # Start state at the unconditional variance implied by the params.
        if self.asymmetric:
            denom = max(1e-12, 1.0 - self._alpha - self._beta - 0.5 * self._gamma)
        else:
            denom = max(1e-12, 1.0 - self._alpha - self._beta)
        self._sigma2 = self._omega / denom

    def fit(self, train: pd.Series) -> "GARCHForecaster":
        s = train.dropna() * self.rescale
        self._history = s.tolist()
        self._fit_params(pd.Series(self._history))
        self._bars_since_refit = 0
        return self

    def forecast_next_var(self) -> float:
        # Convert back from rescaled units (variance scales by rescale^2).
        return self._sigma2 / (self.rescale**2)

    def update(self, observed_return: float) -> None:
        r = observed_return * self.rescale
        a = r - self._mu
        a2 = a * a
        # Recurse the conditional variance forward one bar.
        if self.asymmetric:
            asym = self._gamma * a2 * (a < 0)
            self._sigma2 = self._omega + self._alpha * a2 + asym + self._beta * self._sigma2
        else:
            self._sigma2 = self._omega + self._alpha * a2 + self._beta * self._sigma2

        self._history.append(r)
        self._bars_since_refit += 1
        if self.refit_every and self._bars_since_refit >= self.refit_every:
            # Re-estimate parameters on the accumulated history. State (sigma2)
            # continues from where it was — we don't reset it; we just refresh
            # the dynamics it follows.
            sigma2_carry = self._sigma2
            self._fit_params(pd.Series(self._history))
            self._sigma2 = sigma2_carry
            self._bars_since_refit = 0


# ---------------------------------------------------------------------------
# Walk-forward + scoring for volatility
# ---------------------------------------------------------------------------


class JointARGARCHForecaster:
    """AR(1)-GJR-GARCH(1,1)-Student-t fitted as one joint model.

    Difference from running ARIMAForecaster + GARCHForecaster separately:

    - Joint MLE: the AR(1) mean coefficient and the GJR-GARCH variance
      parameters are estimated simultaneously on the *correct* likelihood
      (Student-t with the GARCH variance). Separate fits implicitly assume
      OLS-style errors for the mean and homoskedastic-like residuals for the
      variance — both wrong on financial data.

    - The variance equation is driven by the residual ``a_t = r_t - mu -
      phi * r_{t-1}``, not the raw return. With small AR coefficients the
      difference in vol forecast is small; with bigger AR coefficients
      (rare on liquid data) it can matter.

    - One model exposes BOTH a one-step-ahead mean forecast and a one-step-
      ahead variance forecast — the right primitive for a "direction + size"
      strategy that wants its size to react to the same shocks its direction
      saw.

    Mean equation:  r_t = mu + phi * r_{t-1} + a_t,  a_t = sigma_t * z_t
    Var equation:   sigma2_{t+1} = omega + alpha * a_t^2
                                 + gamma * a_t^2 * 1[a_t < 0]
                                 + beta  * sigma2_t
    Innovations:    z_t ~ Student-t(nu)

    We follow the same design pattern as GARCHForecaster: fit once with the
    `arch` package, extract parameters, then recurse mean and variance state
    manually for cheap walk-forward updates.
    """

    def __init__(
        self,
        asymmetric: bool = True,
        dist: str = "t",
        refit_every: int | None = None,
        rescale: float = 100.0,
    ) -> None:
        self.asymmetric = asymmetric
        self.dist = dist
        self.refit_every = refit_every
        self.rescale = rescale
        self.name = ("joint_ar1_gjr_garch" if asymmetric else "joint_ar1_garch") + f"_{dist}"

        self._mu = 0.0
        self._phi = 0.0  # AR(1) coefficient
        self._omega = 0.0
        self._alpha = 0.0
        self._beta = 0.0
        self._gamma = 0.0
        self._nu = float("nan")
        self._sigma2 = 0.0       # current conditional variance (rescaled units)
        self._last_ret = 0.0     # last observed rescaled return (for AR(1) lag)
        self._bars_since_refit = 0
        self._history: list[float] = []

    def _fit_params(self, returns_rescaled: pd.Series) -> None:
        # Rename to a predictable key so the AR coefficient is always 'y[1]'
        s = returns_rescaled.rename("y")
        kw = dict(mean="ARX", lags=1, vol="GARCH", p=1, q=1, dist=self.dist, rescale=False)
        if self.asymmetric:
            kw["o"] = 1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = arch_model(s, **kw).fit(disp="off", show_warning=False)

        p = res.params
        self._mu = float(p.get("Const", 0.0))
        self._phi = float(p["y[1]"])
        self._omega = float(p["omega"])
        self._alpha = float(p["alpha[1]"])
        self._beta = float(p["beta[1]"])
        self._gamma = float(p["gamma[1]"]) if self.asymmetric else 0.0
        self._nu = float(p.get("nu", float("nan")))

        # Unconditional variance for state initialization
        if self.asymmetric:
            denom = max(1e-12, 1.0 - self._alpha - self._beta - 0.5 * self._gamma)
        else:
            denom = max(1e-12, 1.0 - self._alpha - self._beta)
        self._sigma2 = self._omega / denom

    def fit(self, train: pd.Series) -> "JointARGARCHForecaster":
        s = train.dropna() * self.rescale
        self._history = s.tolist()
        self._fit_params(pd.Series(self._history, name="y"))
        self._last_ret = float(self._history[-1])
        self._bars_since_refit = 0
        return self

    def forecast_next_ret(self) -> float:
        """E[r_{t+1} | F_t] = mu + phi * r_t  (in raw return units)."""
        return float((self._mu + self._phi * self._last_ret) / self.rescale)

    def forecast_next_var(self) -> float:
        """Var[r_{t+1} | F_t] = sigma^2_{t+1}  (in raw return units)."""
        return float(self._sigma2 / (self.rescale ** 2))

    def update(self, observed_return: float) -> None:
        r = float(observed_return) * self.rescale
        # Residual from the AR(1) mean for THIS bar (drives the variance recursion)
        a = r - self._mu - self._phi * self._last_ret
        a2 = a * a
        if self.asymmetric:
            asym = self._gamma * a2 * (a < 0)
            self._sigma2 = self._omega + self._alpha * a2 + asym + self._beta * self._sigma2
        else:
            self._sigma2 = self._omega + self._alpha * a2 + self._beta * self._sigma2

        self._last_ret = r
        self._history.append(r)
        self._bars_since_refit += 1
        if self.refit_every and self._bars_since_refit >= self.refit_every:
            sigma2_carry = self._sigma2
            self._fit_params(pd.Series(self._history, name="y"))
            self._sigma2 = sigma2_carry
            self._bars_since_refit = 0


class JointARMAXGARCHForecaster:
    """AR(1) + exogenous regressors in the mean, GJR-GARCH(1,1)-t in variance.

    Mean equation:
        r_t = mu + phi * r_{t-1} + sum_k beta_k * X_{t,k} + a_t
        a_t = sigma_t * z_t,    z_t ~ Student-t(nu)

    Variance equation:
        sigma2_{t+1} = omega + alpha * a_t^2 + gamma * a_t^2 * 1[a_t<0]
                              + beta_g * sigma2_t

    Why this beats own-lag-only on intraday:
      - The mean equation now sees broad-market direction (NIFTY lag),
        fear regime (VIX), session structure (time-of-day), and the
        previous bar's microstructure (Parkinson range). These are precisely
        the things own-lag ARIMA is blind to.
      - The variance equation is unchanged — but because the residual is
        now computed against a richer mean, sigma2 picks up *only* the
        unexplained shock variance, which is theoretically more stable.

    Fitting is via arch_model(mean='ARX', x=X), which jointly estimates
    AR(1), exog coefficients, GARCH parameters, and t degrees-of-freedom.

    For walk-forward: parameters are frozen after fit, and we recurse
    BOTH the mean state (last return) and the variance state (sigma2)
    manually each bar. We also keep a pointer into the test-time X matrix
    so each ``forecast_next_ret()`` can grab the regressors for the next
    bar (which must be ALREADY OBSERVABLE — strictly lagged or deterministic
    like time-of-day).
    """

    def __init__(
        self,
        asymmetric: bool = True,
        dist: str = "t",
        rescale: float = 100.0,
    ) -> None:
        self.asymmetric = asymmetric
        self.dist = dist
        self.rescale = rescale
        self.name = "armax_gjr_garch_t" if asymmetric else "armax_garch_t"

        self._mu = 0.0
        self._phi = 0.0
        self._beta_x: np.ndarray | None = None
        self._x_cols: list[str] | None = None
        self._omega = 0.0
        self._alpha = 0.0
        self._beta_g = 0.0
        self._gamma = 0.0
        self._nu = float("nan")
        self._sigma2 = 0.0
        self._last_ret = 0.0
        # Pointer to next-bar exogenous row; set externally before each forecast.
        self._next_x: np.ndarray | None = None

    def fit(self, y_train: pd.Series, x_train: pd.DataFrame) -> "JointARMAXGARCHForecaster":
        common = y_train.index.intersection(x_train.index)
        y_aln = y_train.loc[common].rename("y")
        x_aln = x_train.loc[common]
        self._x_cols = list(x_aln.columns)

        y_rs = y_aln * self.rescale  # rescale returns; X stays in original units

        kw = dict(mean="ARX", lags=1, vol="GARCH", p=1, q=1, dist=self.dist, rescale=False)
        if self.asymmetric:
            kw["o"] = 1

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = arch_model(y_rs, x=x_aln, **kw).fit(disp="off", show_warning=False)

        p = res.params
        self._mu = float(p.get("Const", 0.0))
        self._phi = float(p["y[1]"])
        self._beta_x = np.array([float(p[c]) for c in self._x_cols])
        self._omega = float(p["omega"])
        self._alpha = float(p["alpha[1]"])
        self._beta_g = float(p["beta[1]"])
        self._gamma = float(p["gamma[1]"]) if self.asymmetric else 0.0
        self._nu = float(p.get("nu", float("nan")))

        # Unconditional variance for state init
        if self.asymmetric:
            denom = max(1e-12, 1.0 - self._alpha - self._beta_g - 0.5 * self._gamma)
        else:
            denom = max(1e-12, 1.0 - self._alpha - self._beta_g)
        self._sigma2 = self._omega / denom
        self._last_ret = float(y_rs.iloc[-1])
        return self

    def set_next_x(self, x_row: pd.Series | np.ndarray) -> None:
        """Provide the exogenous row that will be used for the NEXT bar's forecast.

        Must be called before ``forecast_next_ret`` between bars in a walk-forward.
        """
        if isinstance(x_row, pd.Series):
            arr = x_row.reindex(self._x_cols).to_numpy(dtype=float)
        else:
            arr = np.asarray(x_row, dtype=float)
        self._next_x = arr

    def forecast_next_ret(self) -> float:
        if self._next_x is None or self._beta_x is None:
            return float(self._mu / self.rescale)
        mean_rs = self._mu + self._phi * self._last_ret + float(self._beta_x @ self._next_x)
        return float(mean_rs / self.rescale)

    def forecast_next_var(self) -> float:
        return float(self._sigma2 / (self.rescale ** 2))

    def update(self, observed_return: float, x_used_for_this_bar: np.ndarray | pd.Series | None) -> None:
        """Advance state by one bar.

        x_used_for_this_bar is the exogenous row that was used to PREDICT the
        bar we just observed (i.e. what was set via set_next_x prior to this bar).
        Required so the residual a_t = r_t - predicted_mean is computed correctly.
        """
        r = float(observed_return) * self.rescale
        if x_used_for_this_bar is None or self._beta_x is None:
            x_term = 0.0
        else:
            if isinstance(x_used_for_this_bar, pd.Series):
                x_arr = x_used_for_this_bar.reindex(self._x_cols).to_numpy(dtype=float)
            else:
                x_arr = np.asarray(x_used_for_this_bar, dtype=float)
            x_term = float(self._beta_x @ x_arr)
        predicted_mean = self._mu + self._phi * self._last_ret + x_term
        a = r - predicted_mean
        a2 = a * a
        if self.asymmetric:
            asym = self._gamma * a2 * (a < 0)
            self._sigma2 = self._omega + self._alpha * a2 + asym + self._beta_g * self._sigma2
        else:
            self._sigma2 = self._omega + self._alpha * a2 + self._beta_g * self._sigma2
        self._last_ret = r


def walk_forward_vol(
    returns: pd.Series,
    model: VolForecaster,
    train_size: int,
) -> pd.DataFrame:
    """One-step-ahead variance forecasts, walk-forward, no look-ahead.

    Returns a DataFrame indexed like the test slice, with columns:
        forecast_var : sigma^2_{t} predicted at t-1
        actual_ret   : realized return at t
        actual_sq    : realized r_t^2 (noisy proxy for true variance)
    """
    s = returns.dropna()
    if train_size >= len(s):
        raise ValueError("train_size must be smaller than series length.")

    train = s.iloc[:train_size]
    test = s.iloc[train_size:]
    model.fit(train)

    n = len(test)
    var_fc = np.empty(n)
    actuals = test.to_numpy()
    for i, r in enumerate(actuals):
        var_fc[i] = model.forecast_next_var()
        model.update(float(r))

    return pd.DataFrame(
        {
            "forecast_var": var_fc,
            "actual_ret": actuals,
            "actual_sq": actuals**2,
        },
        index=test.index,
    )


def qlike(forecast_var: np.ndarray, realized_sq: np.ndarray) -> float:
    """QLIKE loss: log(sigma^2) + r^2 / sigma^2.

    QLIKE is the standard loss for volatility forecast comparison because,
    unlike MSE on r^2, it is robust to the fact that r^2 is a hideously
    noisy proxy for true variance (Patton 2011). Lower is better.
    """
    eps = 1e-20
    v = np.maximum(forecast_var, eps)
    return float(np.mean(np.log(v) + realized_sq / v))


def score_vol(results: pd.DataFrame) -> dict:
    """Aggregate volatility-forecast metrics."""
    v = results["forecast_var"].to_numpy()
    r2 = results["actual_sq"].to_numpy()

    return {
        "n": int(len(results)),
        "mse_on_r2": float(np.mean((r2 - v) ** 2)),
        "mae_on_r2": float(np.mean(np.abs(r2 - v))),
        "qlike": qlike(v, r2),
        "mean_forecast_vol": float(np.mean(np.sqrt(v))),
        "mean_realized_vol": float(np.mean(np.sqrt(r2))),
    }
