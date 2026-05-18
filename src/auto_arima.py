"""Lightweight auto-ARIMA via AIC grid search.

We deliberately avoid `pmdarima` (no Python 3.14 wheels and the extra abstraction
hides what's happening). This module exposes a single function that fits every
ARIMA(p, d, q) on a grid and returns the lowest-AIC order.

Trade-off vs pmdarima: no stepwise pruning, so a 4x4 grid means 16 fits instead
of ~6. For our 15m sample sizes this is still fast (<30s) and the brute-force
search is more transparent for teaching.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA


@dataclass(frozen=True)
class ARIMAOrder:
    p: int
    d: int
    q: int
    aic: float

    @property
    def order(self) -> tuple[int, int, int]:
        return (self.p, self.d, self.q)

    def __repr__(self) -> str:
        return f"ARIMA({self.p},{self.d},{self.q})  AIC={self.aic:.2f}"


def grid_search(
    series: pd.Series,
    max_p: int = 3,
    max_q: int = 3,
    d: int = 0,
    trend: str | None = "n",
) -> tuple[ARIMAOrder, pd.DataFrame]:
    """Return the lowest-AIC ARIMA order plus the full grid as a DataFrame.

    Args:
        series: Stationary series (typically log-returns). Pass already-differenced
            data and keep d=0, rather than passing raw prices with d=1 — gives you
            cleaner control over what's being modelled.
        max_p, max_q: Inclusive upper bounds on AR and MA orders.
        d: Differencing order. Set to 0 if `series` is already stationary.
        trend: statsmodels trend spec. 'n' (none) is correct for zero-mean returns;
            'c' adds an intercept (almost always insignificant on returns).
    """
    results: list[dict] = []
    s = series.dropna()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", UserWarning)
        for p, q in product(range(max_p + 1), range(max_q + 1)):
            if p == 0 and q == 0:
                # ARIMA(0,0,0) with no trend is just a constant — degenerate.
                continue
            try:
                fit = ARIMA(s, order=(p, d, q), trend=trend).fit(method_kwargs={"warn_convergence": False})
                results.append({"p": p, "d": d, "q": q, "aic": fit.aic, "bic": fit.bic})
            except (np.linalg.LinAlgError, ValueError):
                # Some orders fail to converge on short / near-noise series; skip.
                continue

    if not results:
        raise RuntimeError("No ARIMA order converged on this series.")

    grid = pd.DataFrame(results).sort_values("aic").reset_index(drop=True)
    best_row = grid.iloc[0]
    best = ARIMAOrder(int(best_row.p), int(best_row.d), int(best_row.q), float(best_row.aic))
    return best, grid
