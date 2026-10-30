"""Radiocarbon utilities.

AtmC14 is the single representation of the atmospheric 14C record used by all
models: a piecewise-constant lookup over age with a constant tail.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

__all__ = ["AtmC14", "load_atm14c", "default_atmosphere"]


class AtmC14:
    """Piecewise-constant atmospheric 14C lookup with a constant tail.

    R(a) = fm[i] for ages[i] <= a < ages[i+1], and tail_fm beyond the last
    knot. This matches the zero-order interpolation the models used before,
    which also held the last value beyond the data.

    Attributes
    ----------
    ages : np.ndarray
        Ascending non-negative ages, starting at 0: years before 2000, or before
        the reference year passed to load_atm14c.
    fm : np.ndarray
        Corresponding R_14C (fraction modern) values at the knots.
    tail_fm : float
        R_14C beyond the last knot; defaults to the last value, fm[-1].
    """

    __slots__ = ("ages", "fm", "tail_fm", "_log_rate_response")

    # Grid of u = ln k (k in yr^-1) for rate-space radiocarbon calculations.
    # The 0.005 step resolves log-rate densities with sigma >= 0.01, and the range
    # covers k from e^-100 to e^70 per year with room for 12-sigma tails.
    LOG_RATE_GRID = np.linspace(-100., 70., 34001)

    def __init__(self, ages: np.ndarray, fm: np.ndarray, tail_fm: float | None = None):
        self.ages = np.ascontiguousarray(ages, dtype=np.float64)
        self.fm = np.ascontiguousarray(fm, dtype=np.float64)
        self.tail_fm = float(self.fm[-1] if tail_fm is None else tail_fm)
        self._log_rate_response = None

    def R(self, a):
        """Atmospheric 14C ratio a years before 2000 (scalar or array, a >= 0)."""
        a = np.asarray(a, dtype=np.float64)
        index = np.searchsorted(self.ages, a, side="right") - 1
        result = np.where(a >= self.ages[-1], self.tail_fm, self.fm[np.clip(index, 0, None)])
        return float(result) if result.ndim == 0 else result

    def laplace(self, alpha: float) -> float:
        """Closed form for the integral of R(a) exp(-alpha a) over a >= 0, alpha > 0."""
        e = np.exp(-alpha * self.ages)
        body = self.fm[:-1] * (e[:-1] - e[1:])
        return (body.sum() + self.tail_fm * e[-1]) / alpha

    def log_rate_response(self):
        """(u, I(e^u + lambda)) on LOG_RATE_GRID, with I = laplace; computed once.

        A pool decaying at rate k contributes I(k + lambda) to the steady-state
        14C stock per unit input, where lambda is the 14C decay constant.
        """
        if self._log_rate_response is None:
            from soil_diskin.constants import LAMBDA_14C
            u = self.LOG_RATE_GRID
            self._log_rate_response = (u, np.array([self.laplace(k + LAMBDA_14C) for k in np.exp(u)]))
        return self._log_rate_response


def load_atm14c(path: str = "data/14C_atm_annot.csv", reference_year: float = 2000) -> AtmC14:
    """Load atmospheric 14C CSV and return an `AtmC14` with ages counted back from `reference_year`.

    Reads columns 4 (`years_before_2000`) and 5 (`R_14C`), sorts by age, keeps
    ages >= 0 (years after the reference year are never needed), and holds the
    value at the oldest age constant beyond the data.

    Parameters
    ----------
    path : str
        CSV of the atmospheric record, as written by 01_preprocess_14C_history.py
        (data/14C_atm_annot.csv).
    reference_year : float
        Calendar year from which ages are counted: carbon of age a was fixed from
        the atmosphere in reference_year - a. The default, 2000, is the year of the
        Shi et al. (2020) radiocarbon map used for the Balesdent sites; samples
        from other years (e.g. ISRaD profiles) need their sampling year.

    Returns
    -------
    AtmC14
        The record as R_14C against age, in years before `reference_year`.

    Raises
    ------
    ValueError
        If the record ends before `reference_year`, so that the youngest ages
        are not covered.
    """
    df = pd.read_csv(path)
    raw_age = df.iloc[:, 3].to_numpy(dtype=np.float64) - (2000 - reference_year)  # years before reference_year
    if raw_age.min() > 0:
        raise ValueError(f'the atmospheric record ends {raw_age.min():g} years before {reference_year}')
    raw_fm = df.iloc[:, 4].to_numpy(dtype=np.float64)   # R_14C
    perm = np.argsort(raw_age)
    a = raw_age[perm]
    f = raw_fm[perm]
    mask = a >= 0.0
    return AtmC14(a[mask], f[mask])


@lru_cache(maxsize=1)
def default_atmosphere() -> AtmC14:
    """The project's atmospheric record, loaded once from constants.C14_DATA_PATH."""
    from soil_diskin import constants
    return load_atm14c(constants.C14_DATA_PATH)
