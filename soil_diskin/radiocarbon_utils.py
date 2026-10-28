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
        Ascending non-negative ages (years before 2000), starting at 0.
    fm : np.ndarray
        Corresponding R_14C (fraction modern) values at the knots.
    tail_fm : float
        R_14C beyond the last knot; defaults to the last value, fm[-1].
    """

    __slots__ = ("ages", "fm", "tail_fm")

    def __init__(self, ages: np.ndarray, fm: np.ndarray, tail_fm: float | None = None):
        self.ages = np.ascontiguousarray(ages, dtype=np.float64)
        self.fm = np.ascontiguousarray(fm, dtype=np.float64)
        self.tail_fm = float(self.fm[-1] if tail_fm is None else tail_fm)

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


def load_atm14c(path: str = "data/14C_atm_annot.csv") -> AtmC14:
    """Load atmospheric 14C CSV and return an `AtmC14`.

    Reads columns 4 (`years_before_2000`) and 5 (`R_14C`), sorts by age, keeps
    ages >= 0 (years after 2000 are never needed), and holds the value at the
    oldest age constant beyond the data.
    """
    df = pd.read_csv(path)
    raw_age = df.iloc[:, 3].to_numpy(dtype=np.float64)  # years_before_2000
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
