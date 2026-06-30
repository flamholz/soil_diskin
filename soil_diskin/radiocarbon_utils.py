"""Radiocarbon utilities.

Small helper container types for atmospheric 14C data used by the
lognormal/Diskin ports in `notebooks/experimental`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["AtmC14", "load_atm14c"]


class AtmC14:
    """Piecewise-constant atmospheric 14C lookup with a constant tail.

    Replicates the behavior of prior Mathematica code. 

    Attributes
    ----------
    ages : np.ndarray
        Ascending non-negative ages (years before 2000).
    fm : np.ndarray
        Corresponding R_14C (fraction modern) values at the knots.
    mean_R : float
        Mean R used for the constant tail beyond the last knot.
    """

    __slots__ = ("ages", "fm", "mean_R")

    def __init__(self, ages: np.ndarray, fm: np.ndarray, mean_R: float):
        self.ages = np.ascontiguousarray(ages, dtype=np.float64)
        self.fm = np.ascontiguousarray(fm, dtype=np.float64)
        self.mean_R = float(mean_R)


def load_atm14c(path: str = "data/14C_atm_annot.csv") -> AtmC14:
    """Load atmospheric 14C CSV and return an `AtmC14`.

    This replicates the behavior of the experimental port:
    - read CSV with header
    - take columns 4 (`years_before_2000`) and 5 (`R_14C`)
    - mean_R = mean of R_14C over the LAST 50 000 rows (in original file order)
    - keep only ages >= 0, sorted ascending
    """
    df = pd.read_csv(path)
    raw_age = df.iloc[:, 3].to_numpy(dtype=np.float64)  # years_before_2000
    raw_fm = df.iloc[:, 4].to_numpy(dtype=np.float64)   # R_14C
    # Unclear why the limit of 50k rows is here. 
    # TODO: Discuss with Yinon and document the rationale if we keep it.
    mean_R = float(raw_fm[-50_000:].mean())
    perm = np.argsort(raw_age)
    a = raw_age[perm]
    f = raw_fm[perm]
    mask = a >= 0.0
    return AtmC14(a[mask], f[mask], mean_R)
