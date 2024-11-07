import numpy as np
from scipy.interpolate import make_interp_spline, interp1d


class ClampedInterpolator:
    """A class that interpolates between points using a spline interpolator.

    Returns the nearest value if the input is outside the range of the data.
    """

    def __init__(self, x, y):
        """Uses a spline interpolator to interpolate between points in x and y.

        x: array-like
            The x values of the data points
        y: array-like
            The y values of the data points
        """
        #self.interpolator = make_interp_spline(x, y)
        self.interpolator = interp1d(x, y, kind='zero')
        self.lb, self.ub = x.min(), x.max()

    def call_single(self, x):
        """Apply the interpolator to a scalar x value.

        If x is outside the range of the data, the interpolator is clamped to the nearest value.
        """
        if x <= self.lb:
            return self.interpolator(self.lb)
        if x >= self.ub:
            return self.interpolator(self.ub)
        return self.interpolator(x)

    def __call__(self, x):
        """Apply the interpolator to the given x value.

        If x is outside the range of the data, the interpolator is clamped to the nearest value.
        """
        if isinstance(x, (list, np.ndarray)):
            xs = np.array(x)
            new_xs = np.where(xs <= self.lb, self.lb, xs)
            new_xs = np.where(new_xs >= self.ub, self.ub, new_xs)
            return self.interpolator(new_xs)
        return self.call_single(x)