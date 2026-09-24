"""Independent layers: allocate NPP, configure the built-in model, fit stock/Fm.

Depths: cm; stocks: kg C/m²; inputs: kg C/m²/year; rates: year⁻¹.
"""
from dataclasses import asdict, dataclass
import numpy as np
from scipy.optimize import least_squares
from .continuum_models import LognormalDisKinFast

N_LAYERS, DZ = 10, 10.0


@dataclass(frozen=True)
class InputAllocation:
    input_depth_cm: float = 30.
    surface_fraction: float = 0.
    soil_npp_fraction: float = 1.

    def __post_init__(self):
        if not np.isfinite(self.input_depth_cm) or self.input_depth_cm <= 0:
            raise ValueError('input_depth_cm must be finite and positive')
        if not 0 <= self.surface_fraction < 1:
            raise ValueError('surface_fraction must be in [0, 1)')
        if not 0 < self.soil_npp_fraction <= 1:
            raise ValueError('soil_npp_fraction must be in (0, 1]')

    @property
    def soil_input_fractions(self) -> np.ndarray:
        h = self.input_depth_cm
        weights = np.exp(-np.arange(N_LAYERS)*DZ/h)*-np.expm1(-DZ/h)/-np.expm1(-N_LAYERS*DZ/h)
        weights *= 1-self.surface_fraction
        weights[0] += self.surface_fraction
        return weights

    @property
    def npp_fractions(self) -> np.ndarray:
        return self.soil_npp_fraction*self.soil_input_fractions

    def layer_inputs(self, npp: float) -> np.ndarray:
        if not np.isfinite(npp) or npp <= 0:
            raise ValueError('site NPP must be finite and positive')
        return npp*self.soil_npp_fraction*self.soil_input_fractions

    @property
    def metadata(self) -> dict:
        return {**asdict(self), 'layer_input_weights': self.soil_input_fractions.tolist(),
                'layer_npp_fractions': self.npp_fractions.tolist()}


def layer_model(atmosphere, log_rate_step: float = .05) -> LognormalDisKinFast:
    """Prepare the existing model's fixed quadrature once for all layer fits."""
    model = LognormalDisKinFast(0., 1., atmosphere)
    model.prepare_quadrature(log_rate_step=log_rate_step)
    return model


@dataclass
class FitResult:
    """One start's complete result schema, including failures."""
    start_id: int = -1
    mu: float = np.nan
    sigma: float = np.nan
    stock_pred_kg_m2: float = np.nan
    fm_pred: float = np.nan
    stock_scaled_residual: float = np.nan
    fm_scaled_residual: float = np.nan
    success: bool = False
    objective: float = np.inf
    nfev: int = 0
    mu_at_bound: bool = False
    sigma_at_bound: bool = False
    jacobian_rank: int = 0
    message: str = ''
    candidate_id: int = 0
    near_best: bool = False
    model_turnover_years: float = np.nan


def fit_layer(model: LognormalDisKinFast, stock: float, fm: float, input_rate: float, *,
              max_nfev: int = 500, stock_relative_scale: float = .1,
              fm_scale: float = .02) -> list[FitResult]:
    """Fit stock/Fm from three starts, best first; observed f_new is never an input."""
    if (not np.isfinite([stock, fm, input_rate, stock_relative_scale, fm_scale]).all()
            or min(stock, input_rate, stock_relative_scale, fm_scale) <= 0
            or not isinstance(max_nfev, int) or max_nfev < 1):
        raise ValueError('finite observations, positive stock/input/scales, and positive max_nfev required')
    lower, upper = np.array([model.mu_bounds, model.sigma_bounds]).T
    scales = np.array([stock_relative_scale*stock, fm_scale])

    def residual(parameters):
        prediction = model.predict(*parameters, input_rate)
        return (np.array([prediction.stock, prediction.fm])-[stock, fm])/scales

    candidates = []
    for start_id, sigma0 in enumerate([2.5, 1., 4.]):
        sigma0 = np.clip(sigma0, lower[1], upper[1])
        mu0 = sigma0**2/2 + np.log(input_rate)-np.log(stock)
        row = FitResult(start_id=start_id)
        try:
            fit = least_squares(residual, np.clip([mu0, sigma0], lower, upper),
                                bounds=(lower, upper), x_scale='jac', max_nfev=max_nfev,
                                ftol=1e-10, xtol=1e-10, gtol=1e-10)
            pred = model.predict(float(fit.x[0]), float(fit.x[1]), input_rate)
            singular = np.linalg.svd(fit.jac*(upper-lower), compute_uv=False)
            at_bound = np.minimum(fit.x-lower, upper-fit.x)/(upper-lower) < 1e-5
            row = FitResult(start_id=start_id, mu=fit.x[0], sigma=fit.x[1], stock_pred_kg_m2=pred.stock,
                       fm_pred=pred.fm, stock_scaled_residual=fit.fun[0],
                       fm_scaled_residual=fit.fun[1], success=bool(fit.success),
                       objective=float(fit.fun@fit.fun), nfev=fit.nfev,
                       mu_at_bound=bool(at_bound[0]), sigma_at_bound=bool(at_bound[1]),
                       jacobian_rank=int(np.count_nonzero(singular > singular[0]*1e-8)),
                       message=fit.message, model_turnover_years=model.T)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
            row.message = str(error)
        candidates.append(row)
    candidates.sort(key=lambda row: row.objective)
    for index, row in enumerate(candidates):
        row.candidate_id = index
        row.near_best = row.success and row.objective <= candidates[0].objective*1.01+1e-6
    return candidates
