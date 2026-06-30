import argparse
from os import makedirs, path
import warnings

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from pytensor.compile.ops import wrap_py
from scipy.integrate import IntegrationWarning, quad
from scipy.optimize import least_squares
from soil_diskin.constants import GAMMA
from soil_diskin.continuum_models import GeneralPowerLawDisKin
from tqdm import tqdm

"""
Script should be run from the project root directory.

Fits a full posterior for the generalized power law model in PyMC with:

1. One shared beta across all sites.
2. A hierarchical prior on log(t_min), so the posterior learns the between-site
   spread instead of fixing it by optimization.
3. Site-specific log(t_max - t_min) parameters.

Because the model predictions rely on scipy.quad and the existing
GeneralPowerLawDisKin implementation, the likelihood is wrapped as a PyTensor
black-box Op. That means gradient-based samplers like NUTS are not available
here, so the default sampler is DEMetropolisZ.
"""

RELATIVE_EPS = 1e-6
LOG_PARAM_BOUNDS = (-16.0, 16.0)
LOG_LIKELIHOOD_FLOOR = -1e30
DEFAULT_BETA = 0.76
DEFAULT_BETA_MIN = 0.05
DEFAULT_BETA_MAX = 0.99
DEFAULT_DRAWS = 300
DEFAULT_TUNE = 300
DEFAULT_CHAINS = 4
DEFAULT_CORES = 4
DEFAULT_LOG_T_GAP_PRIOR_SCALE = 1.5
DEFAULT_HDI_PROB = 0.94
DEFAULT_TURNOVER_SIGMA_REL_INIT = 0.05
DEFAULT_FM_SIGMA_REL_INIT = 0.03
DEFAULT_SAMPLER = 'demetropolisz'
DEFAULT_RANDOM_SEED = 20260423
LEAST_SQUARES_KWARGS = {
    'method': 'trf',
    'ftol': 1e-10,
    'xtol': 1e-10,
    'gtol': 1e-10,
}


def unpack_log_params(log_params):
    """Convert unconstrained log parameters into ordered model parameters."""
    log_t_min, log_t_gap = log_params
    t_min = np.exp(log_t_min)
    t_max = t_min + np.exp(log_t_gap)
    return t_min, t_max


def initial_log_guess(site_data):
    """Choose a scale-aware initial guess for the site being optimized."""
    t_min_guess = max(1e-6, float(site_data['turnover']) / 1000)
    t_gap_guess = max(float(site_data['turnover']), 1000)
    return np.log([t_min_guess, t_gap_guess])


def safe_model_predictions(log_params, beta):
    """Evaluate modeled turnover and radiocarbon with failure protection."""
    t_min, t_max = unpack_log_params(log_params)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', IntegrationWarning)
            model = GeneralPowerLawDisKin(t_min=t_min, t_max=t_max, beta=beta)
            modeled_14c = quad(
                model.radiocarbon_age_integrand,
                0,
                np.inf,
                limit=1500,
                epsabs=1e-3,
            )[0]
    except (FloatingPointError, IntegrationWarning, OverflowError, RuntimeError, ValueError):
        return np.nan, np.nan, False

    finite = np.isfinite(model.T) and np.isfinite(modeled_14c)
    return model.T, modeled_14c, finite and model.params_valid()


def objective_residuals(log_params, beta, observed_fm, observed_turnover):
    """Relative residual vector comparing model predictions and observations."""
    modeled_turnover, modeled_14c, valid = safe_model_predictions(log_params, beta)
    if not valid:
        return np.array([1e6, 1e6], dtype=float)

    relative_diff_14c = (modeled_14c - observed_fm) / (observed_fm + RELATIVE_EPS)
    relative_diff_turnover = (
        modeled_turnover - observed_turnover
    ) / (observed_turnover + RELATIVE_EPS)
    return np.array([relative_diff_14c, relative_diff_turnover], dtype=float)


def objective_function(log_params, beta, observed_fm, observed_turnover):
    """Return the sum of squared relative residuals for diagnostics/output."""
    residuals = objective_residuals(log_params, beta, observed_fm, observed_turnover)
    return float(np.dot(residuals, residuals))


def fit_sites_independently(merged_site_data, beta):
    """Independent per-site fit used to initialize the PyMC model."""
    n_sites = len(merged_site_data)
    site_log_params = np.zeros((n_sites, 2), dtype=float)
    objective_values = np.zeros(n_sites, dtype=float)
    lower = np.full(2, LOG_PARAM_BOUNDS[0], dtype=float)
    upper = np.full(2, LOG_PARAM_BOUNDS[1], dtype=float)

    print(f'Running independent initialization at beta={beta:.4f}...')
    iterator = tqdm(merged_site_data.iterrows(), total=n_sites)
    for idx, (_, row) in enumerate(iterator):
        res = least_squares(
            objective_residuals,
            initial_log_guess(row),
            bounds=(lower, upper),
            args=(beta, float(row['fm']), float(row['turnover'])),
            **LEAST_SQUARES_KWARGS,
        )
        site_log_params[idx] = res.x
        objective_values[idx] = objective_function(
            res.x,
            beta,
            float(row['fm']),
            float(row['turnover']),
        )

    return site_log_params, objective_values


def predict_sites(beta, log_t_min, log_t_gap):
    """Evaluate model predictions for all sites at one parameter draw."""
    n_sites = len(log_t_min)
    modeled_turnover = np.zeros(n_sites, dtype=float)
    modeled_14c = np.zeros(n_sites, dtype=float)
    params_valid = np.zeros(n_sites, dtype=bool)

    for site_idx in range(n_sites):
        current_turnover, current_14c, valid = safe_model_predictions(
            np.array([log_t_min[site_idx], log_t_gap[site_idx]], dtype=float),
            float(beta),
        )
        modeled_turnover[site_idx] = current_turnover
        modeled_14c[site_idx] = current_14c
        params_valid[site_idx] = valid

    return modeled_turnover, modeled_14c, params_valid


def gaussian_loglike(observed, modeled, sigma):
    """Elementwise Gaussian log-likelihood summed over a vector."""
    standardized = (observed - modeled) / sigma
    return float(-0.5 * np.sum(standardized ** 2 + 2 * np.log(sigma) + np.log(2 * np.pi)))


def make_joint_loglike_op(observed_fm, observed_turnover):
    """Create a PyTensor Op wrapping the existing numerical model likelihood."""
    observed_fm = np.asarray(observed_fm, dtype=float)
    observed_turnover = np.asarray(observed_turnover, dtype=float)

    @wrap_py(
        itypes=[pt.dscalar, pt.dvector, pt.dvector, pt.dscalar, pt.dscalar],
        otypes=[pt.dscalar],
    )
    def joint_loglike_op(beta, log_t_min, log_t_gap, turnover_sigma_rel, fm_sigma_rel):
        beta = float(np.asarray(beta))
        log_t_min = np.asarray(log_t_min, dtype=float)
        log_t_gap = np.asarray(log_t_gap, dtype=float)
        turnover_sigma_rel = float(np.asarray(turnover_sigma_rel))
        fm_sigma_rel = float(np.asarray(fm_sigma_rel))

        if turnover_sigma_rel <= 0 or fm_sigma_rel <= 0:
            return np.asarray(LOG_LIKELIHOOD_FLOOR, dtype=np.float64)

        modeled_turnover, modeled_14c, params_valid = predict_sites(beta, log_t_min, log_t_gap)
        if not np.all(params_valid):
            return np.asarray(LOG_LIKELIHOOD_FLOOR, dtype=np.float64)

        turnover_sigma = np.maximum(turnover_sigma_rel * observed_turnover, RELATIVE_EPS)
        fm_sigma = np.maximum(fm_sigma_rel * np.maximum(np.abs(observed_fm), RELATIVE_EPS), RELATIVE_EPS)
        logp = gaussian_loglike(observed_turnover, modeled_turnover, turnover_sigma)
        logp += gaussian_loglike(observed_fm, modeled_14c, fm_sigma)

        if not np.isfinite(logp):
            logp = LOG_LIKELIHOOD_FLOOR
        return np.asarray(logp, dtype=np.float64)

    return joint_loglike_op


def build_model(
    merged_site_data,
    beta_init=DEFAULT_BETA,
    beta_min=DEFAULT_BETA_MIN,
    beta_max=DEFAULT_BETA_MAX,
    log_t_gap_prior_scale=DEFAULT_LOG_T_GAP_PRIOR_SCALE,
):
    """Construct the PyMC model and initialization diagnostics."""
    observed_fm = merged_site_data['fm'].to_numpy(dtype=float)
    observed_turnover = merged_site_data['turnover'].to_numpy(dtype=float)
    n_sites = len(merged_site_data)

    independent_log_params, independent_objectives = fit_sites_independently(
        merged_site_data,
        beta=beta_init,
    )
    independent_log_t_min = independent_log_params[:, 0]
    independent_log_t_gap = independent_log_params[:, 1]
    mu_log_t_min_init = float(np.mean(independent_log_t_min))
    sigma_log_t_min_init = float(max(np.std(independent_log_t_min), 0.2))
    mu_log_t_min_prior_scale = float(max(2.0 * sigma_log_t_min_init, 1.0))
    sigma_log_t_min_prior_scale = float(max(2.0 * sigma_log_t_min_init, 0.5))
    log_t_min_offset_init = np.clip(
        (independent_log_t_min - mu_log_t_min_init) / sigma_log_t_min_init,
        -3.0,
        3.0,
    )

    coords = {'site': np.arange(n_sites, dtype=int)}
    joint_loglike_op = make_joint_loglike_op(observed_fm, observed_turnover)

    with pm.Model(coords=coords) as model:
        beta = pm.TruncatedNormal(
            'beta',
            mu=beta_init,
            sigma=0.2,
            lower=beta_min,
            upper=beta_max,
            initval=beta_init,
        )
        mu_log_t_min = pm.Normal(
            'mu_log_t_min',
            mu=mu_log_t_min_init,
            sigma=mu_log_t_min_prior_scale,
            initval=mu_log_t_min_init,
        )
        sigma_log_t_min = pm.HalfNormal(
            'sigma_log_t_min',
            sigma=sigma_log_t_min_prior_scale,
            initval=sigma_log_t_min_init,
        )
        log_t_min_offset = pm.Normal(
            'log_t_min_offset',
            mu=0.0,
            sigma=1.0,
            dims='site',
            initval=log_t_min_offset_init,
        )
        log_t_min = pm.Deterministic(
            'log_t_min',
            mu_log_t_min + sigma_log_t_min * log_t_min_offset,
            dims='site',
        )
        log_t_gap = pm.Normal(
            'log_t_gap',
            mu=independent_log_t_gap,
            sigma=log_t_gap_prior_scale,
            dims='site',
            initval=independent_log_t_gap,
        )
        t_min = pm.Deterministic('t_min', pt.exp(log_t_min), dims='site')
        pm.Deterministic('t_max', t_min + pt.exp(log_t_gap), dims='site')
        turnover_sigma_rel = pm.HalfNormal(
            'turnover_sigma_rel',
            sigma=0.1,
            initval=DEFAULT_TURNOVER_SIGMA_REL_INIT,
        )
        fm_sigma_rel = pm.HalfNormal(
            'fm_sigma_rel',
            sigma=0.05,
            initval=DEFAULT_FM_SIGMA_REL_INIT,
        )

        pm.Potential(
            'site_likelihood',
            joint_loglike_op(beta, log_t_min, log_t_gap, turnover_sigma_rel, fm_sigma_rel),
        )

    init_data = {
        'independent_log_params': independent_log_params,
        'independent_objectives': independent_objectives,
        'mu_log_t_min_init': mu_log_t_min_init,
        'sigma_log_t_min_init': sigma_log_t_min_init,
        'observed_fm': observed_fm,
        'observed_turnover': observed_turnover,
    }
    return model, init_data


def sample_model(model, sampler, draws, tune, chains, cores, random_seed):
    """Run posterior sampling with a sampler compatible with the black-box likelihood."""
    with model:
        if sampler == 'smc':
            return pm.sample_smc(
                draws=draws,
                chains=chains,
                cores=cores,
                random_seed=random_seed,
                return_inferencedata=True,
                progressbar=True,
            )

        if sampler == 'demetropolisz':
            if chains < 2:
                raise ValueError('DEMetropolisZ needs at least 2 chains. Use --chains >= 2 or --sampler metropolis.')
            step = pm.DEMetropolisZ()
        elif sampler == 'metropolis':
            step = pm.Metropolis()
        else:
            raise ValueError(f'Unsupported sampler: {sampler}')

        return pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            step=step,
            random_seed=random_seed,
            return_inferencedata=True,
            discard_tuned_samples=True,
            progressbar=True,
        )


def hdi_bounds(data_array, hdi_prob):
    """Return lower and upper HDI bounds with the hdi axis last."""
    values = np.asarray(data_array)
    flat_values = values.reshape((-1,) + values.shape[2:])
    if flat_values.ndim == 1:
        return highest_density_interval(flat_values, hdi_prob)

    lower = np.zeros(flat_values.shape[1:], dtype=float)
    upper = np.zeros(flat_values.shape[1:], dtype=float)
    for idx in np.ndindex(flat_values.shape[1:]):
        lower[idx], upper[idx] = highest_density_interval(flat_values[(slice(None),) + idx], hdi_prob)
    return lower, upper


def highest_density_interval(samples, hdi_prob):
    """Compute a highest-density interval from a 1D sample vector."""
    sorted_samples = np.sort(np.asarray(samples, dtype=float))
    n_samples = sorted_samples.size
    if n_samples == 0:
        raise ValueError('Cannot compute an HDI from an empty sample vector.')

    interval_size = int(np.ceil(hdi_prob * n_samples))
    if interval_size >= n_samples:
        return float(sorted_samples[0]), float(sorted_samples[-1])

    interval_width = sorted_samples[interval_size:] - sorted_samples[:n_samples - interval_size]
    start_idx = int(np.argmin(interval_width))
    end_idx = start_idx + interval_size
    return float(sorted_samples[start_idx]), float(sorted_samples[end_idx])


def global_summary_dataframe(idata, hdi_prob):
    """Summarize the key global posterior parameters."""
    posterior = idata.posterior
    parameter_names = [
        'beta',
        'mu_log_t_min',
        'sigma_log_t_min',
        'turnover_sigma_rel',
        'fm_sigma_rel',
    ]
    rows = []

    for name in parameter_names:
        values = np.asarray(posterior[name]).reshape(-1)
        lower, upper = hdi_bounds(posterior[name], hdi_prob=hdi_prob)
        rows.append({
            'parameter': name,
            'mean': float(np.mean(values)),
            'median': float(np.median(values)),
            'sd': float(np.std(values, ddof=1)),
            'hdi_lower': float(np.asarray(lower)),
            'hdi_upper': float(np.asarray(upper)),
        })

    return pd.DataFrame(rows)


def site_summary_dataframe(merged_site_data, idata, init_data, hdi_prob):
    """Summarize the site-level posterior parameters and posterior-mean fit."""
    posterior = idata.posterior
    result_df = merged_site_data.reset_index(drop=True).copy()
    posterior_mean_beta = float(np.asarray(posterior['beta']).mean())

    summary_vars = ['log_t_min', 't_min', 'log_t_gap', 't_max']
    for name in summary_vars:
        data_array = posterior[name]
        mean_values = np.asarray(data_array.mean(dim=('chain', 'draw')))
        median_values = np.asarray(data_array.median(dim=('chain', 'draw')))
        lower, upper = hdi_bounds(data_array, hdi_prob=hdi_prob)
        result_df[f'{name}_post_mean'] = mean_values
        result_df[f'{name}_post_median'] = median_values
        result_df[f'{name}_hdi_lower'] = lower
        result_df[f'{name}_hdi_upper'] = upper

    posterior_mean_log_t_min = np.asarray(posterior['log_t_min'].mean(dim=('chain', 'draw')))
    posterior_mean_log_t_gap = np.asarray(posterior['log_t_gap'].mean(dim=('chain', 'draw')))
    modeled_turnover, modeled_14c, params_valid = predict_sites(
        posterior_mean_beta,
        posterior_mean_log_t_min,
        posterior_mean_log_t_gap,
    )

    independent_t_min = np.exp(init_data['independent_log_params'][:, 0])
    independent_t_max = independent_t_min + np.exp(init_data['independent_log_params'][:, 1])
    result_df['posterior_mean_beta'] = posterior_mean_beta
    result_df['posterior_mean_modeled_T'] = modeled_turnover
    result_df['posterior_mean_modeled_14C'] = modeled_14c
    result_df['posterior_mean_params_valid'] = params_valid
    result_df['independent_t_min_init'] = independent_t_min
    result_df['independent_t_max_init'] = independent_t_max
    result_df['independent_objective_init'] = init_data['independent_objectives']
    return result_df


def parse_args():
    """Command-line configuration for the PyMC posterior fit."""
    parser = argparse.ArgumentParser(
        description='PyMC posterior calibration for the generalized power law model.'
    )
    parser.add_argument(
        '--input-path',
        default='results/all_sites_14C_turnover.csv',
        help='CSV with fm and turnover observations for each site.',
    )
    parser.add_argument(
        '--output-dir',
        default='results/03_calibrate_models',
        help='Directory for posterior outputs.',
    )
    parser.add_argument(
        '--output-prefix',
        default='general_powerlaw_model_pymc',
        help='Prefix used for all output filenames.',
    )
    parser.add_argument(
        '--site-limit',
        type=int,
        default=None,
        help='Optional number of sites to keep from the top of the input table for smoke tests.',
    )
    parser.add_argument(
        '--draws',
        type=int,
        default=DEFAULT_DRAWS,
        help='Posterior draws per chain.',
    )
    parser.add_argument(
        '--tune',
        type=int,
        default=DEFAULT_TUNE,
        help='Warmup steps per chain for MCMC samplers.',
    )
    parser.add_argument(
        '--chains',
        type=int,
        default=DEFAULT_CHAINS,
        help='Number of chains.',
    )
    parser.add_argument(
        '--cores',
        type=int,
        default=DEFAULT_CORES,
        help='Number of parallel worker processes.',
    )
    parser.add_argument(
        '--sampler',
        choices=['demetropolisz', 'metropolis', 'smc'],
        default=DEFAULT_SAMPLER,
        help='Sampler to use. NUTS is not available because the likelihood is a black-box Op.',
    )
    parser.add_argument(
        '--beta-init',
        type=float,
        default=DEFAULT_BETA,
        help='Starting value for the shared beta parameter.',
    )
    parser.add_argument(
        '--beta-min',
        type=float,
        default=DEFAULT_BETA_MIN,
        help='Lower bound for the shared beta parameter.',
    )
    parser.add_argument(
        '--beta-max',
        type=float,
        default=DEFAULT_BETA_MAX,
        help='Upper bound for the shared beta parameter.',
    )
    parser.add_argument(
        '--log-t-gap-prior-scale',
        type=float,
        default=DEFAULT_LOG_T_GAP_PRIOR_SCALE,
        help='Prior scale for site-level log(t_max - t_min).',
    )
    parser.add_argument(
        '--hdi-prob',
        type=float,
        default=DEFAULT_HDI_PROB,
        help='Posterior HDI probability for summaries.',
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help='Random seed for posterior sampling.',
    )
    parser.add_argument(
        '--gamma-default',
        action='store_true',
        help='Initialize beta with exp(-GAMMA) instead of the explicit default.',
    )
    return parser.parse_args()


def main():
    """Run the PyMC posterior fit and write posterior summaries to disk."""
    args = parse_args()
    beta_init = np.exp(-GAMMA) if args.gamma_default else args.beta_init

    merged_site_data = pd.read_csv(args.input_path)
    if args.site_limit is not None:
        merged_site_data = merged_site_data.head(args.site_limit).copy()

    model, init_data = build_model(
        merged_site_data,
        beta_init=beta_init,
        beta_min=args.beta_min,
        beta_max=args.beta_max,
        log_t_gap_prior_scale=args.log_t_gap_prior_scale,
    )
    idata = sample_model(
        model,
        sampler=args.sampler,
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        cores=args.cores,
        random_seed=args.random_seed,
    )

    global_summary_df = global_summary_dataframe(idata, hdi_prob=args.hdi_prob)
    site_summary_df = site_summary_dataframe(
        merged_site_data,
        idata,
        init_data,
        hdi_prob=args.hdi_prob,
    )

    makedirs(args.output_dir, exist_ok=True)
    inference_output_path = path.join(args.output_dir, f'{args.output_prefix}_posterior.nc')
    global_summary_path = path.join(args.output_dir, f'{args.output_prefix}_global_summary.csv')
    site_summary_path = path.join(args.output_dir, f'{args.output_prefix}_site_summary.csv')

    idata.to_netcdf(inference_output_path)
    global_summary_df.to_csv(global_summary_path, index=False)
    site_summary_df.to_csv(site_summary_path, index=False)

    beta_summary = global_summary_df.loc[global_summary_df['parameter'] == 'beta'].iloc[0]
    print('Finished PyMC posterior calibration.')
    print(f'sampler = {args.sampler}')
    print(f'n_sites = {len(merged_site_data)}')
    print(f'beta mean = {beta_summary["mean"]:.5f}')
    print(f'beta {args.hdi_prob:.2f} HDI = [{beta_summary["hdi_lower"]:.5f}, {beta_summary["hdi_upper"]:.5f}]')
    print(f'wrote posterior draws to {inference_output_path}')
    print(f'wrote global summary to {global_summary_path}')
    print(f'wrote site summary to {site_summary_path}')


if __name__ == '__main__':
    main()