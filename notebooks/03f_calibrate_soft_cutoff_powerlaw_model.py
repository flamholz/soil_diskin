"""
Script should be run from the project root directory.

Calibrates the soft-cutoff power law model (SoftCutoffPowerLawDisKin) with
alpha fixed, sweeping alpha from 0.2 to 1.0 in steps of 0.1, plus 0.55, 0.65
and 0.75. For each alpha, t_min and t_max are fit per site to turnover and
radiocarbon with the shared calibration
(soil_diskin.calibration.soft_cutoff_powerlaw_spec), each site starting from
its fit at the previous alpha. Sites run in parallel.

All alpha values are written to a single long-format file with one row per
(alpha, site).
"""
import numpy as np
import pandas as pd

from soil_diskin.calibration import calibrate_sweep, soft_cutoff_powerlaw_spec

DATA_PATH = 'results/all_sites_14C_turnover.csv'
OUTPUT_PATH = 'results/03_calibrate_models/soft_cutoff_powerlaw_model_alpha_sweep.csv'
# alpha < 0.2 is excluded: near alpha = 0 (a log-uniform rate distribution) A/T
# grows only logarithmically in t_max/t_min, too slowly to match the observed
# radiocarbon. In a 0.0-1.0 sweep, t_min hit its 1e-8 yr lower bound and only
# 6/99 (alpha = 0.0) and 49/99 (alpha = 0.1) sites fit, vs. 98-99/99 for alpha >= 0.2.
# 0.55-0.75 add resolution where f_new prediction skill peaks (MSRE at 0.5-0.6, KGE at 0.7-0.8).
ALPHA_VALUES = np.sort(np.append(np.round(np.linspace(0.2, 1.0, 9), 1), [0.55, 0.65, 0.75]))

# Snakemake's thread allowance when run by Snakemake, otherwise all cores.
N_JOBS = snakemake.threads if 'snakemake' in globals() else -1  # noqa: F821


def main():
    site_data = pd.read_csv(DATA_PATH)
    print(f'Running soft-cutoff power law alpha sweep over {ALPHA_VALUES.tolist()}...')
    sweep = calibrate_sweep(soft_cutoff_powerlaw_spec, [float(alpha) for alpha in ALPHA_VALUES],
                            site_data, n_jobs=N_JOBS)
    for alpha, fits in sweep.groupby('alpha'):
        print(f'alpha = {alpha:g}: {fits.calibration_status.value_counts().to_dict()}; '
              f'maximum objective {fits.objective.max():.3e}')
    sweep = sweep.join(site_data[['fm_obs', 'turnover']], on='site_index')
    sweep.to_csv(OUTPUT_PATH, index=False)
    print(f'Saved results to {OUTPUT_PATH}')


if __name__ == '__main__':
    main()
