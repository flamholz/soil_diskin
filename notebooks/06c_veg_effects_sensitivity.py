"""Vegetation effects sensitivity analysis.

New carbon reaches the soil only after passing through the vegetation, so the
age of soil carbon counted from photosynthesis is its age in the soil plus its
time spent in the vegetation. This script shows how much accounting for this
delay changes the predicted F_new of the calibrated power-law (03a) and
lognormal (03b) models.

The vegetation residence time is taken to be exponential with a mean of
450/50 = 9 years (stocks of 450 GtC turned over by an NPP of 50 GtC/yr), and is
independent of the soil age. The probability that carbon is younger than the
labeling duration L is then

    F_conv(L) = integral_0^L veg(s) cdfA(L - s) ds,

where veg(s) is the exponential density and cdfA the soil age CDF of the
model. The integral is computed with a fixed Gauss-Legendre rule over
s in [0, min(L, 40 * 9 yr)]; the vegetation density beyond 40 mean residence
times is below e^-40. Without the vegetation, F_new is cdfA(L), which equals the
predictions of script 04.

This replaces a calculation that discretized the age distributions on a 0.1-yr
grid and convolved them numerically (on a 1e7-yr grid for the power law, and
from 06a's tabulated CDFs truncated at 1e5 yr for the lognormal). Its results
differed from the direct calculation by up to 6e-3 for the power law (the
rectangle rule of the discrete convolution) and up to 3e-2 for the lognormal
(normalizing by the stock younger than 1e5 yr, which leaves out the 1-5% of
carbon older than that at some sites).

Output: the processed site data with the columns
  pred_conv, pred_no_conv                  power law with / without vegetation
  pred_conv_lognorm, pred_no_conv_lognorm  lognormal with / without vegetation

Run from the project root directory:
    python notebooks/06c_veg_effects_sensitivity.py
"""
import argparse

import numpy as np
import pandas as pd
from numpy.polynomial.legendre import leggauss

from soil_diskin.continuum_models import LognormalDisKin, PowerLawDisKin

SITES_PATH = 'results/processed_balesdent_2018.csv'
POWERLAW_PATH = 'results/03_calibrate_models/powerlaw_model_optimization_results.csv'
LOGNORMAL_PATH = 'results/03_calibrate_models/lognormal_model_optimization_results.csv'
OUTPUT_PATH = 'results/06_sensitivity_analysis/06c_model_predictions_veg_effects.csv'

# Mean residence time of carbon in the vegetation: 450 GtC of stocks over an
# NPP of 50 GtC/yr.
VEG_MEAN = 450 / 50
VEG_RATE = 1 / VEG_MEAN
VEG_S_MAX = 40 * VEG_MEAN

# Gauss-Legendre nodes and weights on [-1, 1] for the vegetation integral.
GL_X, GL_W = leggauss(400)


def veg_convolved_fnew(cdf, duration):
    """Fraction of carbon younger than `duration` counting its time in the vegetation.

    Integrates veg(s) * cdf(duration - s) over the vegetation residence time s,
    where `cdf` is the vectorized soil age CDF of a model.
    """
    upper = min(duration, VEG_S_MAX)
    s = 0.5 * upper * (GL_X + 1)
    veg_pdf = VEG_RATE * np.exp(-VEG_RATE * s)
    return 0.5 * upper * np.sum(GL_W * veg_pdf * cdf(duration - s))


def predict(model, duration):
    """F_new of `model` at `duration` with and without the vegetation delay."""
    return veg_convolved_fnew(model.cdfA, duration), float(model.cdfA(duration))


def main(sites_path=SITES_PATH, powerlaw_path=POWERLAW_PATH,
         lognormal_path=LOGNORMAL_PATH, output_path=OUTPUT_PATH):
    site_data = pd.read_csv(sites_path)
    powerlaw_fits = pd.read_csv(powerlaw_path)
    lognormal_fits = pd.read_csv(lognormal_path)

    for i, duration in site_data['Duration_labeling'].items():
        powerlaw = PowerLawDisKin(powerlaw_fits.loc[i, 't_min'], powerlaw_fits.loc[i, 't_max'])
        site_data.loc[i, ['pred_conv', 'pred_no_conv']] = predict(powerlaw, duration)
        lognormal = LognormalDisKin(lognormal_fits.loc[i, 'mu'], lognormal_fits.loc[i, 'sigma'])
        site_data.loc[i, ['pred_conv_lognorm', 'pred_no_conv_lognorm']] = predict(lognormal, duration)

    site_data.to_csv(output_path, index=False)
    print(f'wrote {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sites', default=SITES_PATH, help='Processed sites, for the labeling durations.')
    parser.add_argument('--powerlaw', default=POWERLAW_PATH, help='Power-law calibration results from script 03a.')
    parser.add_argument('--lognormal', default=LOGNORMAL_PATH, help='Lognormal calibration results from script 03b.')
    parser.add_argument('--output', default=OUTPUT_PATH, help='Output CSV.')
    args = parser.parse_args()
    main(args.sites, args.powerlaw, args.lognormal, args.output)
