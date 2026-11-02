"""
Script should be run from the project root directory.

A permutation null for the calibrations: one observation is shuffled across
sites, so that models calibrated to the shuffled data show how much predictive
skill the site-level pairing of turnover and radiocarbon carries.

By default turnover is shuffled, for both datasets: each site keeps its own
radiocarbon (and its identifiers) and gets another site's turnover, so the
shuffled turnover no longer equals the row's stock / NPP.
  * ISRaD (--year-column sampling_year --atm ...): each profile's radiocarbon
    was measured in its own sampling year and is only meaningful with that
    year's atmosphere, which is why radiocarbon, not turnover, stays put. Only
    profiles sampled within the atmospheric record (which can be calibrated)
    take part; the others are left out of the output.
  * Balesdent: every site's radiocarbon refers to the year 2000 (the Shi et al.
    map), so either column can be shuffled (--column fm_obs shuffles radiocarbon
    instead); turnover is shuffled to match ISRaD.

Which column is shuffled matters once predictions are scored against each
site's own observations (e.g. F_new at its labeling duration). Both choices
pair random turnovers with random radiocarbon values, so the fitted parameters
have the same distribution. But shuffling turnover keeps each site's own
radiocarbon, so the null asks whether turnover adds skill beyond radiocarbon,
and shuffling fm_obs asks the reverse. For Balesdent, the lognormal null that
keeps radiocarbon predicts F_new nearly as well as the real fits (RMSE 0.108
against 0.106; 0.118 when radiocarbon is shuffled instead).

Writes the output in long format, one row per (replicate, site): replicate,
site_index (the row in the input table), then the site's columns with the
shuffled column replaced and the original kept as <column>_original. Only
central turnover is calibrated, so the turnover_q05/q95 columns are dropped.

The number of replicates and the seed are in config.yaml under `permutation`.
Replicate r is shuffled with np.random.default_rng([seed, r]), so adding
replicates never changes existing ones.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

INPUT_PATH = 'results/all_sites_14C_turnover.csv'
OUTPUT_PATH = 'results/all_sites_14C_turnover_permuted.csv'


def permute(sites, replicates, seed, column='turnover', include=None):
    """Long-format copies of sites, with `column` shuffled across sites in each replicate.

    include, if given, is a boolean mask of the sites that take part; the others
    are left out. site_index keeps the original row labels.
    """
    if include is not None:
        sites = sites[np.asarray(include, dtype=bool)]
    frames = []
    for replicate in range(replicates):
        rng = np.random.default_rng([seed, replicate])
        frame = sites.drop(columns=['turnover_q05', 'turnover_q95'], errors='ignore')
        frame.insert(0, 'site_index', sites.index)
        frame.insert(0, 'replicate', replicate)
        frame[f'{column}_original'] = sites[column]
        frame[column] = rng.permutation(sites[column].to_numpy())
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def within_record(years, atm_path):
    """True for years no later than the last year of the atmospheric record."""
    return np.asarray(years, dtype=float) <= pd.read_csv(atm_path).year.max()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', default=INPUT_PATH, help='turnover and fm table from step 02')
    parser.add_argument('--output', default=OUTPUT_PATH, help='permuted table')
    parser.add_argument('--column', default='turnover', help='column to shuffle across sites')
    parser.add_argument('--year-column', help='only shuffle among rows whose year in this column is within the '
                                              'atmospheric record (--atm)')
    parser.add_argument('--atm', default='data/14C_atm_annot.csv', help='atmospheric record, for --year-column')
    parser.add_argument('--config', default='config.yaml', help='config with permutation replicates and seed')
    args = parser.parse_args(argv)
    config = yaml.safe_load(Path(args.config).read_text())['permutation']
    sites = pd.read_csv(args.input)
    include = within_record(sites[args.year_column], args.atm) if args.year_column else None
    permuted = permute(sites, config['replicates'], config['seed'], args.column, include)
    unchanged = (permuted[args.column] == permuted[f'{args.column}_original']).groupby(permuted.replicate).sum()
    n_sites = permuted.site_index.nunique()
    print(f"{config['replicates']} replicate(s) of {n_sites} of {len(sites)} sites; "
          f'sites keeping their own {args.column} by chance: {unchanged.tolist()}')
    permuted.to_csv(args.output, index=False)
    print(f'wrote {args.output}')


if __name__ == '__main__':
    main()
