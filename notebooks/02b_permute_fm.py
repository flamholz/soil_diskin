"""
Script should be run from the project root directory.

Permutes the observed fm across sites, a null for the calibrations: each site
keeps its turnover but gets another site's radiocarbon, so models calibrated
to the permuted data show how much predictive skill the site-level
radiocarbon carries.

Writes results/all_sites_14C_turnover_permuted.csv in long format, one row per
(replicate, site): replicate, site_index (the row in all_sites_14C_turnover.csv),
then the site's columns with fm_obs shuffled across all sites and the original
kept as fm_obs_original. Only central turnover is calibrated, so the
turnover_q05/q95 columns are dropped.

The number of replicates and the seed are in config.yaml under `permutation`.
Replicate r is shuffled with np.random.default_rng([seed, r]), so adding
replicates never changes existing ones.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

INPUT_PATH = 'results/all_sites_14C_turnover.csv'
OUTPUT_PATH = 'results/all_sites_14C_turnover_permuted.csv'


def permute_fm(sites, replicates, seed):
    """Long-format copies of sites, with fm_obs shuffled across sites in each replicate."""
    frames = []
    for replicate in range(replicates):
        rng = np.random.default_rng([seed, replicate])
        frame = sites.drop(columns=['turnover_q05', 'turnover_q95'], errors='ignore')
        frame.insert(0, 'site_index', sites.index)
        frame.insert(0, 'replicate', replicate)
        frame['fm_obs_original'] = sites['fm_obs']
        frame['fm_obs'] = rng.permutation(sites['fm_obs'].to_numpy())
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def main():
    config = yaml.safe_load(Path('config.yaml').read_text())['permutation']
    sites = pd.read_csv(INPUT_PATH)
    permuted = permute_fm(sites, config['replicates'], config['seed'])
    unchanged = (permuted.fm_obs == permuted.fm_obs_original).groupby(permuted.replicate).sum()
    print(f"{config['replicates']} replicate(s) of {len(sites)} sites; "
          f'sites keeping their own fm by chance: {unchanged.tolist()}')
    permuted.to_csv(OUTPUT_PATH, index=False)
    print(f'wrote {OUTPUT_PATH}')


if __name__ == '__main__':
    main()
