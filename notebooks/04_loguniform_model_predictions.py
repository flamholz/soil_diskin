"""Predict site-level fraction new carbon with the log-uniform model."""

from pathlib import Path

import pandas as pd

from soil_diskin.continuum_models import LogUniformDisKin


def main():
    sites = pd.read_csv('results/processed_balesdent_2018.csv')
    params = pd.read_csv(
        'results/03_calibrate_models/loguniform_model_optimization_results.csv'
    )
    result = sites.copy()
    # The calibration carries no site identifier: its rows follow the processed sites,
    # and each is paired with the site at the same index label (for its labeling
    # duration). Stop if rows were dropped, added or duplicated rather than mispair them.
    if params.index.has_duplicates or set(params.index) != set(sites.index):
        raise ValueError(f'calibration rows ({len(params)}) do not match the {len(sites)} sites')

    for i, row in params.iterrows():
        duration = sites.loc[i, 'Duration_labeling']
        result.loc[i, 'predicted_fnew'] = LogUniformDisKin(
            row.k_min, row.log_width
        ).cdfA(duration)
        for suffix in ('05', '95'):
            k_min = row.get(f'k_min_{suffix}')
            log_width = row.get(f'log_width_{suffix}')
            if pd.notna(k_min) and pd.notna(log_width):
                result.loc[i, f'predicted_fnew_{suffix}'] = LogUniformDisKin(
                    k_min, log_width
                ).cdfA(duration)

    output = Path('results/04_model_predictions/loguniform_model_predictions.csv')
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(f'wrote {output}')


if __name__ == '__main__':
    main()
