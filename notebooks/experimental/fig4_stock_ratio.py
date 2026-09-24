"""Re-score Figure 4 with 1 m Cnew/Ctotal observations; keep predictions fixed.

Run from the repository root: python -m notebooks.experimental.fig4_stock_ratio
Outputs go to results/fig4_stock_ratio; the normal Figure 4 is not overwritten.
"""
from pathlib import Path
import hashlib
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from permetrics.regression import RegressionMetric

from notebooks.viz import color_palette
from soil_diskin.data_wrangling import process_balesdent_data

OUT = Path('results/fig4_stock_ratio')
OUT.mkdir(parents=True, exist_ok=True)
PRED = Path('results/04_model_predictions')
RAW = Path('data/balesdent_2018/balesdent_2018_raw.xlsx')
SITES = Path('results/processed_balesdent_2018.csv')
KEYS = ['Latitude', 'Longitude', 'Duration_labeling']
sources = [RAW, SITES, Path(__file__), Path('notebooks/fig4.py'),
           Path('notebooks/viz.py'), Path('notebooks/style.mpl'),
           Path('soil_diskin/data_wrangling.py')]
sites = pd.read_csv(SITES)
raw = pd.read_excel(RAW, sheet_name='Profiles', skiprows=7)
observations = sites[KEYS + ['total_fnew']].rename(columns={'total_fnew': 'original'})
# Match Excel/CSV coordinates despite floating-point serialization differences.
observations[KEYS] = observations[KEYS].round(10)
raw[KEYS] = raw[KEYS].round(10)
assert not observations.duplicated(KEYS).any()
coverage = {}
for name, suffix in [('direct', ''), ('estimated', 'estim')]:
    total = raw[f'Ctotal_0-100{suffix}']
    ratio = raw[f'Cnew_0-100{suffix}'] / total
    valid = np.isfinite(total) & total.gt(0) & np.isfinite(ratio) & ratio.between(0, 1)
    # Preserve the original arithmetic mean across replicate profiles at a site.
    grouped = raw.loc[valid, KEYS].assign(**{name: ratio[valid]}).groupby(KEYS).mean()
    observations = observations.merge(grouped, on=KEYS, how='left', validate='one_to_one')
    assert observations[name].notna().sum() == len(grouped)
    endpoints = process_balesdent_data(raw.loc[valid].reset_index(drop=True))
    endpoints = endpoints[KEYS + ['total_fnew']].rename(
        columns={'total_fnew': f'{name}_same_profiles'})
    observations = observations.merge(endpoints, on=KEYS, how='left', validate='one_to_one')
    coverage[name] = {'raw_profiles': int(valid.sum()), 'bulk_records': len(grouped)}
np.testing.assert_allclose(observations[KEYS], sites[KEYS], atol=1e-10, rtol=0)
mask = observations.estimated.notna()
np.testing.assert_allclose(observations.loc[mask, 'original'],
                           observations.loc[mask, 'estimated_same_profiles'])
observations.to_csv(OUT / 'observations.csv', index=False)

pal = color_palette()
models = []
for filename, title, color in [
    ('lognormal_model_predictions.csv', 'lognormal model', 'dark_blue'),
    ('power_law_model_predictions.csv', r'power law model ($\alpha = 1$)', 'blue'),
    ('general_power_law_model_predictions.csv', r'power law model ($\alpha = e^{-\gamma}$)', 'light_blue'),
]:
    sources.append(PRED / filename)
    frame = pd.read_csv(PRED / filename)
    pd.testing.assert_frame_equal(frame[sites.columns], sites)
    prediction = frame.predicted_fnew.to_numpy()
    error = frame[['predicted_fnew_05', 'predicted_fnew_95']].sub(
        frame.predicted_fnew, axis=0).abs().fillna(0).to_numpy().T
    models.append((title, prediction, error, pal[color]))
for filename, title, color in [('CLM45_fnew.csv', 'CLM4.5', 'dark_purple'),
                               ('JSBACH_fnew.csv', 'JSBACH', 'purple')]:
    sources.append(PRED / filename)
    models.append((title, pd.read_csv(PRED / filename, header=None)[0].to_numpy(), None, pal[color]))
sources.append(PRED / 'RCM.csv')
rcm = pd.read_csv(PRED / 'RCM.csv')
assert list(rcm.columns) == ['CESM1', 'IPSL-CM5A-LR', 'MRI-ESM1']
for title, color in zip(rcm, ['dark_green', 'green', 'light_green']):
    models.append((title + r' ($^{14} C$ corrected)', rcm[title].to_numpy(), None, pal[color]))
# ESM/RCM files have no site keys: retain the row alignment used by Figure 4.
assert all(len(prediction) == len(sites) and np.isfinite(prediction).all()
           for _, prediction, _, _ in models)
plt.style.use('notebooks/style.mpl')


def scores(observed, predicted):
    """Use the same KGE (2012 default) and RMSE as notebooks/fig4.py."""
    return {'n': len(observed), 'rmse': float(np.sqrt(np.mean((observed - predicted) ** 2))),
            'kge': float(RegressionMetric(y_true=observed, y_pred=predicted)
                         .kling_gupta_efficiency(force_finite=False))}


metrics = []
for cohort, selector in [('all', pd.Series(True, index=observations.index)),
                         ('direct', observations.direct.notna()),
                         ('estimated', observations.estimated.notna())]:
    columns = ['original'] if cohort == 'all' else ['original', cohort, f'{cohort}_same_profiles']
    for column in columns:
        x = observations.loc[selector, column].to_numpy()
        assert np.isfinite(x).all()
        # The extra same-profile baseline checks the two mixed-coverage direct groups.
        plot = not column.endswith('_same_profiles')
        if plot:
            fig, axes = plt.subplots(2, 4, figsize=(7.24, 3.5), dpi=300,
                                     constrained_layout=True, sharex=True, sharey=True)
            axes = axes.flatten()
        for i, (title, prediction, error, color) in enumerate(models):
            y = prediction[selector]
            result = scores(x, y)
            metrics.append({'cohort': cohort, 'observation': column, 'model': title, **result})
            if not plot:
                continue
            ax = axes[i]
            ax.plot([0, 1], [0, 1], color='grey', linestyle='--', zorder=-10, lw=1)
            if error is None:
                ax.scatter(x, y, color=color, edgecolor='k', lw=0.5, s=20, alpha=0.9)
            else:
                ax.errorbar(x, y, yerr=error[:, selector], fmt='o', color=color,
                            ecolor='k', elinewidth=0.5, capsize=2, mec='k', mew=0.5,
                            markersize=5, alpha=0.9)
            ax.text(0.05, 0.95, f'KGE = {result["kge"]:.2f}\nRMSE = {result["rmse"]:.2f}',
                    transform=ax.transAxes, fontsize=6, va='top',
                    bbox=dict(boxstyle='round', facecolor=pal['light_yellow'],
                              edgecolor=pal['dark_grey'], alpha=0.8))
            ax.set(title=title, xticks=[0, .5, 1], yticks=[0, .5, 1])
            ax.text(-0.25 if i in (0, 4) else -0.15, 1.1, 'ABCDEFGH'[i],
                    transform=ax.transAxes, fontsize=7, va='top', ha='left')
        if plot:
            for ax in axes[4:]:
                ax.set_xlabel(r'observed F$_{new}$ ($\delta^{13}C$ based)')
            for ax in axes[[0, 4]]:
                ax.set_ylabel('predicted F$_{new}$')
            label = {'original': 'Original observations', 'direct': 'Cnew / Ctotal, direct 0–100 cm',
                     'estimated': 'Cnew / Ctotal, estimated 0–100 cm'}[column]
            fig.suptitle(f'{label} · {len(x)} records', fontsize=9)
            for ext in ['png', 'pdf']:
                fig.savefig(OUT / f'fig4_{cohort}_{column}.{ext}', dpi=300, bbox_inches='tight')
            plt.close(fig)

metrics = pd.DataFrame(metrics)
metrics.to_csv(OUT / 'metrics.csv', index=False)
changes = metrics[metrics.observation.isin(['direct', 'estimated'])].merge(
    metrics[metrics.observation == 'original'], on=['cohort', 'model', 'n'],
    suffixes=('_ratio', '_original'), validate='one_to_one')
for metric in ['rmse', 'kge']:
    changes[f'delta_{metric}'] = changes[f'{metric}_ratio'] - changes[f'{metric}_original']
changes.to_csv(OUT / 'metric_changes.csv', index=False)
(OUT / 'summary.json').write_text(json.dumps({
    'coverage': coverage, 'original_records': len(sites), 'refitted': False,
    'grouping': 'Per-profile ratios averaged by rounded coordinates and labeling duration',
    'prediction_alignment': 'Verified continuum keys; ESM/RCM retain original Figure 4 row order',
    'sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
}, indent=2) + '\n')
print(changes.to_string(index=False))
