"""Score primary layer fits and plot their labeling-time f_new predictions."""
from pathlib import Path
import numpy as np
import pandas as pd
from permetrics.regression import RegressionMetric


def score(frame: pd.DataFrame) -> dict:
    """Use the entire supplied cohort; failed or missing predictions invalidate selection."""
    observed, predicted = frame[['fnew_obs', 'fnew_pred']].to_numpy(float).T
    status, rmse, kge = 'complete', np.nan, np.nan
    if not len(frame):
        status = 'no_evaluable_observations'
    elif not np.isfinite([observed, predicted]).all():
        status = 'nonfinite_values'
    else:
        rmse = float(np.sqrt(np.mean((predicted-observed)**2)))
        if len(frame) > 1 and min(observed.std(), predicted.std(), observed.mean(), predicted.mean()) > 1e-14:
            kge = float(RegressionMetric(y_true=observed, y_pred=predicted).kling_gupta_efficiency(force_finite=False))
    converged, checked = frame.success.fillna(False).astype(bool), frame.quadrature_ok.fillna(False).astype(bool)
    return {'n_profiles': frame.profile_id.nunique(), 'n_layer_pairs': len(frame),
            'n_locations': len(frame[['latitude', 'longitude']].drop_duplicates()) if 'latitude' in frame else np.nan,
            'rmse': rmse, 'kge_2012': kge, 'evaluation_status': status,
            'converged_layers': int(converged.sum()), 'quadrature_ok_layers': int(checked.sum()),
            'unconverged_layer_pairs': int((~converged).sum()), 'unchecked_layer_pairs': int((~checked).sum()),
            'eligible': bool(status == 'complete' and converged.all() and checked.all()),
            'bound_layers': int((frame.mu_at_bound | frame.sigma_at_bound).sum()),
            'stock_relative_rmse_percent': float(100*np.sqrt(np.mean(((frame.stock_pred_kg_m2-frame.stock_kg_m2)/frame.stock_kg_m2)**2))),
            'radiocarbon_rmse_permil': float(1000*np.sqrt(np.mean((frame.fm_pred-frame.fm_obs)**2)))}


def plot_comparison(layers: pd.DataFrame, output: Path) -> None:
    """Evaluate primary fits with observed f_new; always write metrics and PNG/PDF."""
    import matplotlib.pyplot as plt

    pairs = layers[np.isfinite(layers.fnew_obs) & np.isfinite(layers.duration_years) & layers.duration_years.ge(0)]
    summary = score(pairs)
    pd.DataFrame([summary]).to_csv(output/'metrics.csv', index=False)
    fig, ax = plt.subplots(figsize=(5.6, 5.6), layout='constrained')
    ax.set(xlim=(-.02, 1.02), ylim=(-.02, 1.02), aspect='equal',
           xlabel='Observed new-carbon fraction', ylabel='Predicted new-carbon fraction')
    if pairs.empty:
        ax.set_title('No evaluable new-carbon observations')
        ax.text(.5, .5, 'No observed f_new at labeling times\nRMSE and KGE are undefined',
                transform=ax.transAxes, ha='center', va='center')
    else:
        valid = pairs.success & pairs.quadrature_ok
        ax.scatter(pairs.fnew_obs, pairs.fnew_pred, s=22, alpha=.6,
                   c=np.where(valid, '#2373ac', '#ce6428'))
        if not valid.all():
            ax.text(.04, .04, 'Orange: failed or unchecked', color='#ce6428', transform=ax.transAxes)
        ax.plot([0, 1], [0, 1], '--', color='gray', lw=1)
        setting = pairs.iloc[0]
        ax.set_title(f'No transport · h = {setting.input_depth_cm:g} cm · surface = {setting.surface_fraction:.0%}\n'
                     f'{summary["n_profiles"]} profiles · {len(pairs)} pairs · {setting.soil_npp_fraction:.0%} NPP to soil')
        ax.text(.04, .96, f'RMSE = {summary["rmse"]:.4f}\nKGE (2012) = {summary["kge_2012"]:.3f}',
                transform=ax.transAxes, va='top', bbox={'facecolor': 'white', 'edgecolor': 'lightgray'})
    ax.grid(alpha=.15)
    for extension in ['png', 'pdf']:
        fig.savefig(output/f'fnew_scatter.{extension}', dpi=200)
    plt.close(fig)
