"""One RMSE/KGE definition for workflow plots and fixed-cohort experiments."""
from pathlib import Path

import numpy as np
import pandas as pd
from permetrics.regression import RegressionMetric


def prediction_metrics(observed, predicted) -> dict:
    observed, predicted = np.asarray(observed), np.asarray(predicted)
    if observed.shape != predicted.shape or observed.ndim != 1:
        raise ValueError('observations and predictions must be matching one-dimensional arrays')
    status = 'complete'
    rmse = kge = np.nan
    if not len(observed):
        status = 'no_evaluable_observations'
    elif not (np.isfinite(observed).all() and np.isfinite(predicted).all()):
        status = 'nonfinite_values'
    else:
        rmse = float(np.sqrt(np.mean((predicted-observed)**2)))
        if len(observed) > 1 and min(observed.std(), predicted.std(), observed.mean(), predicted.mean()) > 1e-14:
            kge = float(RegressionMetric(y_true=observed, y_pred=predicted)
                        .kling_gupta_efficiency(force_finite=False))
    return {'rmse': rmse, 'kge_2012': kge, 'evaluation_status': status}


def score(frame: pd.DataFrame) -> dict:
    """Score every expected layer; failed fits invalidate a candidate, never shrink it."""
    agreement = prediction_metrics(frame.fnew_obs.to_numpy(), frame.fnew_pred.to_numpy())
    return {'n_profiles': int(frame.profile_id.nunique()), 'n_layer_pairs': len(frame),
            'n_locations': len(frame[['latitude', 'longitude']].drop_duplicates()), **agreement,
            'converged_layers': int(frame.success.fillna(False).sum()),
            'quadrature_ok_layers': int(frame.quadrature_ok.fillna(False).sum()),
            'eligible': bool(agreement['evaluation_status'] == 'complete' and frame.success.fillna(False).all()
                             and frame.quadrature_ok.fillna(False).all()),
            'bound_layers': int((frame.mu_at_bound | frame.sigma_at_bound).sum()),
            'stock_relative_rmse_percent': float(100*np.sqrt(np.mean(
                ((frame.stock_pred_kg_m2-frame.stock_kg_m2)/frame.stock_kg_m2)**2))),
            'radiocarbon_rmse_permil': float(1000*np.sqrt(np.mean((frame.fm_pred-frame.fm_obs)**2)))}


def plot_comparison(predictions: pd.DataFrame, output: Path) -> None:
    """Always write an evaluation record and figure, including when no targets exist."""
    import matplotlib.pyplot as plt

    if predictions.empty:
        pairs = pd.DataFrame(columns=['profile_id', 'fnew_obs', 'fnew_pred', 'success', 'quadrature_ok'])
    else:
        pairs = predictions[(predictions.candidate_id == 0) & predictions.at_label_duration
                            & np.isfinite(predictions.fnew_obs)]
    agreement = prediction_metrics(pairs.fnew_obs.to_numpy(dtype=float), pairs.fnew_pred.to_numpy(dtype=float))
    summary = {'n_profiles': pairs.profile_id.nunique(), 'n_layer_pairs': len(pairs), **agreement,
               'unconverged_layer_pairs': int((~pairs.success.astype(bool)).sum()),
               'unchecked_layer_pairs': int((~pairs.quadrature_ok.astype(bool)).sum())}
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
        for mask, color, marker, label in [(valid, '#2373ac', 'o', 'Converged and checked'),
                                           (~valid, '#ce6428', 'x', 'Failed or unchecked')]:
            frame = pairs[mask & np.isfinite(pairs.fnew_pred)]
            if len(frame):
                ax.scatter(frame.fnew_obs, frame.fnew_pred, s=22, c=color, marker=marker, alpha=.6, label=label)
        ax.plot([0, 1], [0, 1], '--', color='gray', lw=1)
        surface = pairs.surface_fraction.iloc[0] if 'surface_fraction' in pairs else 0.
        soil = pairs.soil_npp_fraction.iloc[0] if 'soil_npp_fraction' in pairs else 1.
        ax.set_title(f'No transport · h = {pairs.input_depth_cm.iloc[0]:g} cm · surface = {surface:.0%}\n'
                     f'{summary["n_profiles"]} profiles · {len(pairs)} pairs · {soil:.0%} NPP to soil')
        ax.text(.04, .96, f'RMSE = {agreement["rmse"]:.4f}\nKGE (2012) = {agreement["kge_2012"]:.3f}',
                transform=ax.transAxes, va='top', bbox={'facecolor': 'white', 'edgecolor': 'lightgray'})
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc='lower right')
    ax.grid(alpha=.15)
    for extension in ['png', 'pdf']:
        fig.savefig(output/f'fnew_scatter.{extension}', dpi=200)
    plt.close(fig)
