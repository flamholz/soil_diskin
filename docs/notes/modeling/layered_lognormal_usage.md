# Layered log-normal fitting and prediction

The model fits ten independent `(mu, sigma)` pairs per Balesdent profile, conditional
on a supplied shared diffusion coefficient D, downward velocity v, and input depth h.
It predicts carbon stocks, fraction modern, and layer new-carbon fractions. The
[design](layered_lognormal_design.md) records the equations and agreed assumptions.

## Run from the repository root

This small demonstration uses **illustrative, uncalibrated hyperparameters**:

```sh
uv run python -m soil_diskin.layered_workflow \
  --hyper 0.01 0.001 30 \
  --hyper 0 0 30 \
  --limit 2 --starts 4 --seed 0 --max-nfev 500 \
  --times 1 10 100 \
  --output-dir results/layered_example
```

Each `--hyper D V H` adds one specified sensitivity scenario. D is in cm²/yr,
v in cm/yr, h in cm; NPP is converted to kg C/m²/yr internally. Each run needs
a new or empty output directory. Drop `--limit 2` to fit all eligible profiles,
or repeat `--profile 'exact Internal_profile_ID'` to select particular profiles.
Every profile's valid labeling duration is included automatically in prediction times.

The local adapter currently finds **50 eligible profiles at 35 coordinate pairs**.
It uses ten finite positive layer stocks, ten finite native-cell radiocarbon
targets, and finite positive cached NPP. It preserves named profiles at shared
coordinates, reports exclusions, and does not fill missing calibration data.
Observed `f_new` comes from differences of cumulative new-carbon stocks divided
by layer total stocks. It is used only for evaluation at the labeling duration.

Required local inputs already used by the existing workflow:

- `data/balesdent_2018/balesdent_2018_raw.xlsx`
- `data/shi_2020/global_delta_14C.nc`
- `results/all_sites_14C_turnover.csv` (cached NPP)
- `data/14C_atm_annot.csv`

Paths can be overridden with `--balesdent`, `--shi`, `--npp`, and `--atmosphere`.
The file adapter verifies the [published Shi radiocarbon checksum](https://zenodo.org/records/3823612),
because its NetCDF variable misleadingly says `units: year`. For custom data,
use the in-memory `prepare_profiles` API with explicitly known delta-14C units.
The year-2000 reference and atmospheric constant-tail convention follow the
existing model; the Shi NetCDF itself does not identify a reference date.

## Outputs

| File | Contents |
| --- | --- |
| `run.json` | Status, settings, source fingerprints, selection, and completion counts |
| `profiles.csv` | Selected profile/layer observations with units and labeling durations |
| `exclusions.csv` | Ineligible source profiles and reasons |
| `attempts.csv` | Every optimization start, including failures and duplicate solutions |
| `fits.csv` | Distinct candidates, objectives, convergence, rank/bound and quadrature diagnostics |
| `parameters.csv` | Candidate mu/sigma, observed/predicted stocks and radiocarbon, and residuals |
| `predictions.csv` | Candidate layer `f_new` at requested times and each labeling duration |
| `prediction_spread.csv` | Min/max predictions among comparably good, converged, numerically checked candidates |

Result tables include `profile_id`, `hyper_id`, the supplied D/v/h values, and
`candidate_id` where relevant. Candidate 0 has the smallest objective found;
check `success`, `prediction_success`, and `quadrature_ok` before using it.
A converged solver can still have large residuals. Unconverged candidates are
preserved, and other profiles and triples continue. Tables containing no result
rows may be absent. Results are saved incrementally; a hard process termination
can leave `run.json` with status `running` and partial CSVs. Resume by making a
new selected-profile run rather than appending to an existing directory.

The objective sums squared stock residuals divided by 10% of observed stock and
fraction-modern residuals divided by 0.02 (20 per mil). These are configurable
weights, not measured uncertainties. Default parameter bounds are mu in [-15,10]
and sigma in [0.05,5], with no depth smoothing. See `--help` for overrides.
Distinctness uses the maximum parameter difference normalized by its search
range (default tolerance 0.001). A converged candidate is near-best if its
objective is at most `best*(1+near_relative)+near_absolute`, defaulting to 1%
and 1e-6 respectively. Prediction spreads are not confidence intervals.
Choosing a preferred scenario after looking at observed `f_new` uses those
observations for tuning and would require separate evaluation data.

## Python interface

```python
import numpy as np
from soil_diskin.layered_data import load_profiles
from soil_diskin.layered_fitting import FitSettings, fit_profile
from soil_diskin.layered_lognormal import LayeredLognormal
from soil_diskin.radiocarbon_utils import load_atm14c

data = load_profiles()
profile_id = data.profiles.profile_id.iloc[0]
p = data.profiles[data.profiles.profile_id == profile_id].sort_values('layer')
model = LayeredLognormal(0.01, 0.001, 30, load_atm14c())
fit = fit_profile(model, p.stock_kg_m2.to_numpy(), p.fm_obs.to_numpy(),
                  p.npp_kg_m2_yr.iloc[0], settings=FitSettings(n_starts=4, seed=0))
best = fit.best  # Raises if all starts failed before finding an evaluable candidate.
prediction = model.predict(best.mu, best.sigma, p.npp_kg_m2_yr.iloc[0],
                           times=np.array([0., 1., 10., 100.]))
# prediction.stocks and .fm: (10,); prediction.fnew: (4, 10)
```

For direct API use, inspect fit diagnostics and check numerical convergence by
constructing another model with half `log_rate_step`. The batch workflow does
this automatically for every retained candidate. It flags differences larger
than 0.001 of either residual scale or 1e-6 absolute `f_new`.

## Numerical method and checks

Finite-volume transport acts on layer stocks with zero flux at both boundaries.
The input log-rate densities are integrated on a shared uniform log-rate grid,
including the slow tail shifted by `-sigma²`. A positive tridiagonal recurrence
avoids cancellation in the steady-state resolvent when decomposition is much
slower than transport. Historical radiocarbon uses the piecewise-constant
atmosphere and its constant old-age tail. Finite-time source integrals use a
block matrix exponential when subtraction would lose accuracy. All calculations
also support pure advection and zero transport.

Analytic Gaussian-density derivatives supply the fitting Jacobian. Its reported
singular values use parameters scaled by their search ranges, with a relative
rank threshold of 1e-8. This is a local sensitivity diagnostic, not proof of
global identifiability.

```sh
uv run pytest tests/test_layered_lognormal.py tests/test_layered_fitting.py \
  tests/test_layered_data.py tests/test_layered_workflow.py
```
