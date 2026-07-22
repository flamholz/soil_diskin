import os
import shutil
import unittest
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray  # noqa: F401  (registers the .rio accessor used below)
from collections import namedtuple
from scipy.integrate import solve_ivp
from soil_diskin.age_dist_utils import calc_age_dist_cdf
from soil_diskin.compartmental_models import JSBACH
from soil_diskin.constants import DAYS_PER_YEAR, SECS_PER_DAY, T_MELT
import subprocess

JSBACH_FORCING_DIR = 'data/model_params/JSBACH'
JSBACH_FORCING_FILES = [f'{JSBACH_FORCING_DIR}/JSBACH_S3_{v}.nc'
                        for v in ('tas', 'pr', 'npp')]

# The Fortran reference is the unmodified JSBACH yasso routine, built on demand.
# Neither the sources nor the compiler are in git, so the comparison is skipped
# where they are unavailable.
JSBACH_FORTRAN_DIR = 'tests/test_data/jsbach'
JSBACH_FORTRAN_RUNNER = f'{JSBACH_FORTRAN_DIR}/testing_src/run_yasso_test.sh'
JSBACH_FORTRAN_OUTPUT = f'{JSBACH_FORTRAN_DIR}/yasso_output.csv'

# days_per_year in mo_carbon_constants.f90; yasso steps with 1/365.25 of a year,
# which differs from the 365 that DAYS_PER_YEAR uses elsewhere in this project.
JSBACH_DAYS_PER_YEAR = 365.25

class TestJSBACH(unittest.TestCase):
    """Test suite for JSBACH compartmental model."""

    def setUp(self):
        """Set up minimal config and environment data for testing."""
        JSBACHConfig = namedtuple('JSBACHConfig', ['placeholder'])
        self.config = JSBACHConfig(placeholder=True)
        
        JSBACHEnv = namedtuple('JSBACHEnv', ['I', 'T', 'P', 'd'])
        self.env_params = JSBACHEnv(
            I=np.ones(12),
            T=np.random.rand(12) * 20 + 10,  # Temperature 10-30°C
            P=np.random.rand(12) * 100 + 50,  # Precipitation 50-150mm
            d=0.1  # CWD diameter
        )

    def test_initialization(self):
        """Test that JSBACH can be instantiated."""
        model = JSBACH(self.config, self.env_params)
        self.assertIsInstance(model, JSBACH)
        self.assertIsNotNone(model.A)
        self.assertIsNotNone(model.K)
        self.assertIsNotNone(model.u)

    @unittest.skipUnless(
        os.path.exists(JSBACH_FORTRAN_RUNNER)
        and shutil.which(os.environ.get('FC', 'gfortran')),
        f"needs {JSBACH_FORTRAN_RUNNER} and a Fortran compiler"
    )
    def test_comparison_with_fortran(self):
        """Test JSBACH output against Fortran implementation.

        Both sides take one daily Euler step from pools of 1 mol(C)/m2 under a
        constant 25 C / 1 m-per-year climate, for the non-woody litter pools.

        The step uses JSBACH's own year length rather than DAYS_PER_YEAR, so
        that the two discretisations are identical and any difference is a real
        difference in the model. Under that setup the implementations agree to
        ~2e-16; at DAYS_PER_YEAR = 365 the year-length mismatch alone costs
        3e-5, which is coarse enough to hide a 1% error in a transfer
        coefficient.
        """
        # Parameters from JSBACH documentation in
        # from https://pure.mpg.de/rest/items/item_3279802_26/component/file_3316522/content#page=107.51 and 
        # https://gitlab.dkrz.de/icon/icon-model/-/blob/release-2024.10-public/externals/jsbach/src/carbon/mo_carbon_process.f90
        a_i = np.array([0.72, 5.9, 0.28, 0.031])
        a_h = 0.0016
        b1 = 9.5e-2
        b2 = -1.4e-3
        gamma = -1.21
        phi1 = -1.71
        phi2 = 0.86
        r = -0.306

        # Create JSBACH config
        JSBACH_config = namedtuple('Config', ['a_i', 'a_h', 'b1', 'b2', 'gamma', 'phi1', 'phi2', 'r'])
        config = JSBACH_config(a_i = a_i, a_h = a_h, b1 = b1, b2 = b2, gamma = gamma, phi1 = phi1, phi2 = phi2, r = r)
        
        # Create environment parameters. The annual litter input is one day's
        # worth per day, so that the Euler step below sees an input of 1.
        one_vec = np.ones(12)
        JSBACH_env_params = namedtuple('EnvParams', ['I', 'T', 'P', 'd'])
        env_params = JSBACH_env_params(one_vec * JSBACH_DAYS_PER_YEAR, 25 * one_vec, one_vec, 4)

        # Instantiate model and compute output
        model = JSBACH(config=config, env_params=env_params)
        output = model._dX(t=0, X=np.ones(18))[:9]

        # Compile and run the unmodified Fortran yasso routine; the driver that
        # sets up this same experiment is in testing_src/test_yasso_call.f90.
        subprocess.run([JSBACH_FORTRAN_RUNNER], check=True,
                       capture_output=True, text=True)

        # Load Fortran output
        fortran_output = pd.read_csv(JSBACH_FORTRAN_OUTPUT)

        # Assert outputs are almost equal
        expected = output * (1/JSBACH_DAYS_PER_YEAR) + np.ones(9)
        actual = fortran_output['Value'].values[:9]
        for i in range(len(expected)):
            self.assertAlmostEqual(actual[i], expected[i], places=12)

def _load_jsbach_forcing(path):
    """Monthly-mean JSBACH forcing field, gap-filled, as the prediction script does."""
    da = xr.open_dataarray(path).groupby('time.month').mean()
    da = da.rio.write_crs('EPSG:4326')
    da = da.rio.write_nodata(np.nan)
    return da.rio.interpolate_na()


@unittest.skipUnless(
    all(os.path.exists(f) for f in JSBACH_FORCING_FILES),
    f"JSBACH forcing not found in {JSBACH_FORCING_DIR}/ (downloaded, not in git)"
)
class TestJSBACHFnewAccuracy(unittest.TestCase):
    """Check the analytical F_new for JSBACH against a tracer simulation.

    JSBACH's state equation is linear, dX/dt = I(t) u + (A K(t)) X, so its
    steady-state age CDF follows the same analytical form used for CLM4.5:

        F_new(a) = 1 - 1^T expm(M a) eta,   M = A K_bar,  eta = X_ss / sum(X_ss)

    evaluated by soil_diskin.age_dist_utils.calc_age_dist_cdf, which derives
    X_ss = -M^-1 I_bar internally.

    Ground truth is a labeled/unlabeled tracer pair in the style of
    notebooks/clm_tracer_test.ipynb. Two versions are compared, which
    separate the two error sources:

      * against a tracer on the same annual-mean operator -> tests the
        analytical solution itself (agreement ~1e-11);
      * against a tracer driven by the model's own _dX, i.e. the real
        monthly-varying forcing -> additionally absorbs the cost of replacing
        that forcing with its annual mean (2.3e-3 at 5 yr growing to 1.5e-2 at
        200 yr; see test_analytical_matches_monthly_forced_tracer).

    Ages are capped at 200 yr because JSBACH's slowest mode has an e-folding
    time of only ~167 yr, so F_new saturates at 1.0 well before then; beyond
    ~1000 yr the comparison would be trivially satisfied. 90 of the 99
    Balesdent sites have labeling durations within this informative range.
    """

    # Same grid cell as the CLM4.5 F_new test (Balesdent site 19, Amazon).
    LAT, LON = -7.516667, -63.033333
    AGES = np.array([5., 10., 25., 50., 100., 200.])
    SPINUP_TAU_MULTIPLE = 15

    @classmethod
    def setUpClass(cls):
        tas, pr, npp = (_load_jsbach_forcing(f) for f in JSBACH_FORCING_FILES)
        sel = dict(longitude=cls.LON, latitude=cls.LAT, method='nearest')

        # Same unit conversions as notebooks/04_JSBACH_model_predictions.py.
        I = npp.sel(**sel).values * SECS_PER_DAY * DAYS_PER_YEAR * 1000  # gC/m2/yr
        T = tas.sel(**sel).values - T_MELT                               # K -> C
        P = pr.sel(**sel).values * SECS_PER_DAY * DAYS_PER_YEAR / 1000   # -> m/yr

        config = namedtuple('Config', ['a_i'])(np.array([0.72, 5.9, 0.28, 0.031]))
        EnvParams = namedtuple('EnvParams', ['I', 'T', 'P', 'd'])
        cls.model = JSBACH(config, EnvParams(I, T, P, 4))
        # Zeroed input drives the unlabeled pool, which only decays.
        cls.model_no_input = JSBACH(config, EnvParams(np.zeros_like(I), T, P, 4))

        cls.M = cls.model.A @ cls.model.K.mean(axis=0)
        cls.I_bar = np.asarray(cls.model.I).mean() * cls.model.u
        cls.X_ss = np.linalg.solve(-cls.M, cls.I_bar)
        n = cls.M.shape[0]

        real_eigs = np.linalg.eigvals(cls.M).real
        cls.tau_slow = -1.0 / real_eigs[real_eigs < 0].max()
        t_spin = cls.SPINUP_TAU_MULTIPLE * cls.tau_slow

        cls.fnew_analytical = np.asarray(
            calc_age_dist_cdf(cls.M, cls.I_bar, cls.AGES)
        ).reshape(-1)

        # (a) tracer on the annual-mean operator
        lab = solve_ivp(lambda t, X: cls.I_bar + cls.M @ X, (0, cls.AGES[-1]),
                        np.zeros(n), method='LSODA', rtol=1e-10, atol=1e-12,
                        t_eval=cls.AGES)
        unl = solve_ivp(lambda t, X: cls.M @ X, (0, cls.AGES[-1]), cls.X_ss,
                        method='LSODA', rtol=1e-10, atol=1e-12, t_eval=cls.AGES)
        cls.fnew_mean_op = lab.y.sum(axis=0) / (lab.y.sum(axis=0) + unl.y.sum(axis=0))

        # (b) tracer on the model's own monthly-varying forcing. t_eval keeps
        # only the requested times; without it solve_ivp retains every internal
        # step, and _dX changes its forcing 144 times a year.
        spin = solve_ivp(cls.model._dX, (0, t_spin), np.zeros(n),
                         method='LSODA', t_eval=[t_spin])
        cls.X_spun = spin.y[:, -1]
        lab_m = solve_ivp(cls.model._dX, (0, cls.AGES[-1]), np.zeros(n),
                          method='LSODA', t_eval=cls.AGES)
        unl_m = solve_ivp(cls.model_no_input._dX, (0, cls.AGES[-1]), cls.X_spun,
                          method='LSODA', t_eval=cls.AGES)
        cls.labeled_C = lab_m.y.sum(axis=0)
        cls.unlabeled_C = unl_m.y.sum(axis=0)
        cls.fnew_monthly = cls.labeled_C / (cls.labeled_C + cls.unlabeled_C)

    def test_fnew_is_informative_over_the_tested_ages(self):
        """Guard against a vacuous comparison: F_new must not be pinned at 0 or 1."""
        self.assertLess(self.fnew_analytical[0], 0.5)
        self.assertGreater(self.fnew_analytical[-1], 0.5)
        self.assertLess(self.fnew_analytical[-1], 1.0)

    def test_analytical_matches_mean_operator_tracer(self):
        """Analytical CDF reproduces a tracer run on the same operator."""
        np.testing.assert_allclose(self.fnew_analytical, self.fnew_mean_op,
                                   rtol=1e-6)

    def test_analytical_matches_monthly_forced_tracer(self):
        """Analytical CDF survives the model's real monthly-varying forcing.

        The tolerance is set by the annual-mean approximation, not by numerical
        precision, and it is looser than the CLM4.5 equivalent for a real
        reason. X_ss = -(A K_bar)^-1 I_bar averages K before inverting, which is
        not the same as averaging the inverse, and JSBACH's climate modifier
        k_clim = exp(b1 T + b2 T^2) (1 - exp(gamma P)) swings strongly over the
        seasonal cycle. The resulting bias grows with age: measured 2.3e-3 at
        5 yr, 9.2e-3 at 25 yr and 1.5e-2 at 200 yr, versus ~4e-5 for CLM4.5 at
        the same grid cell. 2.5e-2 leaves roughly 1.7x headroom over the worst
        age tested.
        """
        np.testing.assert_allclose(self.fnew_analytical, self.fnew_monthly,
                                   atol=2.5e-2)

    def test_spinup_total_carbon_matches_analytical_steady_state(self):
        """Monthly-forced spin-up lands on sum(X_ss), the F_new denominator.

        The monthly-forced system settles onto a periodic orbit rather than a
        fixed point; total carbon circles sum(X_ss) with a measured amplitude of
        0.30%, and the spun-up state is one phase of that orbit (0.32% off).
        """
        np.testing.assert_allclose(self.X_spun.sum(), self.X_ss.sum(), rtol=2e-2)

    def test_total_carbon_is_conserved_during_labeling(self):
        """labeled + unlabeled stays near steady state, so F_new's denominator holds.

        Each age samples a different phase of the periodic orbit, so the totals
        spread by up to 1.2% rather than being exactly equal.
        """
        total = self.labeled_C + self.unlabeled_C
        np.testing.assert_allclose(total, self.X_ss.sum(), rtol=2e-2)
