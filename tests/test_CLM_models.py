from glob import glob
from os import path
import os
import unittest

import numpy as np
import pandas as pd
import xarray as xr

from scipy.io import loadmat
from scipy.integrate import solve_ivp
from soil_diskin.age_dist_utils import calc_age_dist_cdf
from soil_diskin.compartmental_models import CLM5
from soil_diskin.compartmental_models import ConfigParams, GlobalData
from soil_diskin.constants import DAYS_PER_YEAR, SECS_PER_DAY
from itertools import product
from joblib import Parallel, delayed
from tqdm import tqdm

# Helper function to run the model for a single grid cell
# Defined at module level to be picklable for joblib.Parallel
def run_model_for_cell(lat, lon, config, global_data, empty_gridcell):
    """Run CLM5 model for a single grid cell."""
    ldd = global_data.make_ldd(lat, lon)
    if np.isnan(ldd.w[0,0]):
        res = empty_gridcell.copy()
        res['y'] = lat
        res['x'] = lon
        return res.expand_dims('y').expand_dims('x').stack(cell=('y', 'x'))
    CLM_model = CLM5(config, ldd)
    CLM_model.I = CLM_model.I / DAYS_PER_YEAR / SECS_PER_DAY  # Convert inputs to per second
    res = CLM_model.run(timesteps=range(11), dt= SECS_PER_DAY * 30, tres='M').expand_dims('y').expand_dims('x').stack(cell=('y', 'x'))
    return res


# Helper function to create CLM5 config and global data
# TODO: the configuration and parameterization of this class is 
# quite obtuse as evidenced by my copying this function from 
# from the script that runs CLM5. Simplify and better document
# the configuration and parameterization of this class.
def make_CLM_config():
    fn = 'data/CLM5_global_simulation/soildepth.mat'
    mat = loadmat(fn)
    zisoi = mat['zisoi'].squeeze()
    zsoi = mat['zsoi'].squeeze()
    dz = mat['dz'].squeeze()
    dz_node = mat['dz_node'].squeeze()

    # load gridded nc file with the inputs, initial values, and the environmental variables
    global_da = xr.open_dataset('data/CLM5_global_simulation/global_demo_in.nc')
    global_da = global_da.rename({'LON':'x','LAT':'y'})
    # def fix_lon(ds):
    #     ds['x'] = xr.where(ds['x']>=180,ds['x']-360,ds['x'])
    #     return ds.sortby('x')

    # global_da = fix_lon(global_da)
    global_da = global_da.rio.write_crs("EPSG:4326", inplace=True)

    # define model parameters
    CLM_params = xr.open_dataset('data/CLM5_global_simulation/clm5_params.c171117.nc')
    taus = np.array([CLM_params['tau_cwd'],
                    CLM_params['tau_l1'],
                    CLM_params['tau_l2_l3'],
                    CLM_params['tau_l2_l3'],
                    CLM_params['tau_s1'],
                    CLM_params['tau_s2'],
                    CLM_params['tau_s3']]).squeeze()
    
    taus = taus * DAYS_PER_YEAR * SECS_PER_DAY # Rates are given per year, we convert to per second
    Gamma_soil = 1e-4 / (SECS_PER_DAY * DAYS_PER_YEAR) #TODO: test this
    F_soil = 0

    # create global configuration parameters
    config = ConfigParams(decomp_depth_efolding=0.5, taus=taus, Gamma_soil=Gamma_soil, F_soil=F_soil,
                        zsoi=zsoi, zisoi=zisoi, dz=dz, dz_node=dz_node, nlevels=10, npools=7)
    global_data = GlobalData(global_da)

    return config, global_data


class TestCLM5(unittest.TestCase):
    """Test suite for CLM5 compartmental model."""

    def setUp(self):
        """Set up config and environment data for testing."""
        self.config, self.global_data = make_CLM_config()
        lat, lng = -26.58333333, 151.83333333333334
        self.env_params = self.global_data.make_ldd(lat, lng)

    def test_initialization(self):
        """Test that CLM4.5 can be instantiated."""
        model = CLM5(self.config, self.env_params)
        self.assertIsInstance(model, CLM5)
        self.assertIsNotNone(model.A)
        self.assertIsNotNone(model.V)
        self.assertIsNotNone(model.K_ts)

    def test_tri_diag_matrix(self):
        """Test CLM4.5 tri-diagonal matrix construction against gold standard examples and failing examples."""
        
        SOILDEPTH_FILE = path.join('tests/test_data/CLM45/tridiag_positive_examples/soildepth.mat')
        GOLD_EXAMPLE_FILES = glob(path.join('tests/test_data/CLM45/tridiag_positive_examples/', 'test_example*.mat'))
        FAILING_EXAMPLE_FILES = glob(path.join('tests/test_data/CLM45/tridiag_negative_examples/', 'test_example*.mat'))
        
        # test that we have some example files
        self.assertGreater(len(GOLD_EXAMPLE_FILES), 0)

        def test_file(fname):
            """Test a single file."""
            # load example data
            example_data = loadmat(fname)

            # get example parameters and create the V matrix
            Gamma_soil, F_soil, npools, nlevels = example_data['example'].squeeze()
            result = CLM5.make_V_matrix(Gamma_soil, F_soil, int(npools), int(nlevels),
                    self.config.dz, self.config.dz_node, self.config.zsoi, self.config.zisoi)

            # get the expected result from the matlab code
            expected = example_data['result'].squeeze()
            
            return result, expected
        
        # run test for gold standard examples
        for fname in GOLD_EXAMPLE_FILES:
            result, expected = test_file(fname)
            np.testing.assert_allclose(result, expected)

        # run test for failing examples
        for fname in FAILING_EXAMPLE_FILES:
            result, expected = test_file(fname)
            np.testing.assert_raises(AssertionError, np.testing.assert_allclose, result, expected)
    

    def test_A_matrix(self):
        """Test CLM4.5 A matrix construction against gold standard examples."""
        
        GOLD_EXAMPLE_FILES = glob(path.join('tests/test_data/CLM45/A_matrix_positive_examples/', 'test_*.mat'))
        
        # test that we have some example files
        self.assertGreater(len(GOLD_EXAMPLE_FILES), 0)

        for fname in GOLD_EXAMPLE_FILES:
            # load example data
            example_data = loadmat(fname)

            # get example parameters and create the A matrix
            sand_content = example_data['sand_content_vec'].squeeze()
            result = CLM5.make_A_matrix(sand_content, self.config.nlevels)

            # get the expected result from the matlab code
            expected = example_data['result'].squeeze()
            
            np.testing.assert_allclose(result, expected)

    def test_K_matrix(self):
        """Test CLM4.5 K matrix construction against gold standard examples."""
        
        SOILDEPTH_FILE = path.join('tests/test_data/CLM45/K_matrix_positive_examples/soildepth.mat')
        GOLD_EXAMPLE_FILES = glob(path.join('tests/test_data/CLM45/K_matrix_positive_examples/', 'test_*.mat'))
        
        # test that we have some example files
        self.assertGreater(len(GOLD_EXAMPLE_FILES), 0)

        for fname in GOLD_EXAMPLE_FILES:
            # load example data
            example_data = loadmat(fname)

            # get example parameters and create the K matrix
            inputs = example_data['example'].squeeze()
            decomp_depth_efolding = example_data['decomp_depth_efolding'].squeeze()
            
            w_scalar = inputs[0,:]
            t_scalar = inputs[1,:]
            o_scalar = inputs[2,:]
            n_scalar = inputs[3,:]

            # taus = self.config.taus * DAYS_PER_YEAR * SECS_PER_DAY #TODO: I'm not sure why I need to multiply by these constants here. But this works and if I don't do this the test fails.
            result = CLM5.make_K_matrix(self.config.taus, self.config.zsoi, w_scalar, t_scalar, o_scalar,
                                   n_scalar, decomp_depth_efolding, self.config.nlevels)

            # get the expected result from the matlab code
            expected = example_data['result'].squeeze()
            
            np.testing.assert_allclose(result, expected)
            
    def test_global_run(self):
        """Test a global run of CLM4.5 against gold standard netCDF output."""
     

        GOLD_EXAMPLE_FILES = glob(path.join('tests/test_data/CLM45/global_run_positive_examples/', '*nc'))

        self.assertGreater(len(GOLD_EXAMPLE_FILES), 0)

        # load the expected results
        expected = xr.open_dataarray(GOLD_EXAMPLE_FILES[0], decode_times=False)
        df = expected.sum(dim='time').to_dataframe()
        df = df[df>0].dropna()
        latlons = df.index.tolist()



        empty_gridcell = xr.concat([xr.full_like(self.global_data.make_ldd(-90, 0).X0, fill_value=np.nan)]*12, dim = 'TIME')

        # Run the model in parallel over all grid cells with a progress bar
        # Using module-level function for picklability
        results = Parallel(n_jobs=-1, verbose=10)(
            delayed(run_model_for_cell)(lat, lon, self.config, self.global_data, empty_gridcell) 
            for lat, lon in tqdm(latlons)
        )

        # Merge the results into a single xarray dataset
        global_result_da = xr.concat(results, dim='cell').unstack().transpose('TIME','y', 'x','pools','LEVDCMP1_10')
        
        # Calculate the depth integrated C content
        global_tot_C = (global_result_da * self.config.dz[:self.config.nlevels]).sum(dim=['pools','LEVDCMP1_10'])
        global_tot_C = global_tot_C.where(global_tot_C > 0)

        for i, (lat, lon) in enumerate(latlons):
            try:
                np.testing.assert_allclose(global_tot_C.sel(y=lat, x=lon).fillna(0), expected.sel(latitude = lat, longitude = lon).fillna(0), rtol=1e-2, atol=1e-1)
            except AssertionError as e:
                print(f"Assertion failed for cell {i} lat: {lat}, lon: {lon}")
                raise e

def make_CLM_config_years():
    """CLM4.5 config in YEAR units, as used by the F_new prediction pipeline.

    Distinct from make_CLM_config() above, which converts taus and Gamma_soil to
    seconds. Mirrors notebooks/04_CLM45_model_predictions.py so the analytical
    F_new is tested against the same parameterization the predictions use.
    """
    mat = loadmat('data/CLM5_global_simulation/soildepth.mat')
    zisoi, zsoi = mat['zisoi'].squeeze(), mat['zsoi'].squeeze()
    dz, dz_node = mat['dz'].squeeze(), mat['dz_node'].squeeze()

    global_da = xr.open_dataset('data/CLM5_global_simulation/global_demo_in.nc')
    global_da = global_da.rename({'LON': 'x', 'LAT': 'y'})
    global_da['x'] = xr.where(global_da['x'] >= 180, global_da['x'] - 360, global_da['x'])
    global_da = global_da.sortby('x')

    CLM_params = xr.open_dataset('data/CLM5_global_simulation/clm5_params.c171117.nc')
    taus = np.array([CLM_params['tau_cwd'],
                     CLM_params['tau_l1'],
                     CLM_params['tau_l2_l3'],
                     CLM_params['tau_l2_l3'],
                     CLM_params['tau_s1'],
                     CLM_params['tau_s2'],
                     CLM_params['tau_s3']]).squeeze()

    config = ConfigParams(decomp_depth_efolding=0.5, taus=taus, Gamma_soil=1e-4,
                          F_soil=0, zsoi=zsoi, zisoi=zisoi, dz=dz, dz_node=dz_node,
                          nlevels=10, npools=7)
    return config, GlobalData(global_da)


class TestCLM45FnewAccuracy(unittest.TestCase):
    """Check the analytical F_new against a direct labeled/unlabeled tracer run.

    The analytical F_new (soil_diskin.age_dist_utils.calc_age_dist_cdf, used by
    notebooks/experimental/04_CLM45_model_predictions_sasu.py) claims that

        F_new(a) = 1 - 1^T expm(M a) eta,   eta = X_ss / sum(X_ss)

    is the fraction of soil carbon younger than age `a`. The ground truth here
    is the definition of that quantity, simulated directly in the style of
    notebooks/tracer_simulation_test.ipynb: carry a labeled and an unlabeled
    pool and take F_new = labeled / (labeled + unlabeled).

    Both are evaluated on the annual-mean operator M = A K_bar - V, so this
    isolates the accuracy of the analytical solution. The separate question of
    whether the annual mean stands in for the model's monthly-varying forcing is
    covered by notebooks/experimental/compare_sasu_fnew_methods.py (max
    difference 3.9e-3 across the 99 sites).
    """

    # Balesdent (2018) site 19: Amazon, and one of the longest labeling
    # durations in the data set (4000 yr), which exercises the slow passive SOM
    # pool rather than only the fast litter turnover. make_ldd() snaps to the
    # nearest grid cell, so these rounded coordinates select the same cell.
    LAT, LON = -7.516667, -63.033333
    DURATION_LABELING = 4000.0

    # Integrate the spin-up for this many multiples of the slowest e-folding
    # time. 15 leaves a relative residual of about exp(-15) ~ 3e-7.
    SPINUP_TAU_MULTIPLE = 15

    @classmethod
    def setUpClass(cls):
        config, global_data = make_CLM_config_years()
        model = CLM5(config, global_data.make_ldd(cls.LAT, cls.LON))

        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            cls.M = model.A @ model.K_ts.mean(axis=0) - model.V
        cls.I_bar = np.asarray(model.I).mean(axis=0)
        cls.n = cls.M.shape[0]

        # Slowest mode of the system sets how long spin-up has to run.
        real_eigs = np.linalg.eigvals(cls.M).real
        cls.tau_slow = -1.0 / real_eigs[real_eigs < 0].max()

        # Spin up numerically from bare soil, rather than seeding with the
        # analytical steady state, so the ground truth stays independent of the
        # solution under test.
        t_spin = cls.SPINUP_TAU_MULTIPLE * cls.tau_slow
        # t_eval keeps only the final state; without it solve_ivp retains every
        # internal step, which matters a lot for the monthly-forced variant of
        # this test (see TestCLM45FnewAccuracyMonthlyForcing).
        spin = solve_ivp(lambda t, X: cls.I_bar + cls.M @ X, (0, t_spin),
                         np.zeros(cls.n), method='LSODA', rtol=1e-10, atol=1e-8,
                         t_eval=[t_spin])
        cls.X_spun = spin.y[:, -1]

        # Labeling experiment: at t=0 all incoming carbon becomes labeled while
        # the standing stock decays unlabeled.
        cls.ages = np.array([10.0, 100.0, 1000.0, cls.DURATION_LABELING])
        labeled = solve_ivp(lambda t, X: cls.I_bar + cls.M @ X,
                            (0, cls.ages[-1]), np.zeros(cls.n), method='LSODA',
                            rtol=1e-10, atol=1e-8, t_eval=cls.ages)
        unlabeled = solve_ivp(lambda t, X: cls.M @ X,
                              (0, cls.ages[-1]), cls.X_spun, method='LSODA',
                              rtol=1e-10, atol=1e-8, t_eval=cls.ages)
        cls.labeled_C = labeled.y.sum(axis=0)
        cls.unlabeled_C = unlabeled.y.sum(axis=0)
        cls.fnew_simulated = cls.labeled_C / (cls.labeled_C + cls.unlabeled_C)

        cls.fnew_analytical = np.asarray(
            calc_age_dist_cdf(cls.M, cls.I_bar, cls.ages)
        ).reshape(-1)

    def test_spinup_reaches_the_analytical_steady_state(self):
        """Numerical spin-up must land on the semi-analytical steady state.

        Validates X_ss = -(A K_bar - V)^-1 I_bar, which sets the F_new
        denominator, without assuming it.
        """
        X_ss = np.linalg.solve(-self.M, self.I_bar)
        np.testing.assert_allclose(self.X_spun, X_ss, rtol=1e-4)

    def test_total_carbon_is_conserved_during_labeling(self):
        """Started at steady state, labeled + unlabeled must stay constant.

        If this drifts, the simulated F_new denominator is moving and the
        comparison below would be meaningless.
        """
        total = self.labeled_C + self.unlabeled_C
        np.testing.assert_allclose(total, total[0], rtol=1e-6)

    def test_analytical_fnew_matches_tracer_simulation(self):
        """The analytical age CDF must reproduce labeled / total carbon."""
        np.testing.assert_allclose(self.fnew_analytical, self.fnew_simulated,
                                   rtol=1e-5)

    def test_analytical_fnew_at_the_labeling_duration(self):
        """Spot-check at the site's actual labeling duration (4000 yr)."""
        idx = int(np.argmin(np.abs(self.ages - self.DURATION_LABELING)))
        simulated = self.fnew_simulated[idx]
        analytical = self.fnew_analytical[idx]

        # A meaningful check needs F_new to be informative, not pinned at 0 or 1.
        self.assertGreater(simulated, 0.5)
        self.assertLess(simulated, 1.0)
        self.assertAlmostEqual(analytical, simulated, places=6)


@unittest.skipUnless(
    os.environ.get('RUN_SLOW_TESTS') == '1',
    "slow (~1 hr): spins up 350+ kyr through the model's monthly forcing. "
    "Set RUN_SLOW_TESTS=1 to run."
)
class TestCLM45FnewAccuracyMonthlyForcing(unittest.TestCase):
    """Same check as TestCLM45FnewAccuracy, but driven by the model's own _dX.

    TestCLM45FnewAccuracy integrates the annual-mean operator M = A K_bar - V,
    which isolates the accuracy of the analytical solution. This version instead
    integrates CLM5._dX directly, so the ground truth carries the model's
    monthly-varying environmental scalars. It therefore tests something
    stronger and different: that the annual-mean analytical F_new is still right
    when the underlying system is actually driven by time-varying forcing.

    Cost: _dX cycles its forcing 144 times per year (see the month index
    `int((t % (1/12)) * 144)`), so the stiff solver takes ~10^6 steps to cover a
    350 kyr spin-up. That is roughly an hour, which is why this is opt-in via
    RUN_SLOW_TESTS=1 rather than part of the default suite.

    The tolerance here is set by physics, not numerics: replacing the monthly
    forcing with its annual mean shifts F_new by ~4e-5 at this site (and at most
    3.9e-3 across all 99 sites, per
    notebooks/experimental/compare_sasu_fnew_methods.py), so rtol=1e-3 leaves
    headroom while still catching a genuinely wrong analytical solution.
    """

    LAT, LON = TestCLM45FnewAccuracy.LAT, TestCLM45FnewAccuracy.LON
    DURATION_LABELING = TestCLM45FnewAccuracy.DURATION_LABELING
    SPINUP_TAU_MULTIPLE = 15

    @classmethod
    def setUpClass(cls):
        config, global_data = make_CLM_config_years()
        ldd = global_data.make_ldd(cls.LAT, cls.LON)

        # Model carrying the real carbon input, used for spin-up and for the
        # labeled pool.
        cls.model = CLM5(config, ldd)

        # A copy with the input zeroed drives the unlabeled pool, which only
        # decays. Assigning .I goes through the CLM5 property setter, so the
        # ndarray cache _dX reads stays in sync.
        cls.model_no_input = CLM5(config, ldd)
        cls.model_no_input.I = np.zeros_like(np.asarray(cls.model_no_input.I))

        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            cls.M = cls.model.A @ cls.model.K_ts.mean(axis=0) - cls.model.V
        cls.I_bar = np.asarray(cls.model.I).mean(axis=0)

        real_eigs = np.linalg.eigvals(cls.M).real
        cls.tau_slow = -1.0 / real_eigs[real_eigs < 0].max()

        n = cls.model.X_size
        t_spin = cls.SPINUP_TAU_MULTIPLE * cls.tau_slow
        # t_eval is essential here, not cosmetic: _dX changes its forcing 144
        # times a year, so over 350 kyr the stiff solver takes on the order of
        # 1e8 internal steps. Without t_eval solve_ivp retains the state at
        # every one of them (~70 floats each) and the run is OOM-killed. Asking
        # for only the final time keeps memory flat.
        spin = solve_ivp(cls.model._dX, (0, t_spin), np.zeros(n),
                         method='LSODA', t_eval=[t_spin])
        cls.X_spun = spin.y[:, -1]

        cls.ages = np.array([10.0, 100.0, 1000.0, cls.DURATION_LABELING])
        labeled = solve_ivp(cls.model._dX, (0, cls.ages[-1]), np.zeros(n),
                            method='LSODA', t_eval=cls.ages)
        unlabeled = solve_ivp(cls.model_no_input._dX, (0, cls.ages[-1]),
                              cls.X_spun, method='LSODA', t_eval=cls.ages)

        cls.labeled_C = labeled.y.sum(axis=0)
        cls.unlabeled_C = unlabeled.y.sum(axis=0)
        cls.fnew_simulated = cls.labeled_C / (cls.labeled_C + cls.unlabeled_C)
        cls.fnew_analytical = np.asarray(
            calc_age_dist_cdf(cls.M, cls.I_bar, cls.ages)
        ).reshape(-1)

    def test_spinup_total_carbon_matches_analytical_steady_state(self):
        """Total spun-up carbon must match sum(X_ss), which sets F_new's denominator.

        The monthly-forced system settles onto a periodic orbit rather than a
        fixed point, so individual pools oscillate about the annual-mean steady
        state (see the per-pool test below). The total, which is the quantity
        F_new is normalized by, is far steadier: starting exactly on X_ss and
        running _dX, total carbon stays within 6.4e-4 of sum(X_ss), so 5e-3
        leaves roughly 8x headroom.
        """
        X_ss = np.linalg.solve(-self.M, self.I_bar)
        np.testing.assert_allclose(self.X_spun.sum(), X_ss.sum(), rtol=5e-3)

    def test_spinup_pools_orbit_the_analytical_steady_state(self):
        """Individual pools sit near X_ss, within their periodic-orbit amplitude.

        This tolerance is set by measurement, not by taste. Initializing exactly
        at X_ss and integrating _dX, the orbit amplitude per pool is: Litter1
        5.3% (fastest pool, so it tracks the intra-annual forcing most closely),
        Litter2/3 ~1.5%, CWD 0.8%, SOM1 0.9%, SOM2 0.4%, SOM3 0.01%. 10% keeps
        roughly 2x headroom over the worst pool while still catching a spin-up
        that has genuinely landed in the wrong place.
        """
        X_ss = np.linalg.solve(-self.M, self.I_bar)
        np.testing.assert_allclose(self.X_spun, X_ss, rtol=1e-1)

    def test_total_carbon_is_conserved_during_labeling(self):
        """labeled + unlabeled stays put, so the F_new denominator is stable."""
        total = self.labeled_C + self.unlabeled_C
        np.testing.assert_allclose(total, total[0], rtol=1e-2)

    def test_analytical_fnew_matches_monthly_forced_simulation(self):
        """Analytical F_new survives the model's real time-varying forcing."""
        np.testing.assert_allclose(self.fnew_analytical, self.fnew_simulated,
                                   rtol=1e-3)

    def test_analytical_fnew_at_the_labeling_duration(self):
        """Spot-check at the site's actual labeling duration (4000 yr)."""
        idx = int(np.argmin(np.abs(self.ages - self.DURATION_LABELING)))
        simulated = self.fnew_simulated[idx]

        self.assertGreater(simulated, 0.5)
        self.assertLess(simulated, 1.0)
        np.testing.assert_allclose(self.fnew_analytical[idx], simulated, rtol=1e-3)


if __name__ == '__main__':
    unittest.main()

