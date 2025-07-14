import numpy as np
import pandas as pd
from soil_diskin import age_dist_utils as util

def test_age_dist():
    # Test the validity of the age distribution function by comparing the results of a specific model in 
    # Sierra et al. 2018. Specifically, we look at the RothC results.
    A = np.array([[-10, 0, 0, 0],
                [0, -0.3, 0, 0],
                [1.02, 0.03, -0.59, 0],
                [1.2, 0.04, 0.08, -0.02]
                ])
    ksRC = np.array([10, 0.3, 0.66, 0.02])
    FYMsplit = np.array([0.49, 0.49, 0.02])
    DR=1.44; In=1.7; FYM=0; clay=23.4
    x = 1.67 * (1.85 + 1.60 * np.exp(-0.0786 * clay))
    B = 0.46 / (x + 1) # Proportion that goes to the BIO pool
    H = 0.54 / (x + 1) # Proportion that goes to the HUM pool

    ai3 = B * ksRC
    ai4 = H * ksRC

    ARC = np.diag(-ksRC)
    ARC[2,:] = ARC[2,:] + ai3
    ARC[3,:] = ARC[3,:] + ai4

    RcI=np.array([In * (DR / (DR + 1)) + (FYM * FYMsplit[0]), In * (1 / (DR + 1)) + (FYM * FYMsplit[1]), 0, (FYM * FYMsplit[2])])

    ages = np.arange(1,1001)
    pA = util.box_model_ss_age_dist(ARC,RcI,ages)

    sierra = pd.read_csv('tests/test_data/sierra_2018_RothC.csv')
    np.testing.assert_almost_equal(pA.flatten(), sierra['age_pdf'].values)
