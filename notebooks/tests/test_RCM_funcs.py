import numpy as np
import pandas as pd
import xarray as xr

from .. import process_he_2016 as he

# he.get_RCM_A_u()

ds = xr.open_dataset('results/process_he_2016/CESM_params.nc')
print(ds.drop_vars('spatial_ref').to_dataframe()['CESM'].unstack())
params_df = pd.read_csv(f'data/he_2016/Persistence-master/CodeData/He/CESM/compartmentalParameters.txt',sep=' ')
# As = np.stack([he.get_RCM_A_u(row[1],correct=True)[0] for row in params_df.iterrows()])
# print(u)
# print(As.mean(axis=0))
print(params_df.mean())