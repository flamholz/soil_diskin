model_names = {'CESM':'CESM1','IPSL':'IPSL-CM5A-LR','MRI':'MRI-ESM1'}
file_names = {'CESM':'areacella_fx_CESM1-BGC_1pctCO2_r0i0p0','IPSL':'areacella_fx_IPSL-CM5A-LR_historicalNat_r0i0p0','MRI':'areacella_fx_MRI-ESM1_esmControl_r0i0p0'}

def parse_he_data(model='CESM'):
    
    grid = xr.open_dataset(f'../../data/CMIP5/{model_names[model]}/{file_names[model]}.nc')['areacella']
    params_raster = xr.zeros_like(grid).T
    params_raster = params_raster.where(params_raster!=0)
    params_df = pd.read_csv(f'../../Persistence/CodeData/He/{model}/compartmentalParameters.txt',sep=' ')
    ds = []
    for par in params_df.columns[1:]:
        x = params_raster.values.flatten()
        x[params_df['id']] = params_df[par]
        x2 = x.reshape(params_raster.shape)
        tmp_ds = params_raster.copy()
        tmp_ds.values = np.fliplr(x2)
        ds.append(tmp_ds.T)

    ds = xr.concat(ds,dim='parameter')
    ds['parameter'] = params_df.columns[1:]
    ds.name = model
    ds['lon'] = ds['lon']-180
    return ds

