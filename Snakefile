# Snakefile for soil carbon modeling workflow
# This workflow processes soil data, calibrates models, and generates results

import pandas as pd
import zipfile
import os
import requests
from os import path

HE_2016_URL = "https://git.bgc-jena.mpg.de/csierra/Persistence/-/archive/master/Persistence-master.zip"

# Define the final target files
rule all:
    input:
        # Plotting outputs -- should force all the models to be run
        # "figures/model_predictions.png",
        "figures/fig1.png",
        "figures/fig2.png",
        "figures/fig3.png",
        "figures/figS1.png",
        "figures/figS3.png",
        "figures/figS4.png",
        "figures/figS5.png",
        "figures/figS6.png",

# Step 00: Download necessary data files using curl.
# NOTE: wget was hard to install with UV for some reason. Using curl instead.
# Balesdent et al. 2018 data
rule download_balesdent_data:
    output:
        "data/balesdent_2018/balesdent_2018_raw.xlsx"
    shell:
        """
        mkdir -p data/balesdent_2018
        curl -L -o {output} https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-018-0328-3/MediaObjects/41586_2018_328_MOESM3_ESM.xlsx
        """

rule download_he_2016:
    output:
        # just a sentinel that the data has been downloaded
        "data/he_2016/Persistence-master/CodeData/14C_Respiration.csv"
    shell: # do the above in shell
        """
        curl -L -o he_2016.zip {HE_2016_URL}
        unzip he_2016.zip -d data/he_2016
        rm he_2016.zip
        rm -rf data/he_2016/Persistence-master/CodeData/WorldGrids/
        """

rule download_CLM45_conf:
    output:        
        "data/CLM5_global_simulation/global_demo_in.nc",
        "data/CLM5_global_simulation/soildepth.mat"
    params:
        zipfile="data/CLM5_global_simulation/gcb_matrix_supp_data.zip"
    shell:
        """
        mkdir -p data/CLM5_global_simulation/
        curl -L -o {params.zipfile} https://hs.pangaea.de/model/Huang-etal_2017/gcb_matrix_supp_data.zip
        pushd data/CLM5_global_simulation/ 
        unzip -j gcb_matrix_supp_data.zip
        popd
        """

# 14C data
rule download_shi_data:
    output:
        "data/shi_2020/global_delta_14C.nc"
    shell:
        """
        mkdir -p data/shi_2020/
        curl -L -o {output} https://zenodo.org/records/3823612/files/global_delta_14C.nc
        """

# NPP data
rule download_kang_data:
    output:
        "data/kang_2023/ST_CFE-Hybrid_NT.zip"
    shell:
        """
        mkdir -p data/kang_2023/
        curl -L -o {output} https://zenodo.org/records/8212707/files/ST_CFE-Hybrid_NT.zip?download=1
        pushd data/kang_2023/ && gunzip -k ST_CFE-Hybrid_NT.zip
        popd
        """

# Step 01: Preprocess Balesdent data
rule preprocess_balesdent:
    input:
        "data/balesdent_2018/balesdent_2018_raw.xlsx"
    output:
        "results/processed_balesdent_2018.csv"
    shell:
        "python notebooks/01_preprocess_balesdent_data.py -i {input} -o {output} --backfill"

rule preprocess_14C_data:
    output:
        "data/14C_atm_annot.csv"
    script:
        "notebooks/01_preprocess_14C_data.py"

# Step 02: Get 14C turnover data for the Balesdent sites
rule turnover_14C:
    input:
        "results/processed_balesdent_2018.csv",
        "data/shi_2020/global_delta_14C.nc",
    output:
        "results/all_sites_14C_turnover.csv"
    script:
        "notebooks/02_get_turnover_14C.py"

# Step 03a: Calibrate power law models
rule calibrate_powerlaw:
    input:
        "results/all_sites_14C_turnover.csv",
        "data/14C_atm_annot.csv",
    output:
        "results/03_calibrate_models/powerlaw_model_optimization_results.csv"
    script:
        "notebooks/03a_calibrate_powerlaw_model.py"

rule lognormal_age_scan_mathematica:
    input:
        "data/14C_atm_annot.csv"
    output:
        "results/03_calibrate_models/03b_lognormal_model_age_scan.csv",
        "results/03_calibrate_models/03b_lognormal_model_age_scan_05.csv",
        "results/03_calibrate_models/03b_lognormal_model_age_scan_95.csv"
    shell:
        """
        wolframscript --file notebooks/03b_lognormal_age_scan.wls
        """

rule calibrate_lognormal_python:
    input:
        "results/all_sites_14C_turnover.csv",
        "results/03_calibrate_models/03b_lognormal_model_age_scan.csv",
        "results/03_calibrate_models/03b_lognormal_model_age_scan_05.csv",
        "results/03_calibrate_models/03b_lognormal_model_age_scan_95.csv",
        "data/14C_atm_annot.csv"
    output:
        "results/03_calibrate_models/03b_lognormal_predictions_calcurve.csv",
    script:
        "notebooks/03b_calibrate_lognormal_model.py"

rule calibrate_generalized_powerlaw:
    input:
        "results/all_sites_14C_turnover.csv",
    output:
        "results/03_calibrate_models/general_powerlaw_model_optimization_results.csv",
        "results/03_calibrate_models/general_powerlaw_model_optimization_results_beta_half.csv"
    script:
        "notebooks/03c_calibrate_generalized_powerlaw_model.py"

# Step 03d: Calibrate gamma model
rule calibrate_gamma:
    input:
        "results/all_sites_14C_turnover.csv"
    output:
        "results/03_calibrate_models/gamma_model_optimization_results.csv"
    script:
        "notebooks/03d_calibrate_gamma_model.py"

# Step 04: Generate and collect model predictions for analysis and figures
# Step 04b: Lognormal predictions (Julia)
rule lognormal_predictions_julia:
    input:
        "results/03_calibrate_models/03b_lognormal_predictions_calcurve.csv",
    output:
        "results/04_model_predictions/04b_lognormal_cdfs.csv",
        "results/04_model_predictions/04b_lognormal_cdfs_05.csv",
        "results/04_model_predictions/04b_lognormal_cdfs_95.csv"
    shell:
        """
        julia --project=./ notebooks/04b_lognormal_predictions.jl
        """

# Download JSBACH files for parameterization of other models
rule download_jsbach_data:
    output:
        "data/model_params/JSBACH/JSBACH_S3_tas.nc",
        "data/model_params/JSBACH/JSBACH_S3_pr.nc",
        "data/model_params/JSBACH/JSBACH_S3_npp.nc"
    shell:
        """
        mkdir -p data/model_params/JSBACH
        curl -L -o data/model_params/JSBACH/JSBACH_S3_tas.nc https://gcbo-opendata.s3.eu-west-2.amazonaws.com/trendyv12-gcb2023/JSBACH/S3/JSBACH_S3_tas.nc
        curl -L -o data/model_params/JSBACH/JSBACH_S3_pr.nc https://gcbo-opendata.s3.eu-west-2.amazonaws.com/trendyv12-gcb2023/JSBACH/S3/JSBACH_S3_pr.nc
        curl -L -o data/model_params/JSBACH/JSBACH_S3_npp.nc https://gcbo-opendata.s3.eu-west-2.amazonaws.com/trendyv12-gcb2023/JSBACH/S3/JSBACH_S3_npp.nc
        """

# Collect continuum model predictions
rule continuum_model_predictions:
    input:
        "results/processed_balesdent_2018.csv",
        "results/all_sites_14C_turnover.csv",
        "results/03_calibrate_models/powerlaw_model_optimization_results.csv",
        "results/04_model_predictions/04b_lognormal_cdfs.csv",
        "results/03_calibrate_models/general_powerlaw_model_optimization_results.csv",
        "results/03_calibrate_models/general_powerlaw_model_optimization_results_beta_half.csv",
        "results/03_calibrate_models/gamma_model_optimization_results.csv"
    output:
        "results/04_model_predictions/gamma_model_predictions.csv",
        "results/04_model_predictions/power_law_model_predictions.csv",
        "results/04_model_predictions/lognormal_model_predictions.csv",
        "results/04_model_predictions/general_power_law_model_predictions.csv",
        "results/04_model_predictions/general_power_law_model_predictions_beta_half.csv",
    script:
        "notebooks/04_collect_continuum_model_predictions.py"

# Run CLM4.5
rule CLM45_model_predictions:
    input:
        "data/CLM5_global_simulation/soildepth.mat",
        "data/CLM5_global_simulation/global_demo_in.nc",
        "data/CLM5_global_simulation/clm5_params.c171117.nc",
        "results/processed_balesdent_2018.csv",
        "results/all_sites_14C_turnover.csv",
    output:
        "results/04_model_predictions/CLM45.csv",
        "results/04_model_predictions/CLM45_fnew.csv"
    script:
        "notebooks/04_CLM45_model_predictions.py"

# Run JSBACH
rule JSBACH_model_predictions:
    input:
        "data/model_params/JSBACH/JSBACH_S3_tas.nc",
        "data/model_params/JSBACH/JSBACH_S3_pr.nc",
        "data/model_params/JSBACH/JSBACH_S3_npp.nc",
        "results/processed_balesdent_2018.csv",
        "results/all_sites_14C_turnover.csv",
    output:
        "results/04_model_predictions/JSBACH.csv",
        "results/04_model_predictions/JSBACH_fnew.csv"
    script:
        "notebooks/04_JSBACH_model_predictions.py"

# Run the reduced complexity models from He et al. 2016
rule RC_model_predictions:
    input:
        "results/processed_balesdent_2018.csv",
        "results/all_sites_14C_turnover.csv",
        # Sentinel that the He et al. 2016 data has been downloaded
        "data/he_2016/Persistence-master/CodeData/14C_Respiration.csv"
    output:
        f"results/04_model_predictions/RCM.csv",
    script:
        "notebooks/04_RC_model_predictions.py"

# Step 06a: Turnover Sensitivity analysis
rule turnover_sensitivity_analysis_powerlaw_gamma:
    input:
        "results/processed_balesdent_2018.csv",
        "results/all_sites_14C_turnover.csv",
    output:
        "results/06_sensitivity_analysis/powerlaw_turnover_sensitivity_results.csv",
        "results/06_sensitivity_analysis/gamma_turnover_sensitivity_results.csv",
    script:
        "notebooks/06a_turnover_sensitivity.py"

rule turnover_sensitivity_analysis_lognormal_mathematica:
    input:
        'data/14C_atm_annot.csv',
        'results/all_sites_14C_turnover.csv',
    output:
        "results/06_sensitivity_analysis/06a_lognormal_age_scan0.50.csv",
        "results/06_sensitivity_analysis/06a_lognormal_age_scan0.67.csv",
        "results/06_sensitivity_analysis/06a_lognormal_age_scan1.csv",
        "results/06_sensitivity_analysis/06a_lognormal_age_scan1.50.csv",
        "results/06_sensitivity_analysis/06a_lognormal_age_scan2.csv",
    shell:
        """
        wolframscript --file notebooks/06a_lognormal_turnover_sensitivity.wls
        """

rule turnover_sensitivity_analysis_lognormal_python:
    input:
        "results/06_sensitivity_analysis/06a_lognormal_age_scan0.50.csv",
        "results/06_sensitivity_analysis/06a_lognormal_age_scan0.67.csv",
        "results/06_sensitivity_analysis/06a_lognormal_age_scan1.csv",
        "results/06_sensitivity_analysis/06a_lognormal_age_scan1.50.csv",
        "results/06_sensitivity_analysis/06a_lognormal_age_scan2.csv",
    output:
        "results/06_sensitivity_analysis/lognormal_age_predictions.csv",
    script:
        "notebooks/06a_lognormal_turnover_sensitivity.py"

rule turnover_sensitivity_analysis_lognormal_julia:
    input:
        "results/06_sensitivity_analysis/lognormal_age_predictions.csv",
        "results/all_sites_14C_turnover.csv"
    output:
        "results/06_sensitivity_analysis/06a_lognormal_cdfs_0.50.csv",
        "results/06_sensitivity_analysis/06a_lognormal_cdfs_0.67.csv",
        "results/06_sensitivity_analysis/06a_lognormal_cdfs_1.csv",
        "results/06_sensitivity_analysis/06a_lognormal_cdfs_1.50.csv",
        "results/06_sensitivity_analysis/06a_lognormal_cdfs_2.csv",
    shell:
        """
        julia --project=./ notebooks/06a_lognormal_turnover_sensitivity.jl
        """

# Step 06b: Steady-state sensitivity analysis
rule steady_state_sensitivity_analysis_lognormal:
    input:
        "data/balesdent_2018/balesdent_2018_raw.xlsx",
        "results/03_calibrate_models/03b_lognormal_predictions_calcurve.csv",
    output:
        "results/06_sensitivity_analysis/lognormal_input_data.csv",
        "results/06_sensitivity_analysis/lognormal_mu_data.csv",
        "results/06_sensitivity_analysis/lognormal_sigma_data.csv",
    shell:
        """
        julia --project=./ notebooks/06b_lognormal_steady_state_sensitivity.jl
        """
# Step 06c: Vegetation effects sensitivity analysis
rule vegetation_effects_sensitivity_analysis:
    input:
        'data/balesdent_2018/balesdent_2018_raw.xlsx',
        'results/processed_balesdent_2018.csv',
        'results/03_calibrate_models/powerlaw_model_optimization_results.csv',
        'results/03_calibrate_models/gamma_model_optimization_results.csv',
        'results/06_sensitivity_analysis/06a_lognormal_cdfs_1.csv', 
    output:
        'results/06_sensitivity_analysis/06c_model_predictions_veg_effects.csv',
    script:
        "notebooks/06c_veg_effects_sensitivity.py"
# Step 05: Plot results

rule plot_fig1:
    input:
    output:
        "figures/fig1.png",
    script:
        "notebooks/fig1.py"

rule fig2_calcs:
    input:
    output:
        "results/fig2_calcs.npz",
    script:
        "notebooks/fig2_calcs.py"

rule plot_fig2:
    input:
        "results/fig2_calcs.npz",
    output:
        "figures/fig2.png"
    script:
        "notebooks/fig2.py"

rule fig3_calcs:
    input:
        "results/04_model_predictions/gamma_model_predictions.csv",
        "results/04_model_predictions/power_law_model_predictions.csv",
        "results/04_model_predictions/lognormal_model_predictions.csv",
        "results/04_model_predictions/general_power_law_model_predictions.csv",
        "results/04_model_predictions/general_power_law_model_predictions_beta_half.csv",
        'results/04_model_predictions/CLM45_fnew.csv',
        'results/04_model_predictions/JSBACH_fnew.csv',
        'results/04_model_predictions/RCM.csv',
        'results/processed_balesdent_2018.csv',
    output:
        "results/fig3_calcs.csv",
    script:
        "notebooks/fig3_calcs.py"


rule plot_fig3:
    input:
        "results/04_model_predictions/gamma_model_predictions.csv",
        "results/04_model_predictions/power_law_model_predictions.csv",
        "results/04_model_predictions/lognormal_model_predictions.csv",
        "results/04_model_predictions/general_power_law_model_predictions.csv",
        "results/04_model_predictions/general_power_law_model_predictions_beta_half.csv",
        'results/04_model_predictions/CLM45_fnew.csv',
        'results/04_model_predictions/JSBACH_fnew.csv',
        'results/04_model_predictions/RCM.csv',
        'results/processed_balesdent_2018.csv',
        'results/fig3_calcs.csv',
    output:
        "figures/fig3.png",
        "figures/figS3.png" # also make figS3 here
    script:
        "notebooks/fig3.py"

rule plot_figS1:
    input:
        'data/balesdent_2018/balesdent_2018_raw.xlsx',
    output:
        'figures/figS1.png',
    script:
        "notebooks/figS1.py"

rule plot_figS4:
    input:
        'results/06_sensitivity_analysis/powerlaw_turnover_sensitivity_results.csv',
        'results/06_sensitivity_analysis/gamma_turnover_sensitivity_results.csv',
        "results/06_sensitivity_analysis/06a_lognormal_cdfs_0.50.csv",
        "results/06_sensitivity_analysis/06a_lognormal_cdfs_0.67.csv",
        "results/06_sensitivity_analysis/06a_lognormal_cdfs_1.csv",
        "results/06_sensitivity_analysis/06a_lognormal_cdfs_1.50.csv",
        "results/06_sensitivity_analysis/06a_lognormal_cdfs_2.csv",
    output:
        'figures/figS4.png',
    script:
        "notebooks/figS4.py"

rule plot_figS5:
    input:
        "data/balesdent_2018/balesdent_2018_raw.xlsx",
        "results/processed_balesdent_2018.csv", 
        "results/03_calibrate_models/powerlaw_model_optimization_results.csv",
        "results/03_calibrate_models/gamma_model_optimization_results.csv",
        'results/06_sensitivity_analysis/lognormal_input_data.csv',
        'results/06_sensitivity_analysis/lognormal_mu_data.csv',
        'results/06_sensitivity_analysis/lognormal_sigma_data.csv',
    output:
        "figures/figS5.png"
    script:
        "notebooks/figS5.py"

rule plot_figS6:
    input:
        'results/06_sensitivity_analysis/06c_model_predictions_veg_effects.csv',
    output:
        "figures/figS6.png"
    script:
        "notebooks/figS6.py"



# Clean up rule
rule clean:
    shell:
        """
        rm -rf results/*.csv
        rm -rf figures/*
        rm -rf results/*.html
        rm -rf data/balesdent_2018/*
        rm -rf data/CLM5_global_simulation/global_demo_in.nc
        rm -rf data/CLM5_global_simulation/soildepth.mat
        rm -rf data/he_2016/*
        rm -rf data/kang_2023/*
        rm -rf data/shi_2020/*
        """

rule clean_results:
    shell:
        """
        rm -rf results/*.csv
        rm -rf figures/*
        rm -rf results/*.html
        """
