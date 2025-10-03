# Snakefile for soil carbon modeling workflow
# This workflow processes soil data, calibrates models, and generates results

import pandas as pd
import zipfile
import os
import requests
from os import path

current_date = pd.Timestamp.now().date().strftime("%d-%m-%Y")

HE_2016_URL = "https://git.bgc-jena.mpg.de/csierra/Persistence/-/archive/master/Persistence-master.zip"

# Define the final target files
rule all:
    input:
        # Preprocessing outputs
        "results/processed_balesdent_2018.csv",
        "results/processed_balesdent_2018.csv",
        "results/all_sites_14C_turnover.csv",

        # # Calibration outputs
        "results/03_calibrate_models/powerlaw_model_optimization_results.csv",
        "results/03b_lognormal_site_parameters.csv",
        "results/03b_lognormal_model_predictions_14C.csv",
        "results/03_calibrate_models/powerlaw_model_optimization_results.csv",
        "results/03_calibrate_models/general_powerlaw_model_optimization_results.csv",
        "results/03_calibrate_models/gamma_model_optimization_results.csv",
        "results/03_calibrate_models/03b_lognormal_predictions_calcurve.csv",
        
        # # Prediction outputs
        # "results/model_predictions.csv",
        # "results/lognormal_predictions.csv",
        
        # Plotting outputs
        f"figures/model_predictions_{current_date}.png",

        # # Sensitivity analysis outputs
        # "results/sensitivity_powerlaw.csv",
        # "results/sensitivity_lognormal.csv"

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
        "data/CLM5_global_simulation/gcb_matrix_supp_data.zip"
        "data/CLM5_global_simulation/global_demo_in.nc",
        "data/CLM5_global_simulation/clm5_params.c171117.nc",
        ""
    shell:
        """
        mkdir -p data/CLM5_global_simulation/
        curl -L -o {output} https://hs.pangaea.de/model/Huang-etal_2017/gcb_matrix_supp_data.zip
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
    script:
        "notebooks/01_preprocess_balesdent_data.py"

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

rule calibrate_lognormal_mathematica:
    input:
        "results/all_sites_14C_turnover.csv",
        "data/14C_atm_annot.csv"
    output:
        "results/03b_lognormal_site_parameters.csv",
        "results/03b_lognormal_model_predictions_14C.csv"
    shell:
        """
        wolframscript --file notebooks/03b_calibrate_lognormal_model.wls
        """

rule lognormal_age_scan_mathematica:
    input:
        "data/14C_atm_annot.csv"
    output:
        "results/03_calibrate_models/03b_lognormal_model_age_scan.csv",
    shell:
        """
        wolframscript --file notebooks/03b_lognormal_age_scan.wls
        """

rule calibrate_lognormal_python:
    input:
        "results/all_sites_14C_turnover.csv",
        "results/03b_lognormal_site_parameters.csv",
        "results/03b_lognormal_model_predictions_14C.csv",
        "results/03_calibrate_models/03b_lognormal_model_age_scan.csv",
        "data/14C_atm_annot.csv"
    output:
        "results/03_calibrate_models/03b_lognormal_predictions_calcurve.csv",
    script:
        "notebooks/03b_calibrate_lognormal_model.py"

rule calibrate_generalized_powerlaw:
    input:
        "results/all_sites_14C_turnover.csv",
    output:
        "results/03_calibrate_models/general_powerlaw_model_optimization_results.csv"
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
        "results/04_model_predictions/04b_lognormal_cdfs.csv"
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

# Collects results for the disordered models generated
# by the above rules, and generates predictions from
# our implementation of the box models e.g., CABLE. 
# rule collect_model_predictions_all:
#     input:
#         "data/CLM5_global_simulation/gcb_matrix_supp_data.zip",
#         "results/processed_balesdent_2018.csv",
#         "results/all_sites_14C_turnover.csv",
#         "results/03_calibrate_models/powerlaw_model_optimization_results.csv",
#         "results/04_model_predictions/04b_lognormal_cdfs.csv",
#         "data/model_params/JSBACH/JSBACH_S3_tas.nc",
#         "data/model_params/JSBACH/JSBACH_S3_pr.nc",
#         "data/model_params/JSBACH/JSBACH_S3_npp.nc"
#     output:
#         # TODO: add the rest
#         f"results/04_model_predictions/power_law_{current_date}_all2.csv",
#         f"results/04_model_predictions/lognormal_{current_date}.csv",
#         f"results/04_model_predictions/CABLE_{current_date}.pkl",
#         f"results/04_model_predictions/CLM45_fnew_{current_date}.csv",
#         f"results/04_model_predictions/JSBACH_fnew_{current_date}.csv",
#         f"results/04_model_predictions/RCM_{current_date}.csv",
#     script:
#         "notebooks/04_model_predictions_all.py"

# rule collect_model_predictions:
#     input:
#         "data/CLM5_global_simulation/gcb_matrix_supp_data.zip",
#         "results/processed_balesdent_2018.csv",
#         "results/all_sites_14C_turnover.csv",
#         "results/03_calibrate_models/powerlaw_model_optimization_results.csv",
#         "results/04_model_predictions/04b_lognormal_cdfs.csv",
#         "results/03_calibrate_models/general_powerlaw_model_optimization_results.csv",
#         "data/model_params/JSBACH/JSBACH_S3_tas.nc",
#         "data/model_params/JSBACH/JSBACH_S3_pr.nc",
#         "data/model_params/JSBACH/JSBACH_S3_npp.nc"
#     output:
#         f"results/04_model_predictions/CABLE_{current_date}.pkl",
#     script:
#         "notebooks/04_model_predictions.py"

rule continuum_model_predictions:
    input:
        "results/processed_balesdent_2018.csv",
        "results/all_sites_14C_turnover.csv",
        "results/03_calibrate_models/powerlaw_model_optimization_results.csv",
        "results/04_model_predictions/04b_lognormal_cdfs.csv",
        "results/03_calibrate_models/general_powerlaw_model_optimization_results.csv",
    output:
        f'results/04_model_predictions/gamma_{current_date}.csv',
        f'results/04_model_predictions/power_law_{current_date}.csv',
        f'results/04_model_predictions/lognormal_{current_date}.csv',
        f'results/04_model_predictions/general_power_law_{current_date}.csv',
    script:
        "notebooks/04_collect_continuum_model_predictions.py"

rule cable_model_predictions:
    input:
        "results/processed_balesdent_2018.csv",
        "results/all_sites_14C_turnover.csv",
    output:
        f"results/04_model_predictions/CABLE_{current_date}.pkl",
    script:
        "notebooks/04_CABLE_model_predictions.py"

rule CLM45_model_predictions:
    input:
        "data/CLM5_global_simulation/soildepth.mat",
        "data/CLM5_global_simulation/gcb_matrix_supp_data.zip",
        "data/CLM5_global_simulation/global_demo_in.nc",
        "results/processed_balesdent_2018.csv",
        "results/all_sites_14C_turnover.csv",
    output:
        f"results/04_model_predictions/CLM45_{current_date}.csv",
        f"results/04_model_predictions/CLM45_fnew_{current_date}.csv"
    script:
        "notebooks/04_CLM45_model_predictions.py"

rule JSBACH_model_predictions:
    input:
        "data/model_params/JSBACH/JSBACH_S3_tas.nc",
        "data/model_params/JSBACH/JSBACH_S3_pr.nc",
        "data/model_params/JSBACH/JSBACH_S3_npp.nc",
        "results/processed_balesdent_2018.csv",
        "results/all_sites_14C_turnover.csv",
    output:
        f"results/04_model_predictions/JSBACH_{current_date}.csv",
        f"results/04_model_predictions/JSBACH_fnew_{current_date}.csv"
    script:
        "notebooks/04_JSBACH_model_predictions.py"

rule RC_model_predictions:
    input:
        "results/processed_balesdent_2018.csv",
        "results/all_sites_14C_turnover.csv",
        "data/he_2016/Persistence-master/CodeData/14C_Respiration.csv"
    output:
        f"results/04_model_predictions/RCM_{current_date}.csv",
    script:
        "notebooks/04_RC_model_predictions.py"

# Step 05: Plot results
rule plot_results_v2_all:
    input:
        f'results/04_model_predictions/power_law_{current_date}.csv',
        f'results/04_model_predictions/lognormal_{current_date}.csv',
        f'results/04_model_predictions/CLM45_fnew_{current_date}.csv',
        f'results/04_model_predictions/JSBACH_fnew_{current_date}.csv',
        f'results/04_model_predictions/RCM_{current_date}.csv',
    output:
        f"figures/model_predictions_{current_date}.png"
    script:
        "notebooks/05_plot_results_v2_all.py"

# rule plot_results_v2:
#     input:
#         "results/model_predictions_subset.csv"
#     output:
#         "results/figures/results_plots_subset.png"
#     script:
#         "notebooks/05_plot_results_v2.py"

# rule plot_results:
#     input:
#         "results/model_predictions.csv"
#     output:
#         "results/figures/basic_plots.png"
#     script:
#         "notebooks/05_plot_results.py"

# # Step 06: Sensitivity analysis
# rule sensitivity_analysis_old:
#     input:
#         "results/model_predictions.csv"
#     output:
#         "results/sensitivity_old_results.html"
#     shell:
#         """
#         jupyter nbconvert --to html --execute notebooks/06_sensitivity_analysis_old.ipynb
#         mv notebooks/06_sensitivity_analysis_old.html {output}
#         """

# rule sensitivity_analysis_powerlaw:
#     input:
#         "results/powerlaw_model_params.csv"
#     output:
#         "results/sensitivity_powerlaw.csv"
#     shell:
#         """
#         jupyter nbconvert --to html --execute notebooks/06a_sensitivity_analysis_powerlaw.ipynb
#         touch {output}
#         """

# rule sensitivity_analysis_lognormal:
#     input:
#         "results/lognormal_model_params.csv"
#     output:
#         "results/sensitivity_lognormal.csv"
#     shell:
#         """
#         jupyter nbconvert --to html --execute notebooks/06b_sensitivity_analysis_lognormal.ipynb
#         touch {output}
#         """

# rule sensitivity_analysis_lognormal_julia:
#     input:
#         "results/lognormal_model_params.csv"
#     output:
#         "results/sensitivity_lognormal_julia.csv"
#     shell:
#         """
#         julia notebooks/06b_sensitivity_analysis_lognormal.jl
#         """

# # Create results directories
# rule create_dirs:
#     output:
#         directory("results/figures")
#     shell:
#         "mkdir -p results/figures"

# Clean up rule
rule clean:
    shell:
        """
        rm -rf results/*.csv
        rm -rf figures/*
        rm -rf results/*.html
        rm -rf data/balesdent_2018/*
        rm -rf data/CLM5_global_simulation/*
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
