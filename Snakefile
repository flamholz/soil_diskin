# Snakefile for soil carbon modeling workflow
# This workflow processes soil data, calibrates models, and generates results

import hashlib
import pandas as pd
import zipfile
import os
import requests
from os import path

HE_2016_URL = "https://git.bgc-jena.mpg.de/csierra/Persistence/-/archive/master/Persistence-master.zip"
SHI_URL = "https://zenodo.org/records/3823612/files/global_delta_14C.nc"
SHI_MD5 = "645aa8d54cbc36cb329c29bcffc4352b"  # published on the Zenodo record
# ISRaD v2.9.9 (2025-08-14), pinned to the commit of its database zip. ISRaD publishes
# no checksum; this SHA-256 is of the zip as downloaded on 2026-10-29. To update,
# change the commit, then the checksum (and the file names in the ISRaD scripts).
ISRAD_COMMIT = "2cf05252bd12b7afd1bcf5d9ab15213fcd0d851b"
ISRAD_URL = f"https://github.com/International-Soil-Radiocarbon-Database/ISRaD/raw/{ISRAD_COMMIT}/ISRaD_data_files/database/ISRaD_database_files.zip"
ISRAD_SHA256 = "d37558b9aeded060619a80763cc9a29c7f15717a2aba20b434ffcca4d0a45df1"
ISRAD_LAYERS = "data/ISRaD/ISRaD_data_flat_layer_v 2.9.9.2025-08-14.csv"  # in the pinned zip


def verify_checksum(path, expected, algorithm="md5"):
    """Raise if the checksum of a downloaded file differs from its expected value."""
    with open(path, "rb") as stream:
        digest = hashlib.file_digest(stream, algorithm).hexdigest()
    if digest != expected:
        raise ValueError(f"{path}: {algorithm} {digest} does not match the expected {expected}")

# Define the final target files
rule all:
    input:
        # Plotting outputs -- should force all the models to be run
        # "figures/model_predictions.png",
        "figures/fig1.png",
        "figures/fig2.png",
        "figures/fig3.png",
        "figures/fig2_old.png",
        "figures/figSX.png",
        "figures/figS1.png",
        "figures/figS2.png",
        "figures/figS3.png",
        "figures/figS4.png",
        "figures/figS5.png",
        "figures/figS6.png",
        "figures/figS7.png",
        "figures/figS8.png",
        # Calibration exports, including the permuted-fm null
        "results/04_calibration_exports/powerlaw_calibration_results.csv",
        "results/04_calibration_exports/powerlaw_calibration_results_permuted.csv",
        "results/04_calibration_exports/lognormal_calibration_results.csv",
        "results/04_calibration_exports/lognormal_calibration_results_permuted.csv",

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
        unzip -o he_2016.zip -d data/he_2016
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
    run:
        shell("mkdir -p data/shi_2020/ && curl -L -o {output} {SHI_URL}")
        verify_checksum(output[0], SHI_MD5, "md5")

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

# ISRaD database
rule download_israd_data:
    output:
        "data/ISRaD/.israd_download_complete"
    run:
        shell("mkdir -p data/ISRaD && curl -L -o israd_database_files.zip {ISRAD_URL}")
        verify_checksum("israd_database_files.zip", ISRAD_SHA256, "sha256")
        shell("unzip -o israd_database_files.zip -d data/ISRaD && rm -f israd_database_files.zip && touch {output}")

# Whittaker biome boundaries
rule download_worldclim:
    output:
        directory("data/worldclim")
    shell:
        """
        mkdir -p data/worldclim
        curl -L -o data/worldclim/wc2.1_10m_bio.zip https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_10m_bio.zip
        unzip data/worldclim/wc2.1_10m_bio.zip -d data/worldclim
        rm data/worldclim/wc2.1_10m_bio.zip
        """

rule download_whittaker_biomes:
    output:
        "data/whittaker_biomes/biomes.csv"
    shell:
        """
        mkdir -p data/whittaker_biomes
        curl -L -o {output} https://raw.githubusercontent.com/kunstler/BIOMEplot/refs/heads/master/inst/extdata/biomes.csv
        """

# Step 01: Preprocess Balesdent data
rule preprocess_balesdent:
    input:
        "data/balesdent_2018/balesdent_2018_raw.xlsx"
    output:
        "results/processed_balesdent_2018.csv"
    shell:
        "python notebooks/01_preprocess_balesdent_data.py -i {input} -o {output} --backfill"

# Step 01b: Preprocess 14C atmospheric history data
rule preprocess_14C_data:
    input:
        "data/14C_atm.csv"
    output:
        "data/14C_atm_annot.csv"
    shell:
        "python notebooks/01_preprocess_14C_history.py -i {input} -o {output}"

# Step 02a: NPP for every Balesdent location (bulk and layers) from MODIS via Earth Engine.
# The only step that needs Earth Engine; both step 02 tables read its output.
rule fetch_npp:
    input:
        sites="results/processed_balesdent_2018.csv",
        layers="results/processed_balesdent_2018_sampled_soilgrids.csv",
        code=["notebooks/02a_fetch_npp.py", "soil_diskin/npp.py"],
    output:
        csv="results/02_npp/site_npp.csv",
        meta="results/02_npp/site_npp.json",
    shell:
        "python notebooks/02a_fetch_npp.py --locations {input.sites:q} {input.layers:q} -o {output.csv:q}"

# Step 02: Bulk 0-100 cm radiocarbon and turnover for the Balesdent sites
rule turnover_14C:
    input:
        sites="results/processed_balesdent_2018.csv",
        shi="data/shi_2020/global_delta_14C.nc",
        npp="results/02_npp/site_npp.csv",
        code=["notebooks/02_get_turnover_14C.py", "soil_diskin/site_inputs.py"],
    output:
        csv="results/all_sites_14C_turnover.csv",
        meta="results/all_sites_14C_turnover.json",
    shell:
        "python notebooks/02_get_turnover_14C.py -i {input.sites:q} -o {output.csv:q} --shi {input.shi:q} --npp {input.npp:q}"

# Step 02b: Permutation null -- fm_obs shuffled across sites (see config.yaml)
rule permute_fm:
    input:
        "results/all_sites_14C_turnover.csv",
        "config.yaml",
    output:
        "results/all_sites_14C_turnover_permuted.csv"
    script:
        "notebooks/02b_permute_fm.py"

# Step 01c: Reported Balesdent layers; missing stocks backfilled from SoilGrids
rule preprocess_balesdent_layers:
    input:
        "data/balesdent_2018/balesdent_2018_raw.xlsx"
    output:
        csv="results/processed_balesdent_2018_sampled_soilgrids.csv",
        meta="results/processed_balesdent_2018_sampled_soilgrids.json",
    shell:
        "python notebooks/01_preprocess_balesdent_data.py -i {input} -o {output.csv} --depth-resolved --backfill"

# Step 02c: Layer radiocarbon and turnover; NPP inputs 50% to 0-10 cm, 50% Jackson global
rule turnover_14C_layered:
    input:
        layers="results/processed_balesdent_2018_sampled_soilgrids.csv",
        shi="data/shi_2020/global_delta_14C.nc",
        npp="results/02_npp/site_npp.csv",
        code=["notebooks/02c_get_layer_turnover_14C.py", "soil_diskin/site_inputs.py"],
    output:
        csv="results/all_sites_14C_turnover_depth_resolved.csv",
        meta="results/all_sites_14C_turnover_depth_resolved.json",
    shell:
        "python notebooks/02c_get_layer_turnover_14C.py -i {input.layers:q} -o {output.csv:q} --shi {input.shi:q} --npp {input.npp:q}"

# ISRaD: pseudo-bulk 0-100 cm profiles with radiocarbon, NPP and turnover times, the
# ISRaD counterpart of steps 01-02. Not part of `all` yet; run `snakemake israd`.
rule israd:
    input:
        "results/israd/israd_turnover.csv",
        "results/03_calibrate_models/israd/powerlaw_model_optimization_results.csv",
        "results/03_calibrate_models/israd/lognormal_model_optimization_results.csv",

rule preprocess_israd:
    input:
        download="data/ISRaD/.israd_download_complete",
        balesdent="results/processed_balesdent_2018.csv",
        code=["notebooks/01_preprocess_israd.py", "soil_diskin/israd.py"],
    params:
        layers=ISRAD_LAYERS,
    output:
        csv="results/israd/israd_profiles.csv",
        meta="results/israd/israd_profiles.json",
    shell:
        "python notebooks/01_preprocess_israd.py --israd {params.layers:q} --balesdent {input.balesdent:q} -o {output.csv:q}"

rule fetch_npp_israd:
    input:
        profiles="results/israd/israd_profiles.csv",
        code=["notebooks/02a_fetch_npp.py", "soil_diskin/npp.py"],
    output:
        csv="results/02_npp/israd_npp.csv",
        meta="results/02_npp/israd_npp.json",
    shell:
        "python notebooks/02a_fetch_npp.py --locations {input.profiles:q} -o {output.csv:q} --sd"

rule israd_turnover:
    input:
        profiles="results/israd/israd_profiles.csv",
        npp="results/02_npp/israd_npp.csv",
        code=["notebooks/02_get_israd_turnover.py", "soil_diskin/site_inputs.py"],
    output:
        csv="results/israd/israd_turnover.csv",
        meta="results/israd/israd_turnover.json",
    shell:
        "python notebooks/02_get_israd_turnover.py -i {input.profiles:q} -o {output.csv:q} --npp {input.npp:q}"

# ISRaD calibrations: each profile is fit with the atmosphere of its sampling year;
# profiles sampled after the record ends (2013) get calibration_status no_atmosphere.
rule calibrate_powerlaw_israd:
    input:
        sites="results/israd/israd_turnover.csv",
        atm="data/14C_atm_annot.csv",
        code=["notebooks/03a_calibrate_powerlaw_model.py", "soil_diskin/calibration.py",
              "soil_diskin/continuum_models.py", "soil_diskin/radiocarbon_utils.py"],
    threads: workflow.cores
    output:
        "results/03_calibrate_models/israd/powerlaw_model_optimization_results.csv"
    shell:
        "python notebooks/03a_calibrate_powerlaw_model.py --input {input.sites:q} --output {output:q} "
        "--year-column sampling_year --n-jobs {threads}"

rule calibrate_lognormal_israd:
    input:
        sites="results/israd/israd_turnover.csv",
        atm="data/14C_atm_annot.csv",
        code=["notebooks/03b_lognormal_calibration.py", "soil_diskin/calibration.py",
              "soil_diskin/continuum_models.py", "soil_diskin/radiocarbon_utils.py"],
    threads: workflow.cores
    output:
        "results/03_calibrate_models/israd/lognormal_model_optimization_results.csv"
    shell:
        "python notebooks/03b_lognormal_calibration.py --input {input.sites:q} "
        "--output-dir results/03_calibrate_models/israd --year-column sampling_year --n-jobs {threads}"

# Step 03a: Calibrate power law models
rule calibrate_powerlaw:
    input:
        "results/all_sites_14C_turnover.csv",
        "data/14C_atm_annot.csv",
        "soil_diskin/calibration.py",
        "soil_diskin/continuum_models.py",
        "notebooks/03a_calibrate_powerlaw_model.py",
    threads: workflow.cores
    output:
        "results/03_calibrate_models/powerlaw_model_optimization_results.csv"
    shell:
        "python notebooks/03a_calibrate_powerlaw_model.py --input {input[0]:q} --output {output:q} --n-jobs {threads}"

rule calibrate_powerlaw_permuted:
    input:
        "results/all_sites_14C_turnover_permuted.csv",
        "data/14C_atm_annot.csv",
        "soil_diskin/calibration.py",
        "soil_diskin/continuum_models.py",
        "notebooks/03a_calibrate_powerlaw_model.py",
    threads: workflow.cores
    output:
        "results/03_calibrate_models/powerlaw_model_optimization_results_permuted.csv"
    shell:
        "python notebooks/03a_calibrate_powerlaw_model.py --input {input[0]:q} --output {output:q} --n-jobs {threads}"



rule calibrate_generalized_powerlaw:
    input:
        "results/all_sites_14C_turnover.csv",
        "soil_diskin/calibration.py",
        "soil_diskin/continuum_models.py",
        "notebooks/03c_calibrate_generalized_powerlaw_model.py",
    threads: workflow.cores
    output:
        "results/03_calibrate_models/general_powerlaw_model_optimization_results.csv",
        "results/03_calibrate_models/general_powerlaw_model_optimization_results_alpha_half.csv",
        "results/03_calibrate_models/general_powerlaw_model_alpha_sweep.csv"
    shell:
        "python notebooks/03c_calibrate_generalized_powerlaw_model.py --input {input[0]:q} --output-dir results/03_calibrate_models --n-jobs {threads}"


rule calibrate_soft_cutoff_powerlaw:
    input:
        "results/all_sites_14C_turnover.csv",
        "soil_diskin/calibration.py",
        "soil_diskin/continuum_models.py",
        "notebooks/03f_calibrate_soft_cutoff_powerlaw_model.py",
    threads: workflow.cores
    output:
        "results/03_calibrate_models/soft_cutoff_powerlaw_model_alpha_sweep.csv"
    shell:
        "python notebooks/03f_calibrate_soft_cutoff_powerlaw_model.py --input {input[0]:q} --output {output:q} --n-jobs {threads}"


# Fit lognormal mu/sigma directly to turnover and radiocarbon.
rule run_lognormal_calibration_python:
    input:
        sites="results/all_sites_14C_turnover.csv",
        atm="data/14C_atm_annot.csv",
        code=["notebooks/03b_lognormal_calibration.py", "soil_diskin/calibration.py",
              "soil_diskin/continuum_models.py", "soil_diskin/radiocarbon_utils.py"],
    output:
        "results/03_calibrate_models/lognormal_model_optimization_results.csv",
    threads: workflow.cores
    shell:
        """
        python notebooks/03b_lognormal_calibration.py --input {input.sites:q} --n-jobs {threads}
        """

rule run_layered_lognormal_calibration_python:
    input:
        sites="results/all_sites_14C_turnover_depth_resolved.csv",
        atm="data/14C_atm_annot.csv",
        code=["notebooks/03b_lognormal_calibration.py", "soil_diskin/calibration.py",
              "soil_diskin/continuum_models.py", "soil_diskin/radiocarbon_utils.py"],
    output:
        "results/03_calibrate_models/depth_resolved/lognormal_model_optimization_results.csv",
    threads: workflow.cores
    shell:
        """
        python notebooks/03b_lognormal_calibration.py --input {input.sites:q} \
            --output-dir results/03_calibrate_models/depth_resolved --n-jobs {threads}
        """

rule run_lognormal_calibration_permuted:
    input:
        sites="results/all_sites_14C_turnover_permuted.csv",
        atm="data/14C_atm_annot.csv",
        code=["notebooks/03b_lognormal_calibration.py", "soil_diskin/calibration.py",
              "soil_diskin/continuum_models.py", "soil_diskin/radiocarbon_utils.py"],
    output:
        "results/03_calibrate_models/lognormal_model_optimization_results_permuted.csv",
    threads: workflow.cores
    shell:
        """
        python notebooks/03b_lognormal_calibration.py --input {input.sites:q} \
            --output-name lognormal_model_optimization_results_permuted.csv --n-jobs {threads}
        """

# Step 03d: Calibrate weibull model
rule calibrate_weibull:
    input:
        "results/all_sites_14C_turnover.csv",
        "soil_diskin/calibration.py",
        "soil_diskin/continuum_models.py",
        "notebooks/03d_calibrate_weibull_model.py",
    threads: workflow.cores
    output:
        "results/03_calibrate_models/weibull_model_optimization_results.csv"
    shell:
        "python notebooks/03d_calibrate_weibull_model.py --input {input[0]:q} --output {output:q} --n-jobs {threads}"

# Step 03e: Calibrate log-uniform rate model
rule calibrate_loguniform:
    input:
        "results/all_sites_14C_turnover.csv",
        "data/14C_atm_annot.csv",
        "soil_diskin/calibration.py",
        "soil_diskin/continuum_models.py",
        "notebooks/03e_calibrate_loguniform_model.py",
    threads: workflow.cores
    output:
        "results/03_calibrate_models/loguniform_model_optimization_results.csv"
    shell:
        "python notebooks/03e_calibrate_loguniform_model.py --input {input[0]:q} --output {output:q} --n-jobs {threads}"

# Step 04: Generate and collect model predictions for analysis and figures
# Lognormal predictions (Python-based pipeline used for figures)
rule lognormal_predictions_python:
    input:
        "results/03_calibrate_models/lognormal_model_optimization_results.csv",
    output:
        "results/04_model_predictions/04b_lognormal_cdfs_python.csv",
        "results/04_model_predictions/04b_lognormal_cdfs_05_python.csv",
        "results/04_model_predictions/04b_lognormal_cdfs_95_python.csv"
    shell:
        """
        python notebooks/04b_lognormal_predictions_fast.py --params-path {input}
        """

rule loguniform_model_predictions:
    input:
        "results/processed_balesdent_2018.csv",
        "results/03_calibrate_models/loguniform_model_optimization_results.csv",
    output:
        "results/04_model_predictions/loguniform_model_predictions.csv"
    script:
        "notebooks/04_loguniform_model_predictions.py"

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
        "results/03_calibrate_models/lognormal_model_optimization_results.csv",
        "results/03_calibrate_models/depth_resolved/lognormal_model_optimization_results.csv",
        "results/03_calibrate_models/general_powerlaw_model_optimization_results.csv",
        "results/03_calibrate_models/general_powerlaw_model_optimization_results_alpha_half.csv",
        "results/03_calibrate_models/general_powerlaw_model_alpha_sweep.csv",
        "results/03_calibrate_models/soft_cutoff_powerlaw_model_alpha_sweep.csv",
        "results/03_calibrate_models/weibull_model_optimization_results.csv",
        "results/03_calibrate_models/powerlaw_model_optimization_results_permuted.csv",
        "results/03_calibrate_models/lognormal_model_optimization_results_permuted.csv",
    output:
        "results/04_model_predictions/weibull_model_predictions.csv",
        "results/04_model_predictions/power_law_model_predictions.csv",
        "results/04_model_predictions/lognormal_model_predictions.csv",
        "results/04_model_predictions/depth_resolved/lognormal_model_predictions.csv",
        "results/04_model_predictions/general_power_law_model_predictions.csv",
        "results/04_model_predictions/general_power_law_model_predictions_alpha_half.csv",
        "results/04_model_predictions/general_power_law_model_predictions_alpha_sweep.csv",
        "results/04_model_predictions/soft_cutoff_power_law_model_predictions_alpha_sweep.csv",
        "results/04_model_predictions/power_law_model_predictions_permuted.csv",
        "results/04_model_predictions/lognormal_model_predictions_permuted.csv",
    script:
        "notebooks/04_collect_continuum_model_predictions.py"

# Export calibrated parameters with site identifiers, for the real and permuted fm
rule export_calibration_results:
    input:
        "results/processed_balesdent_2018.csv",
        "results/03_calibrate_models/powerlaw_model_optimization_results.csv",
        "results/03_calibrate_models/powerlaw_model_optimization_results_permuted.csv",
        "results/03_calibrate_models/lognormal_model_optimization_results.csv",
        "results/03_calibrate_models/lognormal_model_optimization_results_permuted.csv",
        "notebooks/04_export_calibration_results.py",
    output:
        "results/04_calibration_exports/powerlaw_calibration_results.csv",
        "results/04_calibration_exports/powerlaw_calibration_results_permuted.csv",
        "results/04_calibration_exports/lognormal_calibration_results.csv",
        "results/04_calibration_exports/lognormal_calibration_results_permuted.csv",
    shell:
        "python notebooks/04_export_calibration_results.py"

# Run CLM4.5
rule CLM45_model_predictions:
    input:
        "data/CLM5_global_simulation/soildepth.mat",
        "data/CLM5_global_simulation/global_demo_in.nc",
        "data/CLM5_global_simulation/clm5_params.c171117.nc",
        "results/processed_balesdent_2018.csv",
        "results/all_sites_14C_turnover.csv",
    output:
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
        "results/06_sensitivity_analysis/powerlaw_turnover_sensitivity_results_q05.csv",
        "results/06_sensitivity_analysis/powerlaw_turnover_sensitivity_results_q95.csv",
    script:
        "notebooks/06a_turnover_sensitivity.py"

rule turnover_sensitivity_analysis_lognormal:
    input:
        turnover='results/all_sites_14C_turnover.csv',
        sites="results/processed_balesdent_2018.csv",
        atm='data/14C_atm_annot.csv',
        code=["notebooks/06a_lognormal_turnover_sensitivity.py", "soil_diskin/calibration.py"],
    threads: workflow.cores
    output:
        "results/06_sensitivity_analysis/lognormal_age_predictions.csv",
        "results/06_sensitivity_analysis/06a_lognormal_cdfs_0.50.csv",
        "results/06_sensitivity_analysis/06a_lognormal_cdfs_0.67.csv",
        "results/06_sensitivity_analysis/06a_lognormal_cdfs_1.csv",
        "results/06_sensitivity_analysis/06a_lognormal_cdfs_1.50.csv",
        "results/06_sensitivity_analysis/06a_lognormal_cdfs_2.csv",
        "results/06_sensitivity_analysis/06a_lognormal_fnew.csv",
        "results/06_sensitivity_analysis/06a_lognormal_fnew_q05.csv",
        "results/06_sensitivity_analysis/06a_lognormal_fnew_q95.csv",
    shell:
        """
        python notebooks/06a_lognormal_turnover_sensitivity.py --input {input.turnover:q} \
            --sites {input.sites:q} --atm {input.atm:q} \
            --output-dir results/06_sensitivity_analysis --n-jobs {threads}
        """

# Step 06b: Steady-state sensitivity analysis
rule steady_state_sensitivity_analysis_lognormal:
    input:
        "data/balesdent_2018/balesdent_2018_raw.xlsx",
        "results/03_calibrate_models/lognormal_model_optimization_results.csv",
    output:
        "results/06_sensitivity_analysis/lognormal_input_data.csv",
        "results/06_sensitivity_analysis/lognormal_mu_data.csv",
        "results/06_sensitivity_analysis/lognormal_sigma_data.csv",
    script:
        "notebooks/06b_lognormal_steady_state_sensitivity.py"
# Step 06c: Vegetation effects sensitivity analysis
rule vegetation_effects_sensitivity_analysis:
    input:
        sites='results/processed_balesdent_2018.csv',
        powerlaw='results/03_calibrate_models/powerlaw_model_optimization_results.csv',
        lognormal='results/03_calibrate_models/lognormal_model_optimization_results.csv',
        code="notebooks/06c_veg_effects_sensitivity.py",
    output:
        'results/06_sensitivity_analysis/06c_model_predictions_veg_effects.csv',
    shell:
        """
        python notebooks/06c_veg_effects_sensitivity.py --sites {input.sites:q} \
            --powerlaw {input.powerlaw:q} --lognormal {input.lognormal:q} --output {output:q}
        """
# Step 05: Plot results

# Figure 1: continuum model structure and age distributions
rule plot_fig1:
    input:
        "graphics/parallel_continuum_model.png",
        "soil_diskin/continuum_models.py",
    output:
        "figures/fig1.png",
    script:
        "notebooks/fig1.py"

# Figure 2
rule plot_fig2:
    input:
        "soil_diskin/continuum_models.py",
    output:
        "figures/fig2.png",
        "figures/fig2.svg",
    script:
        "notebooks/fig2.py"

# Former Figure 1, now supplementary (number to be assigned)
rule figSX_calcs:
    input:
    output:
        "results/figSX_calcs.npz",
    script:
        "notebooks/figSX_calcs.py"

rule plot_figSX:
    input:
        "results/figSX_calcs.npz",
        "graphics/parallel_continuum_model.png",
    output:
        "figures/figSX.png",
    script:
        "notebooks/figSX.py"

# Former Figure 2, not currently in the paper
rule fig2_old_calcs:
    input:
        "data/14C_atm_annot.csv"
    output:
        "results/fig2_old_calcs.npz",
    script:
        "notebooks/fig2_old_calcs.py"

rule plot_fig2_old:
    input:
        "results/fig2_old_calcs.npz",
    output:
        "figures/fig2_old.png"
    script:
        "notebooks/fig2_old.py"

# Step 05: bootstrap RMSE and KGE of every model's f_new predictions
rule evaluate_model_predictions:
    input:
        "results/04_model_predictions/weibull_model_predictions.csv",
        "results/04_model_predictions/power_law_model_predictions.csv",
        "results/04_model_predictions/lognormal_model_predictions.csv",
        "results/04_model_predictions/general_power_law_model_predictions.csv",
        "results/04_model_predictions/general_power_law_model_predictions_alpha_half.csv",
        'results/04_model_predictions/CLM45_fnew.csv',
        'results/04_model_predictions/JSBACH_fnew.csv',
        'results/04_model_predictions/RCM.csv',
        'results/processed_balesdent_2018.csv',
    output:
        "results/05_model_evaluation/bootstrap_metrics.csv",
    script:
        "notebooks/05_evaluate_model_predictions.py"


rule plot_figS3:
    input:
        "results/04_model_predictions/weibull_model_predictions.csv",
        "results/04_model_predictions/power_law_model_predictions.csv",
        "results/04_model_predictions/lognormal_model_predictions.csv",
        "results/04_model_predictions/general_power_law_model_predictions.csv",
        "results/04_model_predictions/general_power_law_model_predictions_alpha_half.csv",
        'results/04_model_predictions/CLM45_fnew.csv',
        'results/04_model_predictions/JSBACH_fnew.csv',
        'results/04_model_predictions/RCM.csv',
        'results/processed_balesdent_2018.csv',
        'results/05_model_evaluation/bootstrap_metrics.csv',
    output:
        "figures/figS3.png"
    script:
        "notebooks/figS3.py"

rule plot_fig3:
    input:
        "results/04_model_predictions/lognormal_model_predictions.csv",
        "results/04_model_predictions/depth_resolved/lognormal_model_predictions.csv",
        "notebooks/fig3.py",
        "notebooks/style.mpl",
        "notebooks/viz.py",
        "soil_diskin/utils.py",
    output:
        "figures/fig3.png",
        "figures/fig3.csv",
    shell:
        "python -m notebooks.fig3"

rule figS2_calcs:
    input:
        "results/all_sites_14C_turnover.csv",
        "data/worldclim",
        "results/03_calibrate_models/lognormal_model_optimization_results.csv",
        "results/03_calibrate_models/powerlaw_model_optimization_results.csv",
        "results/03_calibrate_models/general_powerlaw_model_optimization_results.csv",
        "results/03_calibrate_models/general_powerlaw_model_optimization_results_alpha_half.csv",
    output:
        "results/figS2_calcs.csv",
        "results/figS2_calcs_lognormal_correlations.csv",
        "results/figS2_calcs_powerlaw_alpha1_correlations.csv",
        "results/figS2_calcs_powerlaw_alpha_exp_gamma_correlations.csv",
        "results/figS2_calcs_powerlaw_alpha_exp_gamma_half_correlations.csv",
    script:
        "notebooks/figS2_calcs.py"

rule plot_figS2:
    input:
        "results/03_calibrate_models/powerlaw_model_optimization_results.csv",
        "results/03_calibrate_models/general_powerlaw_model_optimization_results.csv",
        "results/03_calibrate_models/general_powerlaw_model_optimization_results_alpha_half.csv",
        "results/03_calibrate_models/lognormal_model_optimization_results.csv",
        "results/figS2_calcs.csv",
        "results/figS2_calcs_lognormal_correlations.csv",
        "results/figS2_calcs_powerlaw_alpha1_correlations.csv",
        "results/figS2_calcs_powerlaw_alpha_exp_gamma_correlations.csv",
        "results/figS2_calcs_powerlaw_alpha_exp_gamma_half_correlations.csv",
    output:
        "figures/figS2.png",
    script:
        "notebooks/figS2.py"

rule plot_figS1:
    input:
        'data/balesdent_2018/balesdent_2018_raw.xlsx',
        'data/whittaker_biomes/biomes.csv',
    output:
        'figures/figS1.png',
    script:
        "notebooks/figS1.py"

rule plot_figS4:
    input:
        'results/06_sensitivity_analysis/powerlaw_turnover_sensitivity_results.csv',
        'results/06_sensitivity_analysis/powerlaw_turnover_sensitivity_results_q05.csv',
        'results/06_sensitivity_analysis/powerlaw_turnover_sensitivity_results_q95.csv',
        'results/06_sensitivity_analysis/06a_lognormal_fnew.csv',
        'results/06_sensitivity_analysis/06a_lognormal_fnew_q05.csv',
        'results/06_sensitivity_analysis/06a_lognormal_fnew_q95.csv',
        'results/processed_balesdent_2018.csv',
    output:
        'figures/figS4.png',
    script:
        "notebooks/figS4.py"

rule plot_figS5:
    input:
        "data/balesdent_2018/balesdent_2018_raw.xlsx",
        "results/processed_balesdent_2018.csv",
        "results/03_calibrate_models/powerlaw_model_optimization_results.csv",
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

rule plot_figS7:
    input:
        "results/04_model_predictions/depth_resolved/lognormal_model_predictions.csv",
        "data/14C_atm_annot.csv",
        "notebooks/figS7.py",
        "notebooks/fig3.py",
        "soil_diskin/site_inputs.py",
        "notebooks/03b_lognormal_calibration.py",
        "notebooks/04_collect_continuum_model_predictions.py",
        "notebooks/style.mpl",
        "notebooks/viz.py",
        "soil_diskin/continuum_models.py",
        "soil_diskin/radiocarbon_utils.py",
        "soil_diskin/utils.py",
    output:
        "figures/figS7.png"
    threads: 4
    shell:
        "python -m notebooks.figS7 --n-jobs {threads}"

rule plot_figS8:
    input:
        'results/processed_balesdent_2018.csv',
        'results/04_model_predictions/loguniform_model_predictions.csv',
        'results/03_calibrate_models/loguniform_model_optimization_results.csv',
    output:
        "figures/figS8.png",
        "figures/figS8_parameters.png",
    script:
        "notebooks/figS8.py"



# Clean up rule
rule clean:
    shell:
        """
        find results -type f -name '*.csv' -delete || true
        find results -type f -name '*.npz' -delete || true
        rm -rf figures/*
        rm -rf data/balesdent_2018/*
        rm -rf data/CLM5_global_simulation/global_demo_in.nc
        rm -rf data/CLM5_global_simulation/soildepth.mat
        rm -rf data/he_2016/*
        rm -rf data/kang_2023/*
        rm -rf data/shi_2020/*
        rm -rf data/ISRaD/*
        """

rule clean_results:
    shell:
        """
        find results -type f -name '*.csv' -not -path 'results/experimental/*' -delete || true
        find results -type f -name '*.npz' -not -path 'results/experimental/*' -delete || true
        rm -rf figures/*
        """

# Intended to trigger a rerun of continuum models without forcing CLM45 or JSBACH to rerun.
rule clean_light:
    shell:
        """
        find results -type f -name '*.csv' -not -path 'results/experimental/*' ! -name 'CLM45*.csv' ! -name 'JSBACH*.csv' ! -name 'processed_balesdent_2018.csv' ! -name 'all_sites_14C_turnover.csv' ! -name 'processed_balesdent_2018_sampled_soilgrids.csv' ! -name 'all_sites_14C_turnover_depth_resolved.csv' ! -path 'results/02_npp/*' -delete || true
        find results -type f -name '*.npz' -not -path 'results/experimental/*' ! -name 'CLM45*.npz' ! -name 'JSBACH*.npz' -delete || true
        rm -rf figures/*
        """
