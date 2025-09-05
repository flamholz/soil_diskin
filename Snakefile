# Snakefile for soil carbon modeling workflow
# This workflow processes soil data, calibrates models, and generates results

# Define the final target files
rule all:
    input:
        # Preprocessing outputs
        "results/processed_balesdent_2018.csv",
        "results/turnover_14C_data.csv",
        
        # Calibration outputs  
        "results/powerlaw_model_params.csv",
        "results/lognormal_model_params.csv",
        "results/generalized_powerlaw_params.csv",
        "results/gamma_model_params.csv",
        
        # Prediction outputs
        "results/model_predictions.csv",
        "results/lognormal_predictions.csv",
        
        # Plotting outputs
        "results/figures/results_plots.png",
        
        # Sensitivity analysis outputs
        "results/sensitivity_powerlaw.csv",
        "results/sensitivity_lognormal.csv"

# Step 01: Preprocess Balesdent data
rule preprocess_balesdent_all:
    output:
        "results/processed_balesdent_2018.csv"
    script:
        "notebooks/01_preprocess_balesdent_data_all.py"

rule preprocess_balesdent:
    output:
        "results/processed_balesdent_subset.csv"
    script:
        "notebooks/01_preprocess_balesdent_data.py"

# Step 02: Get turnover data from 14C
rule turnover_14C_all:
    input:
        "results/processed_balesdent_2018.csv"
    output:
        "results/turnover_14C_data.csv"
    script:
        "notebooks/02_get_turnover_14C_all.py"

rule turnover_14C:
    input:
        "results/processed_balesdent_subset.csv"
    output:
        "results/turnover_14C_subset.csv"
    script:
        "notebooks/02_get_turnover_14C.py"

# Step 03a: Calibrate power law models
rule calibrate_powerlaw_all:
    input:
        "results/turnover_14C_data.csv"
    output:
        "results/powerlaw_model_params.csv"
    script:
        "notebooks/03a_calibrate_powerlaw_model_all.py"

rule calibrate_powerlaw:
    input:
        "results/turnover_14C_subset.csv"
    output:
        "results/powerlaw_model_params_subset.csv"
    script:
        "notebooks/03a_calibrate_powerlaw_model.py"

# Step 03b: Calibrate lognormal models (Python)
rule calibrate_lognormal:
    input:
        "results/turnover_14C_data.csv"
    output:
        "results/lognormal_model_params.csv"
    script:
        "notebooks/03b_calibrate_lognormal_model.py"

# Step 03b: Calibrate lognormal models (Mathematica notebooks)
rule calibrate_lognormal_nb:
    input:
        "results/turnover_14C_data.csv"
    output:
        "results/lognormal_nb_results.csv"
    shell:
        """
        # Note: Mathematica notebooks (.nb files) need to be run in Mathematica
        # This is a placeholder - actual execution depends on your Mathematica setup
        echo "Mathematica notebook execution for lognormal calibration"
        touch {output}
        """

# Step 03c: Calibrate generalized power law model
rule calibrate_generalized_powerlaw:
    input:
        "results/turnover_14C_data.csv"
    output:
        "results/generalized_powerlaw_params.csv"
    script:
        "notebooks/03c_calibrate_generalized_powerlaw_model.py"

# Step 03d: Calibrate gamma model
rule calibrate_gamma:
    input:
        "results/turnover_14C_data.csv"
    output:
        "results/gamma_model_params.csv"
    script:
        "notebooks/03d_calibrate_gamma_model.py"

# Step 04: Model predictions
rule model_predictions_all:
    input:
        "results/powerlaw_model_params.csv",
        "results/lognormal_model_params.csv",
        "results/generalized_powerlaw_params.csv",
        "results/gamma_model_params.csv"
    output:
        "results/model_predictions.csv"
    script:
        "notebooks/04_model_predictions_all.py"

rule model_predictions:
    input:
        "results/powerlaw_model_params_subset.csv",
        "results/lognormal_model_params.csv"
    output:
        "results/model_predictions_subset.csv"
    script:
        "notebooks/04_model_predictions.py"

# Step 04b: Lognormal predictions (Julia)
rule lognormal_predictions_julia:
    input:
        "results/lognormal_model_params.csv"
    output:
        "results/lognormal_predictions.csv"
    shell:
        """
        julia notebooks/04b_lognormal_predictions.jl
        """

# Step 05: Plot results
rule plot_results_v2_all:
    input:
        "results/model_predictions.csv"
    output:
        "results/figures/results_plots.png"
    script:
        "notebooks/05_plot_results_v2_all.py"

rule plot_results_v2:
    input:
        "results/model_predictions_subset.csv"
    output:
        "results/figures/results_plots_subset.png"
    script:
        "notebooks/05_plot_results_v2.py"

rule plot_results:
    input:
        "results/model_predictions.csv"
    output:
        "results/figures/basic_plots.png"
    script:
        "notebooks/05_plot_results.py"

# Step 06: Sensitivity analysis
rule sensitivity_analysis_old:
    input:
        "results/model_predictions.csv"
    output:
        "results/sensitivity_old_results.html"
    shell:
        """
        jupyter nbconvert --to html --execute notebooks/06_sensitivity_analysis_old.ipynb
        mv notebooks/06_sensitivity_analysis_old.html {output}
        """

rule sensitivity_analysis_powerlaw:
    input:
        "results/powerlaw_model_params.csv"
    output:
        "results/sensitivity_powerlaw.csv"
    shell:
        """
        jupyter nbconvert --to html --execute notebooks/06a_sensitivity_analysis_powerlaw.ipynb
        touch {output}
        """

rule sensitivity_analysis_lognormal:
    input:
        "results/lognormal_model_params.csv"
    output:
        "results/sensitivity_lognormal.csv"
    shell:
        """
        jupyter nbconvert --to html --execute notebooks/06b_sensitivity_analysis_lognormal.ipynb
        touch {output}
        """

rule sensitivity_analysis_lognormal_julia:
    input:
        "results/lognormal_model_params.csv"
    output:
        "results/sensitivity_lognormal_julia.csv"
    shell:
        """
        julia notebooks/06b_sensitivity_analysis_lognormal.jl
        """

# Create results directories
rule create_dirs:
    output:
        directory("results/figures")
    shell:
        "mkdir -p results/figures"

# Clean up rule
rule clean:
    shell:
        """
        rm -rf results/*.csv
        rm -rf results/figures/*
        rm -rf results/*.html
        """
