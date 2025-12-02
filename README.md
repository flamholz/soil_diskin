# soil_diskin

**Continuum models of soil carbon dynamics**

This repository contains code and data for analyzing soil carbon turnover using continuum models (power law, gamma, lognormal distributions) and compartmental models (CLM5, JSBACH). The analysis integrates radiocarbon (¹⁴C) measurements with soil carbon turnover data to calibrate and compare different modeling approaches.

## Features

- **Continuum Models**: PowerLaw, Gamma, Lognormal, and Generalized PowerLaw distributions for soil carbon age
- **Compartmental Models**: CLM5, JSBACH, and Reduced Complexity models from He. et al 2018. 
- **Data Processing**: Automated pipeline for ¹⁴C atmospheric data and soil incubation datasets
- **Model Calibration**: Optimization routines for fitting models to observed data
- **Visualization**: Publication and presentation figures

## Prerequisites

Before starting, ensure you have:

- **Python 3.11 or higher**
- **Julia 1.11+** - [Download here](https://julialang.org/downloads/)
- **Fortran compiler** (for JSBACH model compilation)
  - Linux: `gfortran` via `sudo apt install gfortran`
  - macOS: `gcc` via [Homebrew](https://brew.sh/) with `brew install gcc`
- **Google Earth Engine account** - [Sign up here](https://earthengine.google.com/signup/)
- **WolframScript** (optional, for Mathematica notebooks) - [Download here](https://www.wolfram.com/wolframscript/)
- **~5 GB disk space** for data and results

## Installation

### 1. Install UV (Python package manager)

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Verify installation
uv --version
```

### 2. Clone the repository

```bash
git clone https://github.com/flamholz/soil_diskin.git
cd soil_diskin
```

### 3. Set up Python environment

```bash
# Create environment and install dependencies
uv sync

# Install the soil_diskin package in development mode
uv pip install -e .

# Verify installation
uv run python -c "import soil_diskin; print('soil_diskin version:', soil_diskin.__version__)"
```

### 4. Set up Julia dependencies

```bash
# Install Julia packages from Project.toml
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Verify installation
julia --project=. -e 'using DataFrames, Distributions; println("Julia setup complete")'
```

### 5. Configure Google Earth Engine

```bash
# Authenticate with Google Earth Engine
uv run python -c "import ee; ee.Authenticate()"

# Follow the prompts to authorize access
# Initialize (you only need to do this once)
uv run python -c "import ee; ee.Initialize()"
```

### 6. Install Fortran compiler

**Linux:**
```bash
sudo apt install gfortran
gfortran --version  # Verify installation
```

**macOS:**
```bash
# First install Homebrew if needed: https://brew.sh/
brew install gcc
gfortran --version  # Verify installation
```

## Quick Start

### Run tests to verify installation

```bash
# Run Python tests
uv run pytest tests/

# Run specific test file
uv run pytest tests/test_models.py -v
```

### Run the analysis pipeline

```bash
# Run entire workflow with 3 cores
uv run snakemake --cores 3

# Run specific output (e.g., Figure 3)
uv run snakemake figures/fig3.png --cores 3

# Force regeneration of specific results
uv run snakemake figures/fig3.png --cores 3 --force
```

Make sure you have an internet connection when you run the pipeline, as it will:
- Download relevant datasets
- Preprocess and merge data sources
- Calibrate all model types
- Generate predictions
- Create publication figures

Note: running the whole pipeline on a M2 MacBook Air takes about 2 days. 

## Citation

If you use this code in your research, please cite:

```
[Citation details to be added]
```

## License

See `LICENSE` file for details.

