# soil_diskin
disordered kinetics for soil carbon

steps
* Use UV to setup your Python environment

Install UV [] 

`>uv sync`

from the terminal. 

* install the diskin package locally

pip install -e .

* set up your google earth engine account for API access 

* install wolframscript to run mathematica from the command line

* install julia -- describe version
** install julia deps in Project.toml

* run snakemake to run the entire pipeline

`uv run snakemake --cores 3`

this will download all relevant data 
