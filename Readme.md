# Reproduction steps

- create conda env `conda env create -f environment.yml` (if this does not work use `minimal_environment.yml`)
- activate conda env `conda activate spiking_simulations`
- download and process datasets: `python setup_datasets.py` 
- `cd heidelberg_implementation`
- run pipeline `python pipeline.py`