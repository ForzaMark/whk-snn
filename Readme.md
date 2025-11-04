# Reproduction steps

- create conda env `conda env create -f environment.yml` (if this does not work use `minimal_environment.yml`)
- download and process datasets: `setup_datasets.ipynb` 
- activate conda env `conda activate spiking_simulations`
- `cd heidelberg_implementation`
- run pipeline `python pipeline.py`