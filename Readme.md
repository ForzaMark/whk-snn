1. If data is not available download using the scripts in get_heidelberg_data.ipynb
2. `conda activate whk-snn`
3. `cd whk-snn/heidelberg_implementation`
4. `python train_snn_heidelberg.py`

# Information on Heidelberg procedure
- they used some model of a human ear to create spikes over time
- sparse_data_generator_from_hdf5_spikes turns the spikes into sparse vector representations (see: https://pytorch.org/docs/stable/sparse.html#sparse-coo-docs)