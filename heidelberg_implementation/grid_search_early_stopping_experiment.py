import numpy as np
from constants import (
    BETA,
    NUMBER_INPUT_NEURONS,
    NUMBER_OUTPUT_NEURONS,
    THRESHOLD,
    TIME_STEPS,
)
from neural_nets.configurable_spiking_neural_net import ConfigurableSpikingNeuralNet
from training.train_snn import train_snn

HIDDEN_NEURONS = np.arange(1000, 4000, 1000)
HIDDEN_LAYERS = [1, 2, 3]

num_epochs = 'early_stopping'
sparsity = 0

for number_hidden_neurons in HIDDEN_NEURONS:
    for number_hidden_layer in HIDDEN_LAYERS:
        print(f'Number hidden neurons = {number_hidden_neurons}')
        print(f'Number hidden layer = {number_hidden_layer}')

        model = ConfigurableSpikingNeuralNet(number_input_neurons=NUMBER_INPUT_NEURONS, 
                                            number_hidden_neurons=number_hidden_neurons,
                                            number_output_neurons=NUMBER_OUTPUT_NEURONS, 
                                            beta=BETA, 
                                            threshold=THRESHOLD,
                                            time_steps=TIME_STEPS,
                                            number_hidden_layers=number_hidden_layer,
                                            sparsity=0)

        train_snn(model, 
                            num_epochs=num_epochs,
                            save_model=False,
                            save_plots=False,
                            additional_output_information={'number_hidden_neurons': int(number_hidden_neurons), 'number_hidden_layer': int(number_hidden_layer)},
                            output_file_path=f'./output/experiments_grid_search_early_stopping/neurons_{number_hidden_neurons}_layer_{number_hidden_layer}.json')



