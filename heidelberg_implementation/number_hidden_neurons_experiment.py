from neural_nets.configurable_spiking_neural_net import ConfigurableSpikingNeuralNet
import numpy as np
from constants import NUMBER_INPUT_NEURONS, NUMBER_OUTPUT_NEURONS, BETA, TIME_STEPS, NUMBER_HIDDEN_LAYERS, THRESHOLD
from training.train_simplified_snn import train_simplified_snn

NUM_HIDDEN_NEURONS = np.arange(500, 5500, 500)

num_epochs = 30
sparsity = 0

for number_hidden_neurons in NUM_HIDDEN_NEURONS:
    print(f'Number hidden neurons = {number_hidden_neurons}')
    model = ConfigurableSpikingNeuralNet(number_input_neurons=NUMBER_INPUT_NEURONS, 
                                         number_hidden_neurons=number_hidden_neurons,
                                         number_output_neurons=NUMBER_OUTPUT_NEURONS, 
                                         beta=BETA,
                                         threshold=THRESHOLD,
                                         time_steps=TIME_STEPS,
                                         number_hidden_layers=NUMBER_HIDDEN_LAYERS,
                                         sparsity=0)

    train_simplified_snn(model, 
                        num_epochs=num_epochs,
                        save_model=False,
                        save_plots=False,
                        additional_output_information={'number_hidden_neurons': int(number_hidden_neurons)},
                        output_file_path=f'./output/experiments_number_hidden_neurons/number_hidden_neurons_{number_hidden_neurons}.json')



