from constants import (
    BETA,
    NUMBER_HIDDEN_NEURONS,
    NUMBER_INPUT_NEURONS,
    NUMBER_OUTPUT_NEURONS,
    THRESHOLD,
    TIME_STEPS,
)
from neural_nets.configurable_spiking_neural_net import ConfigurableSpikingNeuralNet
from training.train_snn import train_snn

NUM_HIDDEN_LAYERS = [
    1,
    2,
    3,
    4
]

num_epochs = 30
sparsity = 0

for number_hidden_layer in NUM_HIDDEN_LAYERS:
    print(f'Number hidden layer = {number_hidden_layer}')
    model = ConfigurableSpikingNeuralNet(number_input_neurons=NUMBER_INPUT_NEURONS, 
                                         number_hidden_neurons=NUMBER_HIDDEN_NEURONS, 
                                         number_output_neurons=NUMBER_OUTPUT_NEURONS, 
                                         beta=BETA, 
                                         threshold=THRESHOLD,
                                         time_steps=TIME_STEPS, 
                                         number_hidden_layers=number_hidden_layer,
                                         sparsity=0)
    
    train_snn(model, 
                         num_epochs=num_epochs, 
                         additional_output_information={'number_hidden_layer': number_hidden_layer},
                         output_file_path=f'./output/experiments_multiple_hidden_layer/number_hidden_layer_{number_hidden_layer}.json')

