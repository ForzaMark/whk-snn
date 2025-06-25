from constants import (
    BETA,
    NUMBER_HIDDEN_LAYERS,
    NUMBER_HIDDEN_NEURONS,
    NUMBER_INPUT_NEURONS,
    NUMBER_OUTPUT_NEURONS,
    THRESHOLD,
    TIME_STEPS,
)
from neural_nets.configurable_spiking_neural_net import ConfigurableSpikingNeuralNet
from training.train_snn import train_snn

best_sparsity = 0
best_number_hidden_layer = 2
best_number_hidden_neurons = 1000
num_epochs = 30

best_sparsity_model = ConfigurableSpikingNeuralNet(number_input_neurons=NUMBER_INPUT_NEURONS,
                                                 number_hidden_neurons=NUMBER_HIDDEN_NEURONS,
                                                 number_output_neurons=NUMBER_OUTPUT_NEURONS,
                                                 beta=BETA,
                                                 threshold=THRESHOLD,
                                                 time_steps=TIME_STEPS,
                                                 number_hidden_layers=NUMBER_HIDDEN_LAYERS,
                                                 sparsity=0)

best_number_hidden_layer_model = ConfigurableSpikingNeuralNet(number_input_neurons=NUMBER_INPUT_NEURONS,
                                                 number_hidden_neurons=NUMBER_HIDDEN_NEURONS,
                                                 number_output_neurons=NUMBER_OUTPUT_NEURONS,
                                                 beta=BETA,
                                                 threshold=THRESHOLD,
                                                 time_steps=TIME_STEPS,
                                                 number_hidden_layers=best_number_hidden_layer,
                                                 sparsity=0)

best_number_hidden_neurons_model = ConfigurableSpikingNeuralNet(number_input_neurons=NUMBER_INPUT_NEURONS,
                                                 number_hidden_neurons=best_number_hidden_neurons,
                                                 number_output_neurons=NUMBER_OUTPUT_NEURONS,
                                                 beta=BETA,
                                                 threshold=THRESHOLD,
                                                 time_steps=TIME_STEPS,
                                                 number_hidden_layers=NUMBER_HIDDEN_LAYERS,
                                                 sparsity=0)

best_grid_search_model = ConfigurableSpikingNeuralNet(number_input_neurons=NUMBER_INPUT_NEURONS,
                                                 number_hidden_neurons=3000,
                                                 number_output_neurons=NUMBER_OUTPUT_NEURONS,
                                                 beta=BETA,
                                                 threshold=THRESHOLD,
                                                 time_steps=TIME_STEPS,
                                                 number_hidden_layers=2,
                                                 sparsity=0)

if __name__ == '__main__':
    train_snn(best_grid_search_model, 
                        num_epochs='early_stopping', 
                        save_model='./models/best_grid_search', 
                        save_plots='./output/experiments_number_hidden_neurons/best_grid_search', 
                        additional_output_information={
                            'num_hidden_layer': 2,
                            'num_hidden_neurons': 3000,
                            'sparsity': 0
                        },
                        output_file_path='./output/experiments_number_hidden_neurons/best_grid_search.json')
