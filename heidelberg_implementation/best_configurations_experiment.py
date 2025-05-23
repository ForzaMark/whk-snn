from neural_nets.configurable_spiking_neural_net import ConfigurableSpikingNeuralNet
from training.train_simplified_snn import train_simplified_snn
from constants import NUMBER_INPUT_NEURONS, NUMBER_HIDDEN_NEURONS, NUMBER_OUTPUT_NEURONS, BETA, TIME_STEPS, NUMBER_HIDDEN_LAYERS

best_sparsity = 0
best_number_hidden_layer = 2
best_number_hidden_neurons = 1000
num_epochs = 30

best_sparsity_model = ConfigurableSpikingNeuralNet(number_input_neurons=NUMBER_INPUT_NEURONS,
                                                 number_hidden_neurons=NUMBER_HIDDEN_NEURONS,
                                                 number_output_neurons=NUMBER_OUTPUT_NEURONS,
                                                 beta=BETA,
                                                 time_steps=TIME_STEPS,
                                                 number_hidden_layers=NUMBER_HIDDEN_LAYERS,
                                                 sparsity=0)

best_number_hidden_layer_model = ConfigurableSpikingNeuralNet(number_input_neurons=NUMBER_INPUT_NEURONS,
                                                 number_hidden_neurons=NUMBER_HIDDEN_NEURONS,
                                                 number_output_neurons=NUMBER_OUTPUT_NEURONS,
                                                 beta=BETA,
                                                 time_steps=TIME_STEPS,
                                                 number_hidden_layers=best_number_hidden_layer,
                                                 sparsity=0)

best_number_hidden_neurons_model = ConfigurableSpikingNeuralNet(number_input_neurons=NUMBER_INPUT_NEURONS,
                                                 number_hidden_neurons=best_number_hidden_neurons,
                                                 number_output_neurons=NUMBER_OUTPUT_NEURONS,
                                                 beta=BETA,
                                                 time_steps=TIME_STEPS,
                                                 number_hidden_layers=NUMBER_HIDDEN_LAYERS,
                                                 sparsity=0)

if __name__ == '__main__':
    train_simplified_snn(best_sparsity_model, 
                        num_epochs=num_epochs, 
                        save_model='./models/best_sparsity', 
                        save_plots='./output/experiments_sparsity/best_sparsity', 
                        additional_output_information={
                            'num_hidden_neurons': NUMBER_HIDDEN_NEURONS,
                            'num_hidden_layer': NUMBER_HIDDEN_LAYERS,
                            'sparsity': best_sparsity
                        },
                        output_file_path='./output/experiments_sparsity/best_sparsity.json')

    train_simplified_snn(best_number_hidden_layer_model, 
                        num_epochs=num_epochs, 
                        save_model='./models/best_number_hidden_layer', 
                        save_plots='./output/experiments_multiple_hidden_layer/best_number_hidden_layer', 
                        additional_output_information={
                            'num_hidden_neurons': NUMBER_HIDDEN_NEURONS,
                            'num_hidden_layer': best_number_hidden_layer,
                            'sparsity': 0
                        },
                        output_file_path='./output/experiments_multiple_hidden_layer/best_number_hidden_layer.json')

    train_simplified_snn(best_number_hidden_neurons_model, 
                        num_epochs=num_epochs, 
                        save_model='./models/best_number_hidden_neurons', 
                        save_plots='./output/experiments_number_hidden_neurons/best_number_hidden_neurons', 
                        additional_output_information={
                            'num_hidden_layer': NUMBER_HIDDEN_LAYERS,
                            'num_hidden_neurons': best_number_hidden_neurons,
                            'sparsity': 0
                        },
                        output_file_path='./output/experiments_number_hidden_neurons/best_number_hidden_neurons.json')
