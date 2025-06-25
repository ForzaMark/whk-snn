from constants import (
    BETA,
    NUMBER_INPUT_NEURONS,
    NUMBER_OUTPUT_NEURONS,
    SPIKE_RATE_ESCAPE_BETA,
    SPIKE_RATE_ESCAPE_SLOPE,
    THRESHOLD,
    TIME_STEPS,
)
from neural_nets.configurable_spiking_neural_net import ConfigurableSpikingNeuralNet
from snntorch import surrogate
from training.train_snn import train_snn

best_sparsity = 0
best_number_hidden_layer = 2
best_number_hidden_neurons = 3000
num_epochs = 2

best_grid_search_model_atan = ConfigurableSpikingNeuralNet(number_input_neurons=NUMBER_INPUT_NEURONS,
                                                 number_hidden_neurons=best_number_hidden_neurons,
                                                 number_output_neurons=NUMBER_OUTPUT_NEURONS,
                                                 beta=BETA,
                                                 threshold=THRESHOLD,
                                                 time_steps=TIME_STEPS,
                                                 number_hidden_layers=best_number_hidden_layer,
                                                 sparsity=best_sparsity)

best_grid_search_model_spike_rate_escape = ConfigurableSpikingNeuralNet(number_input_neurons=NUMBER_INPUT_NEURONS,
                                                 number_hidden_neurons=best_number_hidden_neurons,
                                                 number_output_neurons=NUMBER_OUTPUT_NEURONS,
                                                 beta=BETA,
                                                 threshold=THRESHOLD,
                                                 time_steps=TIME_STEPS,
                                                 number_hidden_layers=best_number_hidden_layer,
                                                 sparsity=best_sparsity,
                                                 surrogate_approximation=surrogate.spike_rate_escape(beta=SPIKE_RATE_ESCAPE_BETA, slope=SPIKE_RATE_ESCAPE_SLOPE)
                                                )

if __name__ == '__main__':
    train_snn(best_grid_search_model_atan, 
                            num_epochs='early_stopping', 
                            save_model='./models/experiment_different_surrogate_approximation/atan', 
                            save_plots='./output/experiment_different_surrogate_approximation/atan.jpg', 
                            additional_output_information={
                                'num_hidden_layer': 2,
                                'num_hidden_neurons': 3000,
                                'sparsity': 0,
                                'surrogate': 'atan'
                            },
                            output_file_path='./output/experiment_different_surrogate_approximation/atan.json')
        

    train_snn(best_grid_search_model_spike_rate_escape, 
                    num_epochs='early_stopping', 
                    save_model='./models/experiment_different_surrogate_approximation/spike_rate_escape', 
                    save_plots='./output/experiment_different_surrogate_approximation/spike_rate_escape.jpg', 
                    additional_output_information={
                        'num_hidden_layer': 2,
                        'num_hidden_neurons': 3000,
                        'sparsity': 0,
                        'surrogate': 'spike_rate_escape'
                    },
                    output_file_path='./output/experiment_different_surrogate_approximation/spike_rate_escape.json')

    