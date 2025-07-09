from constants import (
    BETA,
    NUMBER_INPUT_NEURONS,
    NUMBER_OUTPUT_NEURONS,
    THRESHOLD,
    TIME_STEPS,
)
from neural_nets.configurable_spiking_neural_net import ConfigurableSpikingNeuralNet
from training.train_snn import train_snn

num_epochs = 'early_stopping'
sparsity = 0

model = ConfigurableSpikingNeuralNet(number_input_neurons=NUMBER_INPUT_NEURONS, 
                                        number_hidden_neurons=3000, 
                                        number_output_neurons=NUMBER_OUTPUT_NEURONS * 500, 
                                        beta=BETA, 
                                        threshold=THRESHOLD,
                                        time_steps=TIME_STEPS, 
                                        number_hidden_layers=2)

train_snn(model, 
            num_epochs=num_epochs, 
            loss_configuration="population_coding",
            additional_output_information={'loss_config': "population_coding"},
            output_file_path=f'./output/experiments_population_coding/best_grid_search_population_coding.json',
            save_model=f'./models/experiments_population_coding/best_grid_search_population_coding',
            save_plots=f'./output/experiments_population_coding/best_grid_search_population_coding')

