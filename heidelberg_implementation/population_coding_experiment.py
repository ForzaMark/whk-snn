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

num_epochs = 'early_stopping'
sparsity = 0

LOSS_CONFIGURATIONS = [
    "membrane_potential_cross_entropy", 
    "rate_code_cross_entropy"
]

for loss_config in LOSS_CONFIGURATIONS:
    print(f'loss {loss_config}')
    model = ConfigurableSpikingNeuralNet(number_input_neurons=NUMBER_INPUT_NEURONS, 
                                         number_hidden_neurons=3000, 
                                         number_output_neurons=NUMBER_OUTPUT_NEURONS, 
                                         beta=BETA, 
                                         threshold=THRESHOLD,
                                         time_steps=TIME_STEPS, 
                                         number_hidden_layers=2,
                                         sparsity=0)
    
    train_snn(model, 
                num_epochs=num_epochs, 
                loss_configuration=loss_config,
                additional_output_information={'loss_config': loss_config},
                output_file_path=f'./output/experiments_population_coding/best_grid_search_{loss_config}.json',
                save_model=f'./models/experiments_population_coding/best_grid_search_{loss_config}',
                save_plots=f'./output/experiments_population_coding/best_grid_search_{loss_config}')

