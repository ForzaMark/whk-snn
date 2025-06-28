from itertools import product

from constants import NUMBER_INPUT_NEURONS, NUMBER_OUTPUT_NEURONS, TIME_STEPS
from neural_nets.configurable_spiking_neural_net import ConfigurableSpikingNeuralNet
from training.train_snn import train_snn

BEST_NUMBER_HIDDEN_LAYER = 2
BEST_NUMBER_HIDDEN_NEURONS = 3000
#SPARSITY_PARAMETERS = [0, 0.2, 0.7, 0.95]
SPARSITY_PARAMETERS = [0.2, 0.7, 0.95]

#BETA_PARAMETERS = [0.99, 0.8]
BETA_PARAMETERS = [0.99]

#THRESHOLD_PARAMETERS = [1, 0.7]
THRESHOLD_PARAMETERS = [1]

def create_best_grid_search_model(sparsity, beta, threshold):
    return ConfigurableSpikingNeuralNet(number_input_neurons=NUMBER_INPUT_NEURONS,
                                                 number_hidden_neurons=BEST_NUMBER_HIDDEN_NEURONS,
                                                 number_output_neurons=NUMBER_OUTPUT_NEURONS,
                                                 beta=beta,
                                                 threshold=threshold,
                                                 time_steps=TIME_STEPS,
                                                 number_hidden_layers=BEST_NUMBER_HIDDEN_LAYER)

PARAMETER_COMBINATIONS = list(product(BETA_PARAMETERS, THRESHOLD_PARAMETERS, SPARSITY_PARAMETERS))

ALL_CONFIGURATIONS = [{
        'model':create_best_grid_search_model(sparsity, beta, threshold),
        'sparsity': sparsity,
        'beta': beta,
        'threshold': threshold
    } for beta, threshold, sparsity in PARAMETER_COMBINATIONS]

num_epochs = 'early_stopping'

if __name__ == '__main__':
    for configuration in ALL_CONFIGURATIONS:
        model = configuration["model"]
        sparsity = configuration["sparsity"]
        beta = configuration["beta"]
        threshold = configuration["threshold"]

        print(f'sparsity: {sparsity} | beta: {beta} | threshold: {threshold}')

        train_snn(model, 
                            sparsity=sparsity,
                            num_epochs=num_epochs, 
                            save_model=f'./models/experiment_layer_development_investigation/best_grid_search_sparsity_{sparsity}_beta_{beta}_threshold_{threshold}', 
                            save_plots=f'./output/experiments_layer_development_investigation/best_grid_search_sparsity_{sparsity}_beta_{beta}_threshold_{threshold}', 
                            additional_output_information={
                                'num_hidden_layer': BEST_NUMBER_HIDDEN_LAYER,
                                'num_hidden_neurons': BEST_NUMBER_HIDDEN_NEURONS,
                                'sparsity': sparsity,
                                'beta': beta,
                                'threshold': threshold
                            },
                            output_file_path=f'./output/experiments_layer_development_investigation/best_grid_search_sparsity_{sparsity}_beta_{beta}_threshold_{threshold}.json')
    
    print('DONE')