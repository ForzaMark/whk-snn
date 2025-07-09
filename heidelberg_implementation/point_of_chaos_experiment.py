from constants import NUMBER_INPUT_NEURONS, NUMBER_OUTPUT_NEURONS, TIME_STEPS
from neural_nets.configurable_spiking_neural_net import ConfigurableSpikingNeuralNet
from training.train_snn import train_snn

if __name__ == '__main__':
    NUMBER_HIDDEN_NEURONS = 3000
    NUMBER_HIDDEN_LAYER = 2
    beta = 0.8
    threshold = 1

    base_net =  ConfigurableSpikingNeuralNet(number_input_neurons=NUMBER_INPUT_NEURONS,
                                                        number_hidden_neurons=NUMBER_HIDDEN_NEURONS,
                                                        number_output_neurons=NUMBER_OUTPUT_NEURONS,
                                                        beta=beta,
                                                        threshold=threshold,
                                                        time_steps=TIME_STEPS,
                                                        number_hidden_layers=NUMBER_HIDDEN_LAYER)

    models = []

    sparsity = 0.95

    train_snn(base_net, 
                num_epochs=100, 
                sparsity=sparsity,
                save_model_per_epoch=f'./models/experiment_point_of_chaos/best_grid_search_beta_{beta}_threshold_{threshold}',
                additional_output_information={'sparsity': sparsity}, 
                output_file_path=f'./output/experiments_sparsity/sparsity_{sparsity}.json')