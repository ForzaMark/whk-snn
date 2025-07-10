from constants import NUMBER_INPUT_NEURONS, NUMBER_OUTPUT_NEURONS, TIME_STEPS
from neural_nets.configurable_spiking_neural_net import ConfigurableSpikingNeuralNet
from training.train_snn import train_snn

NUMBER_HIDDEN_NEURONS = 3000
NUMBER_HIDDEN_LAYER = 2

BETAS = [0.6, 0.8, 0.99]

sparsity = 0

THRESHOLDS = [0.7, 1]

if __name__ == '__main__':
    for threshold in THRESHOLDS:
        for beta in BETAS:
            base_net =  ConfigurableSpikingNeuralNet(number_input_neurons=NUMBER_INPUT_NEURONS,
                                                                number_hidden_neurons=NUMBER_HIDDEN_NEURONS,
                                                                number_output_neurons=NUMBER_OUTPUT_NEURONS,
                                                                beta=beta,
                                                                threshold=threshold,
                                                                time_steps=TIME_STEPS,
                                                                number_hidden_layers=NUMBER_HIDDEN_LAYER)

            train_snn(base_net, 
                        num_epochs=3, 
                        sparsity=sparsity,
                        save_model=f'./models/experiment_point_of_chaos/3_epochs_beta_{beta}_threshold_{threshold}',
                        additional_output_information={'sparsity': sparsity, 'number_hidden_neurons': NUMBER_HIDDEN_NEURONS, 'number_hidden_layer': NUMBER_HIDDEN_LAYER, 'beta': beta, 'threshold': threshold}, 
                        output_file_path=f'./output/experiments_point_of_chaos/3_epochs_beta_{beta}_threshold_{threshold}.json')