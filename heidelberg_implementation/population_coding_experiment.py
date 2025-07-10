from constants import NUMBER_INPUT_NEURONS, NUMBER_OUTPUT_NEURONS, THRESHOLD, TIME_STEPS
from neural_nets.configurable_spiking_neural_net import ConfigurableSpikingNeuralNet
from training.train_snn import train_snn

num_epochs = 'early_stopping'
sparsity = 0
threshold = 1
BETAS = [0.99, 0.8]
NUMBER_HIDDEN_LAYERS = [1,2]
NUMBER_HIDDEN_NEURONS = [1000, 3000]

for beta in BETAS:
    for number_hidden_layer in NUMBER_HIDDEN_LAYERS:
        for number_hidden_neurons in NUMBER_HIDDEN_NEURONS:
            model = ConfigurableSpikingNeuralNet(number_input_neurons=NUMBER_INPUT_NEURONS, 
                                                    number_hidden_neurons=number_hidden_neurons, 
                                                    number_hidden_layers=number_hidden_layer,
                                                    number_output_neurons=NUMBER_OUTPUT_NEURONS * 500, 
                                                    beta=beta, 
                                                    threshold=THRESHOLD,
                                                    time_steps=TIME_STEPS)

            train_snn(model, 
                    num_epochs=num_epochs, 
                    loss_configuration="population_coding",
                    additional_output_information={
                        'loss_config': "population_coding",
                        'beta': beta,
                        'number_hidden_layer': number_hidden_layer,
                        'number_hidden_neurons': number_hidden_neurons
                    },
                    output_file_path=f'./output/experiments_population_coding/population_coding_beta_{beta}_hidden_layer_{number_hidden_layer}_hidden_neurons_{number_hidden_neurons}.json',
                    save_model=f'./models/experiments_population_coding/population_coding_beta_{beta}_hidden_layer_{number_hidden_layer}_hidden_neurons_{number_hidden_neurons}',
                    save_plots=f'./output/experiments_population_coding/population_coding_beta_{beta}_hidden_layer_{number_hidden_layer}_hidden_neurons_{number_hidden_neurons}'
            )

