from constants import (
    BETA,
    NUMBER_INPUT_NEURONS,
    NUMBER_OUTPUT_NEURONS,
    THRESHOLD,
    TIME_STEPS,
)
from neural_nets.configurable_spiking_neural_net import ConfigurableSpikingNeuralNet
from training.train_snn import train_snn

num_epochs = 500
sparsity = 0
number_hidden_neurons = 1000
number_hidden_layer = 1

model = ConfigurableSpikingNeuralNet(
    number_input_neurons=NUMBER_INPUT_NEURONS,
    number_hidden_neurons=number_hidden_neurons,
    number_output_neurons=NUMBER_OUTPUT_NEURONS,
    beta=BETA,
    threshold=THRESHOLD,
    time_steps=TIME_STEPS,
    number_hidden_layers=number_hidden_layer,
)

train_snn(
    model,
    num_epochs=num_epochs,
    sparsity=sparsity,
    additional_output_information={
        "number_hidden_neurons": int(number_hidden_neurons),
        "number_hidden_layer": int(number_hidden_layer),
    },
    save_model="./models/experiment_point_of_chaos/500_epochs",
    save_model_per_epoch="./models/experiment_point_of_chaos/500_epochs",
    loss_configuration="membrane_potential_cross_entropy",
    save_plots="./output/experiments_point_of_chaos/500_epochs",
    output_file_path=f"./output/experiments_point_of_chaos/500_epochs",
)
