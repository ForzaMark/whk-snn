import numpy as np
from constants import NUMBER_INPUT_NEURONS, NUMBER_OUTPUT_NEURONS, TIME_STEPS
from neural_nets.configurable_spiking_neural_net import ConfigurableSpikingNeuralNet
from training.train_snn import train_snn


def run_snn(
    train_data_loader,
    test_data_loader,
    number_hidden_neurons,
    number_hidden_layer,
    beta,
    threshold,
    num_epochs,
):
    model = ConfigurableSpikingNeuralNet(
        number_input_neurons=NUMBER_INPUT_NEURONS,
        number_hidden_neurons=number_hidden_neurons,
        number_output_neurons=NUMBER_OUTPUT_NEURONS,
        beta=beta,
        threshold=threshold,
        time_steps=TIME_STEPS,
        number_hidden_layers=number_hidden_layer,
    )

    (
        training_acc_history,
        test_acc_history,
        loss_history,
        total_training_time,
        epoch_loss_per_time_step,
        models_per_epoch,
    ) = train_snn(
        model,
        train_data_loader=train_data_loader,
        test_data_loader=test_data_loader,
        num_epochs=num_epochs,
        sparsity=0,
    )

    return np.max(test_acc_history)
