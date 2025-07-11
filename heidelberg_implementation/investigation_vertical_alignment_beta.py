import numpy as np
from constants import NUMBER_INPUT_NEURONS, NUMBER_OUTPUT_NEURONS, THRESHOLD, TIME_STEPS
from neural_nets.configurable_spiking_neural_net import ConfigurableSpikingNeuralNet
from training.train_snn import train_snn
from util.save_plots import save_history_plot
from util.save_utils import (
    save_all_models_per_epoch,
    save_best_performing_model,
    save_configuration_output,
)

num_epochs = 10
sparsity = 0
number_hidden_neurons = 1000
number_hidden_layer = 2
loss_configuration = "membrane_potential_cross_entropy"

BETAS = [0.6, 0.7, 0.8, 0.99]

if __name__ == "__main__":
    for beta in BETAS:
        print("Beta = ", beta)
        model = ConfigurableSpikingNeuralNet(
            number_input_neurons=NUMBER_INPUT_NEURONS,
            number_hidden_neurons=number_hidden_neurons,
            number_output_neurons=NUMBER_OUTPUT_NEURONS,
            beta=beta,
            threshold=THRESHOLD,
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
            num_epochs=num_epochs,
            sparsity=sparsity,
            loss_configuration=loss_configuration,
            use_train_data_subset=1000,
        )

        assert (
            len(training_acc_history) == len(test_acc_history) == len(models_per_epoch)
        )

        save_best_performing_model(
            models_per_epoch,
            test_acc_history,
            f"./models/experiment_investigation_beta_vertical_alignment/beta_{beta}.pth",
        )

        save_all_models_per_epoch(
            models_per_epoch,
            f"./models/experiment_investigation_beta_vertical_alignment/beta_{beta}",
        )

        result_data = {
            "number_input_neurons": NUMBER_INPUT_NEURONS,
            "number_hidden_neurons": number_hidden_neurons,
            "number_output_neurons": NUMBER_OUTPUT_NEURONS,
            "beta": beta,
            "threshold": THRESHOLD,
            "time_steps": TIME_STEPS,
            "number_hidden_layers": number_hidden_layer,
            "epochs": len(test_acc_history),
            "sparsity": sparsity,
            "loss_configuration": loss_configuration,
            "best_test_accuracy": np.max(test_acc_history),
            "training_accuracy": training_acc_history[np.argmax(test_acc_history)],
        }

        save_configuration_output(
            result_data,
            f"./output/experiment_investigation_beta_vertical_alignment/beta_{beta}.json",
        )

        save_history_plot(
            training_acc_history,
            f"./output/experiment_investigation_beta_vertical_alignment/beta_{beta}_train_acc.jpg",
        )

        save_history_plot(
            test_acc_history,
            f"./output/experiment_investigation_beta_vertical_alignment/beta_{beta}_test_acc.jpg",
        )

        save_history_plot(
            loss_history,
            f"./output/experiment_investigation_beta_vertical_alignment/beta_{beta}_loss.jpg",
        )
