from neural_nets.recurrent_snn import RSNN
from training.train_snn import train_snn
from util.save_plots import save_history_plot
from util.save_utils import save_best_performing_model

num_epochs = 20
loss_configuration = "rate_code_cross_entropy"
model = RSNN()

if __name__ == "__main__":
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
        loss_configuration=loss_configuration,
    )

    assert len(training_acc_history) == len(test_acc_history) == len(models_per_epoch)

    save_history_plot(
        training_acc_history,
        f"./output/experiment_rsnn/train_acc.jpg",
    )

    save_history_plot(
        test_acc_history,
        f"./output/experiment_rsnn/test_acc.jpg",
    )

    save_history_plot(
        loss_history,
        f"./output/experiment_rsnn/loss.jpg",
    )

    save_best_performing_model(
        models_per_epoch,
        test_acc_history,
        f"./models/experiment_recurrent_snn/recurrent.pth",
    )
