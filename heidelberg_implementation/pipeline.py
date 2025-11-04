# TODO: how to portability

import matplotlib.pyplot as plt
import numpy as np
from deep_learning.run_cnn import run_cnn
from deep_learning.run_lstm import run_lstm
from eprop.solve_hdd_with_lsnn import run_eprop_lsnn
from eprop.solve_hdd_with_lstm import run_eprop_lstm
from machine_learning.run_logistic_regression import run_logistic_regression
from machine_learning.run_svm import run_svm
from spiking_neural_networks.run_snn import run_snn
from util.create_data_loader import create_data_loader, create_data_loader_deep_models


def plot_model_results(results, save_path="./output/experiment_all_methods/result.jpg"):
    models = list(results.keys())
    means = []
    stds = []

    for v in results.values():
        if isinstance(v, dict):
            means.append(v.get("mean", np.nan))
            stds.append(v.get("std", 0))
        else:
            means.append(v)
            stds.append(0)

    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        models,
        means,
        yerr=stds,
        capsize=5,
        alpha=0.8,
        color="skyblue",
        edgecolor="black",
    )

    for bar, mean, std in zip(bars, means, stds):
        yval = bar.get_height()
        label = f"{mean:.2f}"
        if std > 0:
            label += f"±{std:.2f}"
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 0.01,
            label,
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.xlabel("Model")
    plt.ylabel("Test Accuracy")
    plt.title("Test Accuracy on the SHD for Different Models")
    plt.ylim(0, 1)
    plt.xticks(rotation=30)
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, format="jpg", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    results = {}

    print("######### Loading data #########")
    train_data_loader, test_data_loader = create_data_loader("SHD")
    train_data_loader_cnn, test_data_loader_cnn = create_data_loader_deep_models(
        mode="cnn"
    )
    train_data_loader_lstm, test_data_loader_lstm = create_data_loader_deep_models(
        mode="lstm"
    )

    print("######### SVM #########")
    svm_acc = run_svm(train_data_loader, test_data_loader)
    results["svm"] = svm_acc

    print("######### Logistic Regression #########")
    logistic_regression_acc = run_logistic_regression(
        train_data_loader, test_data_loader
    )
    results["logistic_regression"] = logistic_regression_acc

    averaged_cnn_acc_different_parameter_initialization = []
    for i in range(5):
        print(f"####### CNN {i}/5 #######")
        cnn_acc = run_cnn(train_data_loader_cnn, test_data_loader_cnn, num_epochs=30)

        averaged_cnn_acc_different_parameter_initialization.append(cnn_acc)
    results["cnn"] = {
        "mean": np.mean(averaged_cnn_acc_different_parameter_initialization),
        "std": np.std(averaged_cnn_acc_different_parameter_initialization),
    }

    averaged_lstm_acc_different_parameter_initialization = []
    for i in range(5):
        print(f"####### LSTM {i}/5 #######")
        lstm_acc = run_lstm(
            train_data_loader_lstm, test_data_loader_lstm, num_epochs=30
        )
        averaged_lstm_acc_different_parameter_initialization.append(lstm_acc)
    results["lstm"] = {
        "mean": np.mean(averaged_lstm_acc_different_parameter_initialization),
        "std": np.std(averaged_lstm_acc_different_parameter_initialization),
    }

    averaged_snn_acc_different_parameter_initialization = []
    for i in range(5):
        print(f"####### SNN {i}/5 #######")
        snn_acc = run_snn(
            train_data_loader,
            test_data_loader,
            number_hidden_neurons=3000,
            number_hidden_layer=2,
            beta=0.99,
            threshold=1,
            num_epochs=30,
        )
        averaged_snn_acc_different_parameter_initialization.append(snn_acc)
    snn_key = f"snn\n2 layer\n3000 neurons"
    results[snn_key] = {
        "mean": np.mean(averaged_snn_acc_different_parameter_initialization),
        "std": np.std(averaged_snn_acc_different_parameter_initialization),
    }

    print("######### E-Prop LSNN #########")
    eprop_lsnn_acc = run_eprop_lsnn()
    results["eprop_lsnn"] = eprop_lsnn_acc

    print("######### E-Prop LSTM #########")
    eprop_lstm_acc = run_eprop_lstm()
    results["eprop_lstm"] = eprop_lstm_acc

    print("Results", results)

    plot_model_results(results)
