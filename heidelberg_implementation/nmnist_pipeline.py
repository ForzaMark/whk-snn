try:
    import importlib.metadata as importlib_metadata

    print("try")
except ModuleNotFoundError:
    import sys

    import importlib_metadata

    sys.modules["importlib.metadata"] = importlib_metadata
    print("except")

import time

import matplotlib.pyplot as plt
import numpy as np
from machine_learning.run_logistic_regression import run_logistic_regression
from machine_learning.run_svm import run_svm
from spiking_neural_networks.run_snn import run_snn
from util.get_datasets import get_nmnist_dataset


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


def print_elapsed_time(start, end):

    elapsed = end - start

    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = elapsed % 60

    print(f"Elapsed time: {hours}h {minutes}m {seconds:.2f}s")


if __name__ == "__main__":
    start = time.time()
    results = {}

    print("######### Loading data #########")
    (
        train_data_loader,
        test_data_loader,
    ) = get_nmnist_dataset(use_train_subset=500)

    print("######### SVM #########")
    svm_acc = run_svm(train_data_loader, test_data_loader)
    results["svm"] = svm_acc

    print("######### Logistic Regression #########")
    logistic_regression_acc = run_logistic_regression(
        train_data_loader, test_data_loader
    )
    results["logistic_regression"] = logistic_regression_acc

    averaged_snn_acc_different_parameter_initialization = []
    for i in range(3):
        print(f"####### SNN {i}/3 #######")
        snn_acc = run_snn(
            train_data_loader,
            test_data_loader,
            number_input_neurons=1156,
            number_output_neurons=20,
            number_hidden_neurons=3000,
            number_hidden_layer=2,
            beta=0.99,
            threshold=1,
            num_epochs=3,
        )
        averaged_snn_acc_different_parameter_initialization.append(snn_acc)
    snn_key = f"snn\n2 layer\n3000 neurons"
    results[snn_key] = {
        "mean": np.mean(averaged_snn_acc_different_parameter_initialization),
        "std": np.std(averaged_snn_acc_different_parameter_initialization),
    }

    print("Results", results)

    plot_model_results(results)
    end = time.time()

    print_elapsed_time(start, end)
