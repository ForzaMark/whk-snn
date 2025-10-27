# TODO: setup pipeline and interfaces
# TODO: create plots
# TODO: vary weight initialization and average results + std
# TODO: unify environments
# TODO: how to portability

import matplotlib.pyplot as plt
from deep_learning.run_cnn import run_cnn
from deep_learning.run_lstm import run_lstm
from eprop.solve_hdd_with_lsnn import run_eprop_lsnn
from eprop.solve_hdd_with_lstm import run_eprop_lstm
from machine_learning.run_logistic_regression import run_logistic_regression
from machine_learning.run_svm import run_svm
from spiking_neural_networks.run_snn import run_snn
from util.create_data_loader import create_data_loader, load_train_test_data_deep_models


def create_result_plot(results):
    models = list(results.keys())
    values = list(results.values())

    plt.figure(figsize=(10, 6))
    plt.bar(models, values, color="skyblue")

    plt.xlabel("Model")
    plt.ylabel("Accuracy / Score")
    plt.title("Model Performance Comparison")
    plt.ylim(0, 1)

    plt.xticks(rotation=30)

    plt.savefig(
        "./output/experiment_all_methods/result.jpg",
        format="jpg",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()


if __name__ == "__main__":
    results = {}

    print("######### Loading data #########")
    train_data_loader, test_data_loader = create_data_loader()
    train_data_loader_cnn, test_data_loader_cnn = load_train_test_data_deep_models(
        mode="cnn"
    )
    train_data_loader_lstm, test_data_loader_lstm = load_train_test_data_deep_models(
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

    print("######### CNN #########")
    cnn_acc = run_cnn(train_data_loader_cnn, test_data_loader_cnn)
    results["cnn"] = cnn_acc

    print("######### LSTM #########")
    lstm_acc = run_lstm(train_data_loader_lstm, test_data_loader_lstm)
    results["lstm"] = lstm_acc

    print("######### SNN #########")
    snn_acc = run_snn(
        train_data_loader,
        test_data_loader,
        number_hidden_neurons=3000,
        number_hidden_layer=2,
        beta=0.99,
        threshold=1,
        num_epochs=30,
    )
    results["snn"] = snn_acc

    print("######### E-Prop LSNN #########")
    eprop_lsnn_acc = run_eprop_lsnn()
    results["eprop_lsnn"] = eprop_lsnn_acc

    print("######### E-Prop LSTM #########")
    eprop_lstm_acc = run_eprop_lstm()
    results["eprop_lstm"] = eprop_lstm_acc

    print("Results", results)

    create_result_plot(results)
