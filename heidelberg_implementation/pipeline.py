# TODO: setup pipeline and interfaces
# TODO: create plots
# TODO: vary weight initialization and average results + std
# TODO: unify environments
# TODO: how to portability

from deep_learning.run_cnn import run_cnn
from deep_learning.run_lstm import run_lstm
from machine_learning.run_logistic_regression import run_logistic_regression
from machine_learning.run_svm import run_svm
from spiking_neural_networks.run_snn import run_snn
from util.create_data_loader import create_data_loader, load_train_test_data_deep_models

if __name__ == "__main__":
    train_data_loader, test_data_loader = create_data_loader(use_train_subset=1000)

    # print("ML")
    # svm_acc = run_svm(train_data_loader, test_data_loader)
    # logistic_regression_acc = run_logistic_regression(
    #     train_data_loader, test_data_loader
    # )

    # print(svm_acc)
    # print(logistic_regression_acc)

    # train_data_loader_cnn, test_data_loader_cnn = load_train_test_data_deep_models(
    #     mode="cnn", use_train_subset=1000
    # )
    # train_data_loader_lstm, test_data_loader_lstm = load_train_test_data_deep_models(
    #     mode="lstm", use_train_subset=1000
    # )

    # print("CNN")
    # cnn_acc = run_cnn(train_data_loader_cnn, test_data_loader_cnn)
    # print(cnn_acc)

    # print("LSTM")
    # lstm_acc = run_lstm(train_data_loader_lstm, test_data_loader_lstm)
    # print(lstm_acc)

    print("SNN")
    snn_acc = run_snn(
        train_data_loader,
        test_data_loader,
        number_hidden_neurons=3000,
        number_hidden_layer=2,
        beta=0.99,
        threshold=1,
        num_epochs=2,
    )
    print(snn_acc)


# run_svm
# run_logistic_regression
# run_lstm
# run_cnn
# run_best_snn_architectures
# run_e_prop_lsnn
# run_e_prop_lstm
