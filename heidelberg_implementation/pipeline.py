# TODO: setup pipeline and interfaces
# TODO: create plots
# TODO: vary weight initialization and average results + std
# TODO: unify environments
# TODO: how to portability

from machine_learning.run_logistic_regression import run_logistic_regression
from machine_learning.run_svm import run_svm
from util.create_data_loader import create_data_loader

if __name__ == "__main__":
    train_data_loader, test_data_loader = create_data_loader(use_train_subset=1000)

    svm_acc = run_svm(train_data_loader, test_data_loader)
    logistic_regression_acc = run_logistic_regression(
        train_data_loader, test_data_loader
    )

    print(svm_acc)
    print(logistic_regression_acc)


# run_svm
# run_logistic_regression
# run_lstm
# run_cnn
# run_best_snn_architectures
# run_e_prop_lsnn
# run_e_prop_lstm
