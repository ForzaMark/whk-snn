from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from constants import HEIDELBERG_DATASET_NUMBER_CLASSES
from neural_nets.cnn_classifier import CNN_Classifier
from neural_nets.lstm_classifier import LSTMClassifier
from tonic import datasets
from torch.utils.data import DataLoader, TensorDataset
from util.save_plots import save_history_plot


def convert_to_time_binned_sequences(data):
    X = []
    Y = []

    for i, (spikes, label) in enumerate(data):
        sequences = []
        current_i = 0

        while current_i < 1400000:
            filtered_spikes = spikes[
                (spikes["t"] > current_i) & (spikes["t"] <= current_i + 10000)
            ]

            sequence = np.zeros(700)
            for neuron, count in Counter(filtered_spikes["x"]).items():
                sequence[neuron] = count

            current_i = current_i + 10000
            sequences.append(sequence)

        X.append(sequences)
        Y.append(label)

    return np.array(X), np.array(Y)


def create_data_loader(X, Y):
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(Y, dtype=torch.long)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32)

    return loader


def load_lstm_train_test_data():
    train_data = datasets.SHD("./data", train=True)
    test_data = datasets.SHD("./data", train=False)

    x_train, y_train = convert_to_time_binned_sequences(train_data)
    x_test, y_test = convert_to_time_binned_sequences(test_data)

    train_loader = create_data_loader(x_train, y_train)
    test_loader = create_data_loader(x_test, y_test)

    return train_loader, test_loader


def bin_features_into_64_space_bins(time_binned_sequences):
    space_binned_sequences = []
    for sequence in time_binned_sequences:
        space_binned_sequence = []
        for neurons in sequence:
            assert len(neurons) == 700

            bins = np.array_split(neurons, 64)

            bin_sums = np.array([bin.sum() for bin in bins])
            assert len(bin_sums) == 64

            space_binned_sequence.append(bin_sums)
        space_binned_sequences.append(np.array(space_binned_sequence))

    return np.array(space_binned_sequences)


def load_cnn_train_test_data():
    train_data = datasets.SHD("./data", train=True)
    test_data = datasets.SHD("./data", train=False)

    x_train, y_train = convert_to_time_binned_sequences(train_data)
    x_test, y_test = convert_to_time_binned_sequences(test_data)

    x_train = bin_features_into_64_space_bins(x_train)
    x_test = bin_features_into_64_space_bins(x_test)

    train_loader = create_data_loader(x_train, y_train)
    test_loader = create_data_loader(x_test, y_test)

    return train_loader, test_loader


def calculate_test_accuracy(test_data_loader, preprocess_batches):
    test_correct = 0
    test_total = 0

    for X_batch, y_batch in test_data_loader:
        X_batch = preprocess_batches(X_batch)

        output = model(X_batch)

        _, predicted = torch.max(output, 1)
        test_correct += (predicted == y_batch).sum().item()
        test_total += y_batch.size(0)

    return test_correct / test_total


def load_data(mode):
    if mode == "cnn":
        return load_cnn_train_test_data()
    else:
        return load_lstm_train_test_data()


def save_history_plots(results, path):
    train_acc = [result[0] for result in results]
    test_acc = [result[1] for result in results]
    loss = [result[2] for result in results]

    save_history_plot(
        train_acc, f"./output/experiment_deep_models/{path}_train_acc.jpg"
    )
    save_history_plot(test_acc, f"./output/experiment_deep_models/{path}_test_acc.jpg")
    save_history_plot(loss, f"./output/experiment_deep_models/{path}_loss.jpg")


preprocess_batches_lstm = lambda X_batch: X_batch.permute(1, 0, 2).contiguous()

preprocess_batches_cnn = lambda X_batch: X_batch.permute(0, 2, 1).unsqueeze(0)

loss_function = nn.CrossEntropyLoss()
num_epochs = 10

if __name__ == "__main__":

    for mode in ["lstm", "cnn"]:
        model = (
            CNN_Classifier(
                input_channels=1, num_classes=HEIDELBERG_DATASET_NUMBER_CLASSES
            )
            if mode == "cnn"
            else LSTMClassifier(num_classes=HEIDELBERG_DATASET_NUMBER_CLASSES)
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        train_loader, test_loader = load_data(mode)

        preprocess_batches = (
            preprocess_batches_cnn if mode == "cnn" else preprocess_batches_lstm
        )

        results = []

        for epoch in range(num_epochs):
            correct = 0
            total = 0
            loss = torch.zeros(1)

            for X_batch, y_batch in train_loader:
                X_batch = preprocess_batches(X_batch)
                output = model(X_batch)
                print(output.shape)
                print(y_batch.shape)
                print("--------")
                loss = loss_function(output, y_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(output, 1)
                correct += (predicted == y_batch).sum().item()
                total += y_batch.size(0)

            epoch_acc = correct / total

            test_acc = calculate_test_accuracy(
                test_data_loader=test_loader, preprocess_batches=preprocess_batches
            )

            results.append((epoch_acc, test_acc, loss.item()))

            save_history_plots(results, mode)
