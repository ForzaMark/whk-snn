
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from neural_nets.lstm_classifier import LSTMClassifier
from tonic import datasets
from torch.utils.data import DataLoader, TensorDataset


def convert_to_time_binned_sequences(data):
    X = []
    Y = []

    for i, (spikes, label) in enumerate(data):
        sequences = []
        current_i = 0

        while current_i < 1400000:
            filtered_spikes = spikes[(spikes['t'] > current_i) & (spikes['t'] <= current_i + 10000)]

            sequence = np.zeros(700)
            for neuron, count in Counter(filtered_spikes['x']).items():
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
    loader = DataLoader(dataset, batch_size=1)

    return loader

def load_lstm_train_test_data():
    train_data = datasets.SHD("./data", train=True)
    test_data = datasets.SHD("./data", train=False)

    x_train, y_train = convert_to_time_binned_sequences(train_data)
    x_test, y_test = convert_to_time_binned_sequences(test_data)

    train_loader = create_data_loader(x_train, y_train)
    test_loader = create_data_loader(x_test, y_test)

    return train_loader, test_loader 
   
def calculate_test_accuracy(test_data_loader):
    test_correct = 0
    test_total = 0

    for X_batch, y_batch in test_data_loader:
        X_batch = X_batch.permute(1, 0, 2).contiguous()

        output = model(X_batch)

        _, predicted = torch.max(output, 1)
        test_correct += (predicted == y_batch).sum().item()
        test_total += y_batch.size(0)

    return test_correct / test_total

if __name__ == '__main__':
    train_loader, test_loader = load_lstm_train_test_data()
    print('DATA LOADED')

    model = LSTMClassifier(num_classes=20)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(5):
        correct = 0
        total = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.permute(1, 0, 2).contiguous()

            output = model(X_batch)

            loss = loss_function(output, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(output, 1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)

        epoch_acc = correct / total

        test_acc = calculate_test_accuracy(test_data_loader=test_loader)

        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Train Acc: {epoch_acc}, Test Acc: {test_acc}")


