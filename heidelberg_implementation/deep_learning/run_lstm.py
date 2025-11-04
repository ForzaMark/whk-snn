import torch
import torch.nn as nn
from constants import HEIDELBERG_DATASET_NUMBER_CLASSES
from neural_nets.lstm_classifier import LSTMClassifier
from util.remove_training_from_memory import remove_training_from_memory

from .calculate_test_accuracy import calculate_test_accuracy

preprocess_batches_lstm = lambda X_batch: X_batch.permute(1, 0, 2).contiguous()

loss_function = nn.CrossEntropyLoss()


def run_lstm(train_data_loader, test_data_loader, num_epochs=30):
    model = LSTMClassifier(num_classes=HEIDELBERG_DATASET_NUMBER_CLASSES)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for i in range(num_epochs):
        print(f"LSTM epoch {i} of {num_epochs}")
        loss = torch.zeros(1)

        for X_batch, y_batch in train_data_loader:

            X_batch = preprocess_batches_lstm(X_batch)

            output = model(X_batch)
            loss = loss_function(output, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    test_acc = calculate_test_accuracy(
        test_data_loader=test_data_loader,
        preprocess_batches=preprocess_batches_lstm,
        model=model,
    )

    remove_training_from_memory(model, optimizer)

    return test_acc
