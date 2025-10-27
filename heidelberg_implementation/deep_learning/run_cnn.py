import torch
import torch.nn as nn
from constants import HEIDELBERG_DATASET_NUMBER_CLASSES
from neural_nets.cnn_classifier import CNN_Classifier

from .calculate_test_accuracy import calculate_test_accuracy

preprocess_batches_cnn = lambda X_batch: X_batch.permute(0, 2, 1).unsqueeze(0)

loss_function = nn.CrossEntropyLoss()


def run_cnn(train_data_loader, test_data_loader, num_epochs=10):
    model = CNN_Classifier(
        input_channels=1, num_classes=HEIDELBERG_DATASET_NUMBER_CLASSES
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    preprocess_batches = preprocess_batches_cnn

    for i in range(num_epochs):
        print(f"CNN epoch {i} of {num_epochs}")
        loss = torch.zeros(1)

        for X_batch, y_batch in train_data_loader:

            X_batch = preprocess_batches_cnn(X_batch)

            output = model(X_batch)
            loss = loss_function(output, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    test_acc = calculate_test_accuracy(
        test_data_loader=test_data_loader,
        preprocess_batches=preprocess_batches,
        model=model,
    )

    return test_acc
