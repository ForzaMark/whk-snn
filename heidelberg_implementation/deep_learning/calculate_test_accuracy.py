import torch


def calculate_test_accuracy(test_data_loader, preprocess_batches, model):
    test_correct = 0
    test_total = 0

    for X_batch, y_batch in test_data_loader:
        X_batch = preprocess_batches(X_batch)

        output = model(X_batch)

        _, predicted = torch.max(output, 1)
        test_correct += (predicted == y_batch).sum().item()
        test_total += y_batch.size(0)

    return test_correct / test_total
