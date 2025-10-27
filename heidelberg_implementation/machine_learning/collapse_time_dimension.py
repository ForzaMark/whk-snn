
def collapse_time_dimension(data_loader):
    X = []
    Y = []

    for x, y in data_loader:

        for batch_item in range(x.size(0)):
            item = x[batch_item]
            y_item = y[batch_item]
            item = item.squeeze(1)

            feature_vector = []

            for neuron_idx in range(item.size(1)):
                neuron_values = item[:, neuron_idx].sum().item()
                feature_vector.append(neuron_values)

            X.append(feature_vector)
            Y.append(y_item)

    return X, Y
