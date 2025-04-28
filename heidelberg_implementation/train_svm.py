import numpy as np
import torch
from utils import get_device
from utils import (get_train_test_data, 
                        sparse_data_generator_from_hdf5_spikes)

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from tonic import datasets, transforms

def create_svm_data_representation(data_generator):
    X = []
    Y = []

    for x,y in data_generator:
        sample_vector = []
        x_dense = x.squeeze().to_dense()

        for neuron in range(number_input_neurons):
            summed = x_dense[:, neuron].sum().item()
            sample_vector.append(summed)
        
        X.append(sample_vector)
        Y.append(y.item())

    return X, Y

number_input_neurons  = 700
number_hidden_neurons  = 200
number_output_neurons = 20

number_time_steps = 100
max_time = 1.4

device = get_device()
nb_epochs = 3

frame_transform = transforms.ToFrame(
    sensor_size=datasets.SHD.sensor_size,  
    n_time_bins=number_time_steps
)

if __name__ == "__main__":
    train_data = datasets.SHD("./data", transform=frame_transform, train=True)
    test_data = datasets.SHD("./data", transform=frame_transform, train=False)
    
    print('prepare data')
    train_data_loader = DataLoader(train_data, shuffle=False, batch_size=1)
    test_data_loader = DataLoader(test_data, shuffle=False, batch_size=1)

    x_train, y_train = create_svm_data_representation(train_data_loader)
    x_test, y_test = create_svm_data_representation(test_data_loader)

    print('classification')
    clf = SVC()
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {accuracy:.2f}")
