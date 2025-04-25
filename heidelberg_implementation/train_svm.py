import numpy as np
import torch
from utils import get_device
from utils import (get_train_test_data, 
                        sparse_data_generator_from_hdf5_spikes)

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def create_svm_data_representation(data_generator):
    X = []
    Y = []

    for x,y in data_generator:
        sample_vector = []
        x_dense = x.to_dense()[0]

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

if __name__ == "__main__":
    x_train, y_train, x_test, y_test = get_train_test_data()    

    train_data_generator = sparse_data_generator_from_hdf5_spikes(x_train, y_train,
                                       batch_size=1,
                                       nb_steps=number_time_steps,
                                       nb_units=number_input_neurons,
                                       max_time=max_time,
                                       device=device)
    
    test_data_generator = sparse_data_generator_from_hdf5_spikes(x_test, y_test,
                                       batch_size=1,
                                       nb_steps=number_time_steps,
                                       nb_units=number_input_neurons,
                                       max_time=max_time,
                                       device=device)
    
    print('create train data')
    X_svm_train_data, y_svm_train_data = create_svm_data_representation(train_data_generator)
    
    print('create test data')
    X_svm_test_data, y_svm_test_data = create_svm_data_representation(test_data_generator)

    print('classification')
    clf = SVC()
    clf.fit(X_svm_train_data, y_svm_train_data)

    y_pred = clf.predict(X_svm_test_data)
    accuracy = accuracy_score(y_svm_test_data, y_pred)

    print(f"Accuracy: {accuracy:.2f}")
