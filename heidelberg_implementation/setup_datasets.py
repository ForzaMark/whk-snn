import os

import numpy as np
from numpy.lib.format import open_memmap
from util.create_data_loader import create_data_loader

# datasets = ["SHD", "NMNIST", "DVSGesture"]
datasets = ["SHD"]

for dataset in datasets:
    print(dataset)
    train_data_loader, test_data_loader = create_data_loader(
        dataset=dataset, time_steps=100, use_train_subset=False, batch_size=1
    )

    first_x, first_y = next(iter(train_data_loader))

    first_x_np_train = first_x.squeeze().numpy()

    first_x, first_y = next(iter(test_data_loader))
    first_x_np_test = first_x.squeeze().numpy()

    path = f"../data/{dataset}/numpy_features"
    os.makedirs(path, exist_ok=True)

    shape_features_train = (len(train_data_loader.dataset),) + first_x_np_train.shape
    shape_labels_train = (len(train_data_loader.dataset),)

    shape_features_test = (len(test_data_loader.dataset),) + first_x_np_test.shape
    shape_labels_test = (len(test_data_loader.dataset),)

    features_mmap_train = open_memmap(
        f"{path}/train_features.npy",
        mode="w+",
        dtype="int8",
        shape=shape_features_train,
    )
    labels_mmap_train = open_memmap(
        f"{path}/train_labels.npy", mode="w+", dtype="int8", shape=shape_labels_train
    )

    features_mmap_test = open_memmap(
        f"{path}/test_features.npy", mode="w+", dtype="int8", shape=shape_features_test
    )
    labels_mmap_test = open_memmap(
        f"{path}/test_labels.npy", mode="w+", dtype="int8", shape=shape_labels_test
    )

    for i, (x, y) in enumerate(train_data_loader):
        features_mmap_train[i] = x.squeeze().numpy().astype("int8")
        labels_mmap_train[i] = np.array(y).astype("int8")

    for i, (x, y) in enumerate(test_data_loader):
        features_mmap_test[i] = x.squeeze().numpy().astype("int8")
        labels_mmap_test[i] = np.array(y).astype("int8")
