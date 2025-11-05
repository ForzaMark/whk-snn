import os

import numpy as np
from numpy.lib.format import open_memmap
from util.create_data_loader import create_data_loader

# datasets = ["SHD", "NMNIST", "DVSGesture"]
datasets = ["SHD", "DVSGesture"]

for dataset in datasets:
    print(dataset)
    train_data_loader, test_data_loader = create_data_loader(
        dataset=dataset, time_steps=100, use_train_subset=False, batch_size=1
    )

    first_x, first_y = next(iter(train_data_loader))

    first_x_np = first_x.squeeze().numpy()

    path = f"../data/{dataset}/numpy_features"
    os.makedirs(path, exist_ok=True)

    shape_features = (len(train_data_loader.dataset),) + first_x_np.shape
    shape_labels = (len(train_data_loader.dataset),)

    features_mmap = open_memmap(
        f"{path}/train_features.npy", mode="w+", dtype="int8", shape=shape_features
    )
    labels_mmap = open_memmap(
        f"{path}/train_labels.npy", mode="w+", dtype="int8", shape=shape_labels
    )

    for i, (x, y) in enumerate(train_data_loader):
        features_mmap[i] = x.squeeze().numpy().astype("int8")
        labels_mmap[i] = np.array(y).astype("int8")
