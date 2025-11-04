import os

import numpy as np
from util.create_data_loader import create_data_loader

datasets = ["SHD", "NMNIST"]

for dataset in datasets:
    print(dataset)
    train_data_loader, test_data_loader = create_data_loader(
        dataset=dataset, time_steps=100, use_train_subset=False, batch_size=1
    )

    path = f"../data/{dataset}/numpy_features"
    os.makedirs(path, exist_ok=True)

    first_x, first_y = next(iter(train_data_loader))

    shape = (len(train_data_loader.dataset),) + first_x.squeeze().shape
    features_mmap = np.memmap(
        f"{path}/train_features.npy", dtype="float32", mode="w+", shape=shape
    )
    labels_mmap = np.memmap(
        f"{path}/train_labels.npy",
        dtype="int64",
        mode="w+",
        shape=(len(train_data_loader.dataset),),
    )

    for i, (x, y) in enumerate(train_data_loader):
        features_mmap[i] = x.squeeze().numpy()
        labels_mmap[i] = y.item()

    features_mmap.flush()
    labels_mmap.flush()
