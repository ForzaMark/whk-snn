from typing import Union

import torch
from constants import BATCH_SIZE, TIME_STEPS
from data.preprocessing_utils import (
    bin_features_into_64_space_bins,
    convert_to_time_binned_sequences,
)
from tonic import datasets, transforms
from torch.utils.data import DataLoader, Subset, TensorDataset

create_frame_transform = lambda time_steps, dataset: transforms.ToFrame(
    sensor_size=dataset.sensor_size, n_time_bins=time_steps
)


def create_data_loader(
    dataset,
    time_steps=TIME_STEPS,
    use_train_subset: Union[bool, int] = False,
    batch_size=BATCH_SIZE,
):
    dataset_fn = getattr(datasets, dataset)
    frame_transform = create_frame_transform(time_steps, dataset_fn)
    train_data = dataset_fn("./data", transform=frame_transform, train=True)
    test_data = dataset_fn("./data", transform=frame_transform, train=False)

    if use_train_subset:
        random_indices = torch.randperm(len(train_data))[:use_train_subset]
        train_data = Subset(train_data, random_indices)

    train_data_loader = DataLoader(train_data, shuffle=False, batch_size=batch_size)
    test_data_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)

    return train_data_loader, test_data_loader


def data_loader_factory(X, Y, batch_size=32):
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(Y, dtype=torch.long)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size)

    return loader


def create_data_loader_deep_models(mode, use_train_subset=None):
    train_data = datasets.SHD("./data", train=True)
    test_data = datasets.SHD("./data", train=False)

    if use_train_subset:
        random_indices = torch.randperm(len(train_data))[:use_train_subset]
        train_data = Subset(train_data, random_indices)

    x_train, y_train = convert_to_time_binned_sequences(train_data)
    x_test, y_test = convert_to_time_binned_sequences(test_data)

    if mode == "cnn":
        x_train = bin_features_into_64_space_bins(x_train)
        x_test = bin_features_into_64_space_bins(x_test)

    batch_size = BATCH_SIZE if mode == "lstm" else 1
    train_loader = data_loader_factory(x_train, y_train, batch_size=batch_size)
    test_loader = data_loader_factory(x_test, y_test, batch_size=batch_size)

    return train_loader, test_loader
