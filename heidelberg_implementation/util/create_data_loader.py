from typing import Union

import torch
from constants import BATCH_SIZE, TIME_STEPS
from tonic import datasets, transforms
from torch.utils.data import DataLoader, Subset

create_frame_transform = lambda time_steps: transforms.ToFrame(
    sensor_size=datasets.SHD.sensor_size, n_time_bins=time_steps
)


def create_data_loader(
    time_steps=TIME_STEPS, use_train_subset: Union[bool, int] = False
):
    frame_transform = create_frame_transform(time_steps=time_steps)
    train_data = datasets.SHD("./data", transform=frame_transform, train=True)
    test_data = datasets.SHD("./data", transform=frame_transform, train=False)

    if use_train_subset:
        random_indices = torch.randperm(len(train_data))[:use_train_subset]
        train_data = Subset(train_data, random_indices)

    train_data_loader = DataLoader(train_data, shuffle=False, batch_size=BATCH_SIZE)
    test_data_loader = DataLoader(test_data, shuffle=False, batch_size=BATCH_SIZE)

    return train_data_loader, test_data_loader
