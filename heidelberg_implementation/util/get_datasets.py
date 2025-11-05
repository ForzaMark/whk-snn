from typing import Union

from eprop.util.eprop_dataset import Eprop_Dataset
from util.create_data_loader import create_data_loader, create_data_loader_deep_models


def get_shd_dataset(use_train_subset: Union[bool, int] = False):
    train_data_loader, test_data_loader = create_data_loader(
        "SHD", use_train_subset=use_train_subset
    )
    train_data_loader_cnn, test_data_loader_cnn = create_data_loader_deep_models(
        mode="cnn", use_train_subset=use_train_subset
    )
    train_data_loader_lstm, test_data_loader_lstm = create_data_loader_deep_models(
        mode="lstm", use_train_subset=use_train_subset
    )

    eprop_heidelberg_dataset = Eprop_Dataset(
        32, data_path="../data/SHD/numpy_features/"
    )

    return (
        train_data_loader,
        test_data_loader,
        train_data_loader_cnn,
        test_data_loader_cnn,
        train_data_loader_lstm,
        test_data_loader_lstm,
        eprop_heidelberg_dataset,
    )


def get_nmnist(use_train_subset=False):
    train_data_loader, test_data_loader = create_data_loader("NMNIST")

    raise NotImplementedError(
        "Number of NMNIST features per timestep need to be specified"
    )
    eprop_nmnist_dataset = Eprop_Dataset(
        32, data_path="../data/NMNIST/numpy_features/", n_features=None, n_classes=10
    )

    return (
        train_data_loader,
        test_data_loader,
        train_data_loader,
        test_data_loader,
        train_data_loader,
        test_data_loader,
    )
