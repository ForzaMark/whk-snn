from typing import Union

from eprop.util.eprop_dataset import Eprop_Dataset
from tonic import datasets
from util.create_data_loader import create_data_loader, create_data_loader_deep_models
from util.nmnist_transform import nmnist_deep_model_transform


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


def get_nmnist_dataset(use_train_subset: Union[bool, int] = False):
    number_input_neurons = 1156
    max_timestep = 316000

    train_data_loader, test_data_loader = create_data_loader(
        "NMNIST", use_train_subset=use_train_subset
    )

    prepared_deep_model_nmnist_train_data = datasets.NMNIST(
        "../data", transform=nmnist_deep_model_transform(), train=True
    )
    prepared_deep_model_nmnist_test_data = datasets.NMNIST(
        "../data", transform=nmnist_deep_model_transform(), train=False
    )

    train_data_loader_cnn, test_data_loader_cnn = create_data_loader_deep_models(
        mode="cnn",
        use_train_subset=use_train_subset,
        train_data=prepared_deep_model_nmnist_train_data,
        test_data=prepared_deep_model_nmnist_test_data,
        number_input_neurons=number_input_neurons,
        max_timestep=max_timestep,
    )
    train_data_loader_lstm, test_data_loader_lstm = create_data_loader_deep_models(
        mode="lstm",
        use_train_subset=use_train_subset,
        train_data=prepared_deep_model_nmnist_train_data,
        test_data=prepared_deep_model_nmnist_test_data,
        number_input_neurons=number_input_neurons,
        max_timestep=max_timestep,
    )

    eprop_nmnist_dataset = Eprop_Dataset(
        32,
        data_path="../data/NMNIST/numpy_features/",
        n_classes=20,
        n_features=number_input_neurons,
    )

    return (
        train_data_loader,
        test_data_loader,
        train_data_loader_cnn,
        test_data_loader_cnn,
        train_data_loader_lstm,
        test_data_loader_lstm,
        eprop_nmnist_dataset,
    )
