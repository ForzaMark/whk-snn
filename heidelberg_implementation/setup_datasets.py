import numpy as np
from util.create_data_loader import create_data_loader
import os

datasets = ["SHD"]

for dataset in datasets:
    print(dataset)
    train_data_loader, test_data_loader = create_data_loader(
        dataset=dataset,
        time_steps=100, 
        use_train_subset=False, 
        batch_size=1
    )

    train_features = []
    train_labels = []

    for x,y in train_data_loader:
        features = x.squeeze()

        train_features.append(features.numpy())
        train_labels.append(y.item())    
        
    test_features = []
    test_labels = []

    for x,y in test_data_loader:
        features = x.squeeze()

        test_features.append(features.numpy())
        test_labels.append(y.item())  

    path = f"../data/{dataset}/numpy_features"
    os.makedirs(path, exist_ok=True)
    
    np.save(f"../data/{dataset}/numpy_features/test_features.npy", test_features)
    np.save(f"../data/{dataset}/numpy_features/train_features.npy", train_features)
    np.save(f"../data/{dataset}/numpy_features/test_labels.npy", test_labels)
    np.save(f"../data/{dataset}/numpy_features/train_labels.npy", train_labels)
