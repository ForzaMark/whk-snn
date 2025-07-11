import json

import numpy as np
import torch


def save_best_performing_model(models, test_accs, path):
    best_test_acc_index = np.argmax(test_accs)
    best_model = models[best_test_acc_index]

    torch.save(best_model.state_dict(), path)


def save_all_models_per_epoch(models, path_prefix):
    for index, model in enumerate(models):
        torch.save(model.state_dict(), f"{path_prefix}_epoch_{index}.pth")


def save_configuration_output(result_data, output_file_path):
    with open(output_file_path, "w") as file:
        json.dump(result_data, file, indent=4)
