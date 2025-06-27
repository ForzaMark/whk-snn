import matplotlib.pyplot as plt
import numpy as np


def get_spk_matrices(data, model, selection_index):
    x_selected = data[:, selection_index, :]

    spk_recs, _ = model(data)

    output_spk_rec = spk_recs[-1][:, selection_index, :]
    hidden_spk_rec = [hidden_spk_rec[:, selection_index, :].detach() for hidden_spk_rec in spk_recs[:-1]]

    return [x_selected, *hidden_spk_rec, output_spk_rec.detach()]

def plot_layer_development(models, data, selection_index, sub_titles, super_title, figsize=(20, 12)):
    spike_matrices = [
        get_spk_matrices(data, model, selection_index) for model in models
    ]

    for spike_matrix in spike_matrices:
        assert len(spike_matrix) == len(spike_matrices[0])

    fig, axes = plt.subplots(len(spike_matrices) if len(spike_matrices) > 1 else 2, len(spike_matrices[0]), figsize=figsize)

    fig.suptitle(super_title, fontsize=16)

    for row_index, (spike_matrix, sub_title) in enumerate(zip(spike_matrices, sub_titles)):
        for column_index in range(len(spike_matrices[0])):
            spike_matrix_np = spike_matrix[column_index].numpy()
            times, neurons = np.where(spike_matrix_np == 1)
            ax = axes[row_index, column_index]
            ax.scatter(times, neurons, s=1, color='black')
            ax.set_title(f"{sub_title} - Layer {column_index}")
            ax.set_xlabel("Time step")
            ax.set_ylim(-1, spike_matrix_np.shape[1])
            if column_index == 0:
                ax.set_ylabel("Neuron index")

    plt.tight_layout()
    plt.show()