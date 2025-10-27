from collections import Counter

import numpy as np


def bin_features_into_64_space_bins(time_binned_sequences):
    space_binned_sequences = []
    for sequence in time_binned_sequences:
        space_binned_sequence = []
        for neurons in sequence:
            assert len(neurons) == 700

            bins = np.array_split(neurons, 64)

            bin_sums = np.array([bin.sum() for bin in bins])
            assert len(bin_sums) == 64

            space_binned_sequence.append(bin_sums)
        space_binned_sequences.append(np.array(space_binned_sequence))

    return np.array(space_binned_sequences)


def convert_to_time_binned_sequences(data):
    X = []
    Y = []

    for i, (spikes, label) in enumerate(data):
        sequences = []
        current_i = 0

        while current_i < 1400000:
            filtered_spikes = spikes[
                (spikes["t"] > current_i) & (spikes["t"] <= current_i + 10000)
            ]

            sequence = np.zeros(700)
            for neuron, count in Counter(filtered_spikes["x"]).items():
                sequence[neuron] = count

            current_i = current_i + 10000
            sequences.append(sequence)

        X.append(sequences)
        Y.append(label)

    return np.array(X), np.array(Y)
