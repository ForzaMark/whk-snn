import numpy as np
from tonic import transforms


def keep_polarity_zero(events):
    return events[events["p"] == 0]


def multiply_x_y(events):
    new_events = np.zeros_like(events)

    new_events["x"] = events["x"] * events["y"]
    new_events["y"] = 0
    new_events["p"] = 0
    new_events["t"] = events["t"]

    return new_events


def convert_to_spikes(events):
    events[events > 0] = 1
    return events


def nmnist_transform(time_steps, custom_sensor_size):
    return transforms.Compose(
        [
            keep_polarity_zero,
            multiply_x_y,
            transforms.ToFrame(sensor_size=custom_sensor_size, n_time_bins=time_steps),
            convert_to_spikes,
        ]
    )


def nmnist_deep_model_transform():
    return transforms.Compose([keep_polarity_zero, multiply_x_y])
