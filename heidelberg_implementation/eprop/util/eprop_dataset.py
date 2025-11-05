import numpy as np
import numpy.random as rd


def split_train_validation(feature_stack_train, label_stack_train):
    n_samples = len(feature_stack_train)

    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    val_size = int(0.15 * n_samples)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    feature_stack_validation = feature_stack_train[val_idx]
    label_stack_validation = label_stack_train[val_idx]

    feature_stack_train = feature_stack_train[train_idx]
    label_stack_train = label_stack_train[train_idx]

    return (
        feature_stack_train,
        label_stack_train,
        feature_stack_validation,
        label_stack_validation,
    )


class Eprop_Dataset:
    def __init__(
        self, n_mini_batch, data_path, epsilon=1e-10, n_features=700, n_classes=20
    ):
        self.data_path = data_path
        self.epsilon = epsilon

        self.n_features = n_features
        self.n_classes = n_classes

        (
            self.feature_stack_train,
            self.label_stack_train,
            self.feature_stack_validation,
            self.label_stack_validation,
        ) = split_train_validation(*self.load_data_stack("train"))
        self.feature_stack_test, self.label_stack_test = self.load_data_stack("test")

        self.feature_stack_train = np.array(self.feature_stack_train, dtype=object)
        self.feature_stack_test = np.array(self.feature_stack_test, dtype=object)

        self.n_mini_batch = n_mini_batch
        self.n_train = len(self.feature_stack_train)
        self.n_validation = len(self.feature_stack_validation)
        self.n_test = len(self.feature_stack_test)

        self.mini_batch_indices = self.generate_mini_batch_selection_list()
        self.current_epoch = 0
        self.index_current_minibatch = 0

    def load_data_stack(self, dataset):
        path = f"{self.data_path}{dataset}"

        feature_path = f"{path}_features.npy"
        label_path = f"{path}_labels.npy"

        data_stack = np.load(feature_path)
        label_stack = np.load(label_path)

        return data_stack, label_stack

    def generate_mini_batch_selection_list(self):
        perm = rd.permutation(self.n_train)
        number_of_batches = self.n_train // self.n_mini_batch
        return np.array_split(perm, number_of_batches)

    def load_features(self, dataset, selection):
        if dataset == "train":
            feature_stack = self.feature_stack_train[selection]
            label_stack = self.label_stack_train[selection]
        elif dataset == "test":
            feature_stack = self.feature_stack_test[selection]
            label_stack = self.label_stack_test[selection]
        elif dataset == "validation":
            feature_stack = self.feature_stack_validation[selection]
            label_stack = self.label_stack_validation[selection]

        seq_len = [feature.shape[0] for feature in feature_stack]

        features = np.stack([feature for feature in feature_stack], axis=0)
        labels = np.stack([labels for labels in label_stack], axis=0)

        return features, labels, seq_len

    def get_next_training_batch(self):
        features, labels, seq_len = self.load_features(
            "train", selection=self.mini_batch_indices[self.index_current_minibatch]
        )

        self.index_current_minibatch += 1
        if self.index_current_minibatch >= len(self.mini_batch_indices):
            self.index_current_minibatch = 0
            self.current_epoch += 1

            # Shuffle the training set after each epoch
            number_of_batches = len(self.mini_batch_indices)
            training_set_indices = np.concatenate(self.mini_batch_indices)
            training_set_indices = rd.permutation(training_set_indices)
            self.mini_batch_indices = np.array_split(
                training_set_indices, number_of_batches
            )

        return features, labels, seq_len

    def get_test_batch(self):
        return self.load_features("test", np.arange(self.n_test, dtype=np.int))

    def get_next_test_batch(self, selection):
        return self.load_features("test", selection=selection)

    def get_validation_batch(self):
        return self.load_features(
            "validation", np.arange(self.n_validation, dtype=np.int)
        )

    def get_next_validation_batch(self, selection):
        return self.load_features("validation", selection=selection)
        return self.load_features("validation", selection=selection)
