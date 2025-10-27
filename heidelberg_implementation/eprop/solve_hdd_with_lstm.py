import errno
import json
import os
import time
from datetime import datetime

import numpy as np
import numpy.random as rd
import tensorflow as tf

from .util.hdd_dataset import HDD_Dataset
from .util.lstm_eprop_model import CustomLSTM
from .util.save_experiment_results import save_experiment_results

FLAGS = tf.app.flags.FLAGS


def flag_to_dict(FLAG):
    if float(tf.__version__[2:]) >= 5:
        flag_dict = FLAG.flag_values_dict()
    else:
        flag_dict = FLAG.__flags
    return flag_dict


def batch_to_feed_dict(batch):
    features_np, phns_np, seq_len_np = batch
    n_batch, n_time = phns_np.shape[0], 1

    relevance_mask_np = [
        (np.arange(len(seq_len_np)) < seq_len_np[i]) / seq_len_np[i]
        for i in range(n_batch)
    ]

    return {
        features: features_np,
        phns: phns_np,
        weighted_relevant_mask: relevance_mask_np,
    }


tf.app.flags.DEFINE_string("comment", "", "comment attached to output filenames")
tf.app.flags.DEFINE_string(
    "dataset", "../datasets/timit_processed", "Path to dataset to use"
)
tf.app.flags.DEFINE_string(
    "preproc", "mfccs", "Input preprocessing: fbank, mfccs, cochspec, cochspike, htk"
)
tf.app.flags.DEFINE_string(
    "eprop", None, "options: [None, symmetric, adaptive, random], None means use BPTT"
)
tf.app.flags.DEFINE_bool("adam", True, "use ADAM instead of standard SGD")
tf.app.flags.DEFINE_bool("lstm", True, "plot regularly the predicitons")
#
tf.app.flags.DEFINE_integer("seed", -1, "seed number")
tf.app.flags.DEFINE_integer("n_epochs", 30, "number of iteration ")
tf.app.flags.DEFINE_integer("n_lstm", 200, "number of lstm cells")
tf.app.flags.DEFINE_integer("print_every", 100, "print every and store accuracy")
tf.app.flags.DEFINE_integer("lr_decay_every", 500, "Decay every")
tf.app.flags.DEFINE_integer("batch", 32, "mini_batch size")
#
tf.app.flags.DEFINE_float(
    "init_scale", 0.0, "Provide the scaling of the weights at initialization"
)
tf.app.flags.DEFINE_float("l2", 1e-5, "l2 regularization")
tf.app.flags.DEFINE_float("lr_decay", 0.3, "Learning rate decay")
tf.app.flags.DEFINE_float("lr_init", 0.01, "Initial learning rate")
tf.app.flags.DEFINE_float(
    "adam_epsilon",
    1e-5,
    "Epsilon parameter in adam to cut gradients with small variance",
)
tf.app.flags.DEFINE_float(
    "readout_decay", 1e-3, "weight decay of readout and broadcast weights 0.001"
)

dataset = HDD_Dataset(FLAGS.batch, data_path="./eprop/datasets/hdd/full_spiking_data/")

features = tf.placeholder(
    shape=(None, None, dataset.n_features), dtype=tf.float32, name="Features"
)
phns = tf.placeholder(shape=(None), dtype=tf.int64, name="Labels")
weighted_relevant_mask = tf.placeholder(shape=(None, None), dtype=tf.float32)
audio = tf.placeholder(shape=(None, None), dtype=tf.float32, name="Audio")

lr = tf.Variable(FLAGS.lr_init, dtype=tf.float32, trainable=False)
global_step = tf.Variable(0, dtype=tf.int32, trainable=False)
lr_update = tf.assign(lr, lr * FLAGS.lr_decay)

cell_forward = CustomLSTM(FLAGS.n_lstm, stop_gradients=FLAGS.eprop is not None)
cell_backward = CustomLSTM(FLAGS.n_lstm, stop_gradients=FLAGS.eprop is not None)
initializer = None

with tf.variable_scope("RNNs", initializer=initializer):
    outputs_forward, _ = tf.nn.dynamic_rnn(
        cell_forward, features, dtype=tf.float32, scope="ForwardRNN"
    )
    outputs_backward, _ = tf.nn.dynamic_rnn(
        cell_backward,
        tf.reverse(features, axis=[1]),
        dtype=tf.float32,
        scope="BackwardRNN",
    )
    outputs_backward = tf.reverse(outputs_backward, axis=[1])

    last_forward = outputs_forward[:, -1, :]
    last_backward = outputs_backward[:, 0, :]

    outputs = tf.concat([last_forward, last_backward], axis=1)

with tf.name_scope("Output"):
    w_out_init = rd.randn(FLAGS.n_lstm * 2, dataset.n_classes) / np.sqrt(
        FLAGS.n_lstm * 2
    )  # original
    w_out = tf.Variable(w_out_init, dtype=tf.float32)

    phn_logits = tf.matmul(outputs, w_out)

    b_out = tf.Variable(np.zeros(dataset.n_classes), dtype=tf.float32)

    phn_logits = phn_logits + b_out

with tf.name_scope("Loss"):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=phns, logits=phn_logits
    )
    loss = tf.reduce_mean(loss)

    if FLAGS.l2 > 0:
        losses_l2 = [tf.reduce_sum(tf.square(w)) for w in tf.trainable_variables()]
        loss += FLAGS.l2 * tf.reduce_sum(losses_l2)

    phn_prediction = tf.argmax(phn_logits, axis=1)
    is_correct = tf.equal(phns, phn_prediction)
    is_correct_float = tf.cast(is_correct, dtype=tf.float32)

    acc = tf.reduce_mean(is_correct_float, axis=0)

with tf.name_scope("Train"):
    train_step = tf.train.AdamOptimizer(lr, epsilon=FLAGS.adam_epsilon).minimize(
        loss, global_step=global_step
    )

sess = tf.Session()
sess.run(tf.global_variables_initializer())

overall_results = []
test_result_tensors = {"loss": loss, "acc": acc}
train_result_tensors = {"loss": loss, "acc": acc}


t0 = time.time()


def run_eprop_lstm():
    current_epoch = dataset.current_epoch
    train_results = []

    while dataset.current_epoch <= FLAGS.n_epochs:
        k_iteration = sess.run(global_step)

        if np.mod(k_iteration, FLAGS.lr_decay_every) == 0 and k_iteration > 0:
            sess.run(lr_update)
            print("Decay learning rate: {:.2g}".format(sess.run(lr)))

        if (dataset.current_epoch) != current_epoch or (k_iteration == 0):
            current_epoch = dataset.current_epoch

            train_loss = np.mean([res["loss"] for res in train_results])
            train_acc = np.mean([res["acc"] for res in train_results])

            train_results = []

            test_result = sess.run(
                test_result_tensors,
                feed_dict=batch_to_feed_dict(dataset.get_test_batch()),
            )

            overall_results.append(
                {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "test_acc": test_result["acc"],
                }
            )

            print(
                "Epoch: {} \t loss {:.3g} (train) \t acc: {:.3g} (train) \t acc: {:.3g} (test)".format(
                    dataset.current_epoch, train_loss, train_acc, test_result["acc"]
                )
            )

        _, train_res = sess.run(
            [train_step, train_result_tensors],
            feed_dict=batch_to_feed_dict(dataset.get_next_training_batch()),
        )

        train_results.append(train_res)

    all_test_acc = [result["test_acc"] for result in overall_results]
    return np.max(all_test_acc)
