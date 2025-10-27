import errno
import json
import os
import time
from datetime import datetime

import numpy as np
import numpy.random as rd
import tensorflow as tf

from .util.alif_eligibility_propagation import exp_convolve
from .util.configuration import FLAGS
from .util.get_cell import get_cell
from .util.hdd_dataset import HDD_Dataset
from .util.rate_coding_loss_last_output import rate_coding_loss_last_output
from .util.save_experiment_results import save_experiment_results

N_EPOCHS = 10
EPROP = "symmetric"  # 'symmetric' for eprop


FLAGS = {
    **FLAGS,
    "n_epochs": N_EPOCHS,
    "eprop": EPROP,
    "n_regular": 300,
    "n_adaptive": 100,
}

dataset = HDD_Dataset(
    FLAGS["batch"], data_path="./eprop/datasets/hdd/full_spiking_data/"
)
n_in = dataset.n_features

features = tf.placeholder(
    shape=(None, None, dataset.n_features), dtype=tf.float32, name="Features"
)
phns = tf.placeholder(shape=(None), dtype=tf.int64, name="Labels")
seq_len = tf.placeholder(dtype=tf.int32, shape=[None], name="SeqLen")
keep_prob = tf.placeholder(dtype=tf.float32, shape=(), name="KeepProb")
weighted_relevant_mask = tf.placeholder(
    shape=(None, None), dtype=tf.float32, name="RelevanceMask"
)
batch_size = tf.Variable(0, dtype=tf.int32, trainable=False, name="BatchSize")

lr = tf.Variable(
    FLAGS["lr_init"], dtype=tf.float32, trainable=False, name="LearningRate"
)
global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="GlobalStep")
lr_update = tf.assign(lr, lr * FLAGS["lr_decay"])
gd_noise = tf.Variable(0, dtype=tf.float32, trainable=False, name="GDNoise")

n_iteration_per_epoch = 100
ramping_learning_rate_values = tf.linspace(0.0, 1.0, num=n_iteration_per_epoch)
clipped_global_step = tf.minimum(global_step, n_iteration_per_epoch - 1)
ramping_learning_rate_op = tf.assign(
    lr, FLAGS["lr_init"] * ramping_learning_rate_values[clipped_global_step]
)


def batch_to_feed_dict(batch, is_train):
    features_np, phns_np, seq_len_np = batch
    n_batch = features_np.shape[0]

    relevance_mask_np = [
        (np.arange(len(seq_len_np)) < seq_len_np[i]) / seq_len_np[i]
        for i in range(n_batch)
    ]
    relevance_mask_np = np.array(relevance_mask_np)

    phns_labels = phns_np
    return {
        features: features_np,
        phns: phns_labels,
        seq_len: seq_len_np,
        weighted_relevant_mask: relevance_mask_np,
        batch_size: n_batch,
        keep_prob: FLAGS["drop_out_probability"] if is_train else 1.0,
    }


def leaky_integrate(mem, x, beta=0.99):
    mem = beta * mem + x

    return mem


def compute_result(type, test_result_tensors, sess):
    assert type in ["validation", "test"]
    total_batch_size = dataset.n_validation if type == "validation" else dataset.n_test
    n_minibatch = total_batch_size // FLAGS["test_batch"]
    mini_batch_sizes = [FLAGS["test_batch"] for _ in range(n_minibatch)]
    if total_batch_size - (n_minibatch * FLAGS["test_batch"]) != 0:
        mini_batch_sizes = mini_batch_sizes + [
            total_batch_size - (n_minibatch * FLAGS["test_batch"])
        ]

    feed_dict = None
    collect_results = {k: [] for k in test_result_tensors.keys()}
    for idx, mb_size in enumerate(mini_batch_sizes):
        selection = np.arange(mb_size)
        selection = selection + np.ones_like(selection) * idx * FLAGS["test_batch"]

        if type == "validation":
            data = dataset.get_next_validation_batch(selection)
        elif type == "test":
            data = dataset.get_next_test_batch(selection)

        feed_dict = batch_to_feed_dict(data, is_train=False)
        run_output = sess.run(test_result_tensors, feed_dict=feed_dict)
        for k, value in run_output.items():
            collect_results[k].append(value)

    mean_result = {key: np.mean(collect_results[key]) for key in collect_results.keys()}

    return mean_result, None


def run_eprop_lsnn():
    regularization_f0 = FLAGS["reg_rate"] / 1000

    taua = FLAGS["tau_a"]
    tauv = FLAGS["tau_v"]

    cell_forward = get_cell("FW", n_in, FLAGS)

    with tf.variable_scope("RNNs"):

        def bi_directional_lstm(inputs, layer_number):
            with tf.variable_scope("BiDirectionalLayer" + str(layer_number)):
                if layer_number == 0:
                    cell_f = cell_forward
                else:
                    cell_f = get_cell(
                        "FW" + str(layer_number),
                        n_input=(FLAGS["n_regular"] + FLAGS["n_adaptive"]),
                    )

                outputs_forward, _ = tf.nn.dynamic_rnn(
                    cell_f, inputs, dtype=tf.float32, scope="ForwardRNN"
                )

                outputs = outputs_forward[0]

                return outputs

        inputs = features

        for k_layer in range(FLAGS["n_layer"]):
            outputs = bi_directional_lstm(inputs, k_layer)
            inputs = outputs

        n_outputs = FLAGS["n_regular"] + FLAGS["n_adaptive"]

    with tf.name_scope("Output"):
        N_output_classes = dataset.n_classes
        w_out = tf.Variable(
            rd.randn(n_outputs, N_output_classes) / np.sqrt(n_outputs),
            dtype=tf.float32,
            name="OutWeights",
        )

        b_out = tf.Variable(
            np.zeros(N_output_classes), dtype=tf.float32, name="OutBias"
        )
        orig_shape = tf.shape(outputs)

        _, _, num_units = outputs.shape
        batch_size, time_steps = orig_shape[0], orig_shape[1]
        num_classes = w_out.shape[1]

        flat_outputs = tf.reshape(outputs, [-1, num_units])
        logits = tf.matmul(flat_outputs, w_out) + b_out

        logits = tf.reshape(logits, [batch_size, time_steps, num_classes])
        logits_time_major = tf.transpose(logits, [1, 0, 2])

        init_state = tf.fill([tf.shape(logits)[0], N_output_classes], 0.9)

        all_states = tf.scan(
            fn=lambda prev, cur: leaky_integrate(prev, cur),
            elems=logits_time_major,
            initializer=init_state,
        )

        final_outputs = all_states

    with tf.name_scope("Loss"):
        loss = rate_coding_loss_last_output(labels=phns, outputs=final_outputs)

        phn_prediction = tf.argmax(final_outputs[-1], axis=1)

        is_correct = tf.equal(phns, phn_prediction)
        is_correct_float = tf.cast(is_correct, dtype=tf.float32)

        acc = tf.reduce_mean(is_correct_float, axis=0)

    with tf.name_scope("Train"):
        opt = tf.train.AdamOptimizer(
            lr, epsilon=FLAGS["adam_epsilon"], beta1=FLAGS["momentum"]
        )

        grads = opt.compute_gradients(loss)

        train_var_list = [var for g, var in grads]
        train_step = opt.apply_gradients(grads, global_step=global_step)
        print("NUM OF TRAINABLE", len(train_var_list))
        for v in train_var_list:
            print(v.name)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    train_result_tensors = {"loss": loss, "acc": acc}

    test_result_tensors = {"loss": loss, "acc": acc}

    current_epoch = 0
    overall_results = []
    train_results = []

    print(f'E-Prop: {FLAGS["eprop"]} | n_layers: {FLAGS["n_layer"]}')

    t0 = time.time()

    while dataset.current_epoch <= FLAGS["n_epochs"]:
        k_iteration = sess.run(global_step)

        if (dataset.current_epoch != current_epoch) or (k_iteration == 0):
            train_loss = np.mean([res["loss"] for res in train_results])
            train_acc = np.mean([res["acc"] for res in train_results])

            train_results = []

            current_epoch = dataset.current_epoch

            test_result, _ = compute_result("test", test_result_tensors, sess)

            overall_results.append(
                {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "test_acc": test_result["acc"],
                }
            )

            print(
                "Epoch: {} \t loss {:.3g} (train) \t Acc: {:.3g} (train) \t"
                "ACC: {:.3g} (test)".format(
                    dataset.current_epoch, train_loss, train_acc, test_result["acc"]
                )
            )

        if k_iteration < 100:
            old_lr = sess.run(lr)
            new_lr = sess.run(ramping_learning_rate_op)
            if k_iteration == 0:
                print(
                    "Ramping learning rate during first epoch: {:.2g} -> {:.2g}".format(
                        old_lr, new_lr
                    )
                )

        _, train_result = sess.run(
            [train_step, train_result_tensors],
            feed_dict=batch_to_feed_dict(
                dataset.get_next_training_batch(), is_train=True
            ),
        )
        train_results.append(train_result)

    complete_time = time.time() - t0

    all_test_acc = [result["test_acc"] for result in overall_results]
    return np.max(all_test_acc)
