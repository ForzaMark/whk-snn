import tensorflow as tf

def rate_coding_loss_last_output(outputs, labels):
    time_steps = 100
    num_neurons = 20

    outputs = tf.transpose(outputs, [1, 0, 2])

    last_output = outputs[:, -1, :]

    batch_size = tf.shape(outputs)[0]

    last_output_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=last_output, labels=labels
    )

    single_loss = tf.reduce_mean(last_output_loss)

    return single_loss
