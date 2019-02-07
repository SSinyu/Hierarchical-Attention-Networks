import tensorflow as tf
from tensorflow.contrib import rnn, layers


def length_seq(seq):
    idx = tf.sign(tf.reduce_max(tf.abs(seq), reduction_indices=2))
    seq_len = tf.cast(tf.reduce_sum(idx, reduction_indices=1), tf.int32)
    return seq_len


def BiGRU(x, hidden_size, dropout_prob=None, n_layer=None):
    x_len = length_seq(x)
    if n_layer: # multilayer GRU
        GRU_forwards = [rnn.GRUCell(hidden_size) for _ in range(n_layer)]
        GRU_backwards = [rnn.GRUCell(hidden_size) for _ in range(n_layer)]
        if dropout_prob:
            GRU_forwards = [rnn.DropoutWrapper(i, input_keep_prob=dropout_prob) for i in GRU_forwards]
            GRU_backwards = [rnn.DropoutWrapper(i, input_keep_prob=dropout_prob) for i in GRU_backwards]
        out, _, _ = rnn.stack_bidirectional_dynamic_rnn(cells_fw=GRU_forwards, cells_bw=GRU_backwards, inputs=x, sequence_length=x_len, dtype=tf.float32)
    else:
        GRU_forward = rnn.GRUCell(hidden_size)
        GRU_backword = rnn.GRUCell(hidden_size)
        if dropout_prob:
            GRU_forward = rnn.DropoutWrapper(GRU_forward, input_keep_prob=dropout_prob)
            GRU_backword = rnn.DropoutWrapper(GRU_backword, input_keep_prob=dropout_prob)
        (forward_out, backward_out), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=GRU_forward, cell_bw=GRU_backword, inputs=x, sequence_length=x_len, dtype=tf.float32)
        out = tf.concat((forward_out, backward_out), axis=2)
    return out


def Attention_(x, hidden_size):
    contexts = tf.Variable(tf.truncated_normal(shape=[hidden_size]))
    h = layers.fully_connected(x, hidden_size, activation_fn=tf.nn.tanh)
    alpha = tf.nn.softmax(tf.reduce_sum(tf.multiply(h, contexts), axis=2, keepdims=True), axis=1)
    out = tf.reduce_sum(tf.multiply(x, alpha), axis=1)
    return out


def fc_(x, hidden_size):
    return layers.fully_connected(inputs=x,
                                  num_outputs=hidden_size,
                                  activation_fn=None)
