
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn, layers

def length(sequences):
    used = tf.sign(tf.reduce_max(tf.abs(sequences), reduction_indices=2))
    seq_len = tf.reduce_sum(used, reduction_indices=1)
    return tf.cast(seq_len, tf.int32)

def BidirectionalLSTMEncoder(inputs, name, hidden_size=50):
    with tf.variable_scope(name):
        GRU_cell_fw = rnn.LSTMCell(hidden_size,)
        GRU_cell_bw = rnn.LSTMCell(hidden_size)
        ((fw_outputs, bw_outputs), (_,_)) = tf.nn.bidirectional_dynamic_rnn(cell_fw=GRU_cell_fw, cell_bw=GRU_cell_bw, inputs=inputs, sequence_length=length(inputs), dtype=tf.float32)
        outputs = tf.concat((fw_outputs, bw_outputs), 2)
        return outputs

def AttentionLayer(inputs, name, hidden_size=50):
    with tf.variable_scope(name):
        u_context = tf.Variable(tf.truncated_normal([hidden_size * 2]), name='u_context')
        h = layers.fully_connected(inputs, hidden_size * 2, activation_fn=tf.nn.tanh)
        alpha = tf.nn.softmax(tf.reduce_sum(tf.multiply(h, u_context), axis=2, keepdims=True), dim=1)
        attention_output = tf.reduce_sum(tf.multiply(inputs, alpha), axis=1)
        return attention_output

def make_one_hot(y, vocab):
    lst = []
    for i, inst in enumerate(y):
        labels = [0] * len(set(vocab))
        if vocab.get(inst) == 999:
            labels[0] = 1
        else:
            labels[vocab.get(inst)-1] = 1
        lst.append(labels)
    return np.array(lst)

def batch_generator(X, y, batch_size, vocab_x, vocab_y, shuffle=False):
    # X, y type --> list
    for i in range(0, len(X)-batch_size, batch_size):
        batch_x_lst = X[i:i+batch_size]
        batch_y_lst = y[i:i+batch_size]

        day_length = [len(day_1) for day_3 in batch_x_lst for day_1 in day_3]
        max_day_length = max(day_length)

        day_count = [len(day_3) for day_3 in batch_x_lst]
        max_day_count = max(day_count)

        batch_x = np.zeros([batch_size, max_day_count, max_day_length])
        for i_3, day_3_lst in enumerate(batch_x_lst):
            sub_batch_3 = np.zeros([max_day_count, max_day_length])
            for i_1, day_1_lst in enumerate(day_3_lst):
                sub_batch_1 = np.zeros([max_day_length])
                for voc_i, voc in enumerate(day_1_lst):
                    sub_batch_1[voc_i] = vocab_x.get(voc)
                sub_batch_3[i_1] = sub_batch_1
            batch_x[i_3] = sub_batch_3
        #batch_x = np.array(batch_x_lst)
        batch_y = make_one_hot(batch_y_lst, vocab_y)

        yield batch_x, batch_y, max_day_count, max_day_length


def make_validation(X, y, vocab_x, vocab_y):
    val_x_lst = X.copy()
    val_y_lst = y.copy()

    max_length = max(len(day_1) for day_3 in val_x_lst for day_1 in day_3)
    max_cnt = max(len(day_3) for day_3 in val_x_lst)

    val_x = np.zeros([len(X), max_cnt, max_length])
    for i_3, day_3_lst in enumerate(val_x_lst):
        sub_val_x_3 = np.zeros([max_cnt, max_length])
        for i_1, day_1_lst in enumerate(day_3_lst):
            sub_val_x_1 = np.zeros([max_length])
            for voc_i, voc in enumerate(day_1_lst):
                sub_val_x_1[voc_i] = vocab_x.get(voc)
            sub_val_x_3[i_1] = sub_val_x_1
        val_x[i_3] = sub_val_x_3
    val_y = make_one_hot(val_y_lst, vocab_y)

    return val_x, val_y, max_cnt, max_length


def seq_to_table(seq, window_size, stride=1):
    # seq type --> dictionary
    X = []; y = []
    for ind, channel in enumerate(list(seq.values())):
        for i in range(0, len(channel)-window_size, stride):
            if i == len(channel)-window_size+1:
                break
            subset = channel[i:(i+window_size)]
            X.append(subset[:window_size-1])
            y.append(subset[-1][0])
        if ind+1 % 100 == 0:
            print('{}-{}'.format(ind+1, len(seq)))
    return X, y


def channel_to_category(channel_label, match_path=r"D:\USERLOG\select_user\tv_channel_info.csv"):
    import pandas as pd
    match_data = pd.read_csv(match_path)
    match_data.index = list(match_data.iloc[:,1])
    category_list = []
    for i, channel in enumerate(channel_label):
        channel_ = channel[1:]
        if channel == 'Jap':
            category_list.append('Jap')
        elif int(channel_) in list(match_data.index):
            label = match_data[match_data.index==int(channel_)]['종류'].values[0]
            category_list.append(label)
        else:
            category_list.append('기타')

        if i%10000 == 0: print(i, len(channel_label))
    return category_list

