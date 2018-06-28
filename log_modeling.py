
### TODO ::: GRUCell --> LSTMCell(state_is_tuple=True)
### TODO ::: multivariate input (https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/), (https://stackoverflow.com/questions/38588869/concatenating-features-to-word-embedding-at-the-input-layer-during-run-time)


import os
import json
import pickle
import re
import time
import logging
from gensim.models import word2vec
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from tensorflow.contrib import rnn, layers
from operator import itemgetter
from collections import defaultdict

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
        max_length = max(day_length)

        batch_x = np.zeros([batch_size, 3, max_length])
        for i_3, day_3_lst in enumerate(batch_x_lst):
            sub_batch_3 = np.zeros([3, max_length])
            for i_1, day_1_lst in enumerate(day_3_lst):
                sub_batch_1 = np.zeros([max_length])
                for voc_i, voc in enumerate(day_1_lst):
                    sub_batch_1[voc_i] = vocab_x.get(voc)
                sub_batch_3[i_1] = sub_batch_1
            batch_x[i_3] = sub_batch_3
        #batch_x = np.array(batch_x_lst)
        batch_y = make_one_hot(batch_y_lst, vocab_y)

        yield batch_x, batch_y, max_length


def make_validation(X, y, vocab_x, vocab_y):
    val_x_lst = X.copy()
    val_y_lst = y.copy()

    max_length = max(len(day_1) for day_3 in val_x_lst for day_1 in day_3)
    #max_length = max(day_length)
    val_x = np.zeros([len(X), 3, max_length])
    for i_3, day_3_lst in enumerate(val_x_lst):
        sub_val_x_3 = np.zeros([3, max_length])
        for i_1, day_1_lst in enumerate(day_3_lst):
            sub_val_x_1 = np.zeros([max_length])
            for voc_i, voc in enumerate(day_1_lst):
                sub_val_x_1[voc_i] = vocab_x.get(voc)
            sub_val_x_3[i_1] = sub_val_x_1
        val_x[i_3] = sub_val_x_3
    val_y = make_one_hot(val_y_lst, vocab_y)

    return val_x, val_y, max_length


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



######################
# TODO ::: Experiments
# TODO ::: no additional feature
######################
DATA_PATH = r"D:\USERLOG\select_user\total"

embedding = np.load(os.path.join(DATA_PATH, 'user_100dim.npy'))
with open(os.path.join(DATA_PATH, 'channel_vocab_100dim.pkl'), 'rb') as f:
    ch_vocab = pickle.load(f)
with open(os.path.join(DATA_PATH, 'total_user_dic_10m.pkl'), 'rb') as f:
    user_total_dic = pickle.load(f)

# load input, target
WINDOW = 4 # input3, target1
channel_input, channel_target = seq_to_table(user_total_dic, WINDOW)
#channel_target2 = channel_to_category(channel_target)
#with open(os.path.join(DATA_PATH, 'channel_target2.pkl'), 'wb') as f: pickle.dump(channel_target2, f)
with open(os.path.join(DATA_PATH, 'channel_target2.pkl'), 'rb') as f:
    channel_target2 = pickle.load(f)
target2_vocab = {label:i+1 for i, label in enumerate(list(set(channel_target2)))}

# shuffle
indices = np.arange(len(channel_target))
np.random.shuffle(indices)
channel_input = itemgetter(*indices)(channel_input)
channel_target = itemgetter(*indices)(channel_target2)

# hold out
dev_ = int(len(channel_target) * 0.05)
train_x, dev_x = channel_input[:dev_], channel_input[-dev_:]
train_y, dev_y = channel_target[:dev_], channel_target[-dev_:]
dev_x, dev_y, max_day_length = make_validation(list(dev_x), list(dev_y), ch_vocab, target2_vocab)

# generator test
#train_batch = batch_generator(train_x, train_y, 100, ch_vocab)
#num_batches = len(train_x) // 100
#for i in range(num_batches):
#    bx, by = next(train_batch)
#    if i % 100 == 0: print(bx[0][0])

# parameter
VOCAB_SIZE = embedding.shape[0]
EMBEDDING_SIZE = embedding.shape[1]
HIDDEN_SIZE = 100
BATCH_SIZE = 200
EPOCHS = 100
N_CLASSES = len(set(channel_target))
MAX_DAY_NUM = max(len(day_cnt) for day_cnt in channel_input)

# designing architecture
with tf.name_scope('placeholder'):
    batch_size = tf.placeholder(tf.int32, name='batch_size')
    input_x = tf.placeholder(tf.int32, [None, None, None], name='input_x')
    input_y = tf.placeholder(tf.int32, [None, N_CLASSES])
    MAX_DAY_LENGTH = tf.placeholder(tf.int32, name='max_day_length')

with tf.name_scope('channel2vec'):
    embed = tf.Variable(tf.constant(0.0, shape=[VOCAB_SIZE, EMBEDDING_SIZE]), trainable=False, name='embed')
    embed_placeholder = tf.placeholder(tf.float32, [VOCAB_SIZE, EMBEDDING_SIZE])
    embed_init = embed.assign(embed_placeholder)
    channel_embed = tf.nn.embedding_lookup(embed, input_x)
    ### randomly initialize
    #embed_mat = tf.Variable(tf.truncated_normal((VOCAB_SIZE, EMBEDDING_SIZE)))
    #channel_rand_embed = tf.nn.embedding_lookup(embed_mat, input_x)

with tf.name_scope('day2vec'):
    channel_embed = tf.reshape(channel_embed, [-1, MAX_DAY_LENGTH, EMBEDDING_SIZE])
    channel_encode = BidirectionalLSTMEncoder(channel_embed, name='channel_encoder', hidden_size=HIDDEN_SIZE)
    day_vec = AttentionLayer(channel_encode, name='channel_attention', hidden_size=HIDDEN_SIZE)

with tf.name_scope('user2vec'):
    day_vec = tf.reshape(day_vec, [-1, MAX_DAY_NUM, HIDDEN_SIZE*2])
    day_encode = BidirectionalLSTMEncoder(day_vec, name='day_encoder', hidden_size=HIDDEN_SIZE)
    user_vec = AttentionLayer(day_encode, name='day_attention', hidden_size=HIDDEN_SIZE)

with tf.name_scope('next_channel_clf'):
    out = layers.fully_connected(inputs=user_vec, num_outputs=N_CLASSES, activation_fn=None)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=input_y, logits=out, name='loss'))

with tf.name_scope('accuracy'):
    predict = tf.argmax(out, axis=1, name='predict')
    label = tf.argmax(input_y, axis=1, name='label')
    acc = tf.reduce_mean(tf.cast(tf.equal(predict, label), tf.float32))






with tf.Session() as sess:
    timestamp = str(int(time.time()))
    run_dir = r"D:\seeuser"
    out_dir = os.path.abspath(os.path.join(run_dir, "runs", timestamp))
    print("Writing to {}\n".format(out_dir))

    global_step = tf.Variable(0, trainable=False)
    optimizer = tf.train.AdamOptimizer(1e-4)

    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), 5)
    grads_and_vars = tuple(zip(grads, tvars))
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    # keep track of gradient values and sparsity
    grad_summaries = []
    for g, v in grads_and_vars:
        if g is not None:
            grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
            grad_summaries.append(grad_hist_summary)
    grad_summaries_merged = tf.summary.merge(grad_summaries)

    loss_summary = tf.summary.scalar('loss', loss)
    acc_summary = tf.summary.scalar('accuracy', acc)

    train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

    dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
    dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
    dev_summary_wirter = tf.summary.FileWriter(dev_summary_dir, sess.graph)

    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    ckpt = tf.train.get_checkpoint_state('./check')

    sess.run(tf.global_variables_initializer())
    sess.run(embed_init, feed_dict={embed_placeholder : embedding})

    def train_step(x_batch, y_batch, max_day_len):
        feed = {input_x : x_batch,
                input_y : y_batch,
                MAX_DAY_LENGTH : max_day_len,
                batch_size : BATCH_SIZE}
        _, step, summaries, cost, accuracy = sess.run([train_op, global_step, train_summary_op, loss, acc], feed_dict=feed)
        time_str = str(int(time.time()))
        print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, cost, accuracy))
        train_summary_writer.add_summary(summaries, step)
        return step

    '''
    ### OOM error
    def dev_step(x_batch, y_batch, max_day_len, writer=None):
        feed = {input_x : x_batch,
                input_y : y_batch,
                MAX_DAY_LENGTH : max_day_len,
                batch_size : BATCH_SIZE}
        step, summaries, cost, accuracy = sess.run([global_step, dev_summary_op, loss, acc], feed_dict=feed)
        print("{}dev{}".format("="*3, "="*3))
        print("{}, loss {:g}, acc {:g}".format(step, cost, accuracy))
        print("="*9)
        if writer:
            writer.add_summary(summaries, step)
    '''

    def dev_step(x_batch, y_batch, max_day_len, writer=None):
        ind = np.arange(len(x_batch))
        np.random.shuffle(ind)
        x_batch_ = x_batch[ind[:2000]]
        y_batch_ = y_batch[ind[:2000]]
        split_size = 100
        dev_loss = []
        dev_acc = []
        for i in range(0, len(x_batch_)-split_size, split_size):
            feed = {input_x : x_batch_[i: i+split_size],
                    input_y : y_batch_[i: i+split_size],
                    MAX_DAY_LENGTH : max_day_len,
                    batch_size : BATCH_SIZE}
            step, summaries, cost, accuracy = sess.run([global_step, dev_summary_op, loss, acc], feed_dict=feed)
            dev_loss.append(cost)
            dev_acc.append(accuracy)
        cost_avg = np.average(dev_loss)
        acc_avg = np.average(dev_acc)
        time_str = str(int(time.time()))
        print("{}dev{}".format("="*3, "="*3))
        print("{}, loss {:g}, acc {:g}".format(time_str, cost_avg, acc_avg))
        print("="*9)



    for epoch in range(EPOCHS):
        print('CURRENT EPOCH {}'.format(epoch+1))
        train_batch = batch_generator(train_x, train_y, BATCH_SIZE, ch_vocab, target2_vocab)
        num_batches = len(train_x) // BATCH_SIZE
        for _ in range(num_batches):
            x_batch, y_batch, day_length = next(train_batch)
            step = train_step(x_batch, y_batch, day_length)
            if step % 500 == 0:
                dev_step(dev_x, dev_y, max_day_length, dev_summary_wirter)
        # shuffle
        indices = np.arange(len(train_x))
        np.random.shuffle(indices)
        train_x = itemgetter(*indices)(train_x)
        train_y = itemgetter(*indices)(train_y)



    saver.save(sess, './check/log_han.ckpt', global_step=global_step)



#sess.close()

