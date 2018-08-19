
import os
import pickle
import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn, layers
from operator import itemgetter
import log_util


######################
# TODO ::: no additional feature
######################
DATA_PATH = r"E:\USERLOG\select_user\total"

embedding = np.load(os.path.join(DATA_PATH, 'user_100dim.npy'))
with open(os.path.join(DATA_PATH, 'channel_vocab_100dim.pkl'), 'rb') as f:
    ch_vocab = pickle.load(f)
with open(os.path.join(DATA_PATH, 'total_user_dic_10m.pkl'), 'rb') as f:
    user_total_dic = pickle.load(f)

    
# load input, target
window_range = range(2, 14)
channel_input = []
channel_target = []
for window in window_range:
    part_input, part_target = log_util.seq_to_table(user_total_dic, window)
    for i in range(len(part_target)):
        channel_input.append(part_input[i])
        channel_target.append(part_target[i])
    print("complete window {}".format(window))

#channel_target2 = log_util.channel_to_category(channel_target)
#with open(os.path.join(DATA_PATH, 'channel_target2.pkl'), 'wb') as f: pickle.dump(channel_target2, f)
with open(os.path.join(DATA_PATH, 'channel_target2.pkl'), 'rb') as f:
    channel_target2 = pickle.load(f)
target2_vocab = {label:i+1 for i, label in enumerate(list(set(channel_target2)))}
    
        
# shuffle
indices = np.arange(len(channel_target))
np.random.shuffle(indices)
channel_input = itemgetter(*indices)(channel_input)
channel_target = itemgetter(*indices)(channel_target2)


# split
while True:
    dev_ = int(len(channel_target2) * 0.05)
    choice = list(np.random.choice(len(channel_target2), dev_))
    n_choice = list(set(range(len(channel_target2))) - set(choice))
    train_x = itemgetter(*n_choice)(channel_input)
    train_y = itemgetter(*n_choice)(channel_target2)
    dev_x = itemgetter(*choice)(channel_input)
    dev_y = itemgetter(*choice)(channel_target2)
    print('no....')
    if len(dev_x[0][0]) < 300:
        print('clear!')
        break

dev_x, dev_y, dev_max_day_cnt, dev_max_day_length = log_util.make_validation(list(dev_x), list(dev_y), ch_vocab, target2_vocab)


# parameter
VOCAB_SIZE = embedding.shape[0]
EMBEDDING_SIZE = embedding.shape[1]
HIDDEN_SIZE = 100
BATCH_SIZE = 200
EPOCHS = 100
N_CLASSES = len(set(channel_target))
MAX_DAY_NUM = max(len(day_cnt) for day_cnt in channel_input)


# architecture
with tf.name_scope('placeholder'):
    batch_size = tf.placeholder(tf.int32, name='batch_size')
    input_x = tf.placeholder(tf.int32, [None, None, None], name='input_x')
    input_y = tf.placeholder(tf.int32, [None, N_CLASSES])
    MAX_DAY_COUNT = tf.placeholder(tf.int32, name='max_day_count')
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
    channel_encode = log_util.BidirectionalLSTMEncoder(channel_embed, name='channel_encoder', hidden_size=HIDDEN_SIZE)
    day_vec = log_util.AttentionLayer(channel_encode, name='channel_attention', hidden_size=HIDDEN_SIZE)

with tf.name_scope('user2vec'):
    day_vec = tf.reshape(day_vec, [-1, MAX_DAY_COUNT, HIDDEN_SIZE*2])
    day_encode = log_util.BidirectionalLSTMEncoder(day_vec, name='day_encoder', hidden_size=HIDDEN_SIZE)
    user_vec = log_util.AttentionLayer(day_encode, name='day_attention', hidden_size=HIDDEN_SIZE)

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

    def train_step(x_batch, y_batch, max_day_cnt,  max_day_len):
        feed = {input_x : x_batch,
                input_y : y_batch,
                MAX_DAY_COUNT : max_day_cnt,
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
    def dev_step(x_batch, y_batch, max_day_cnt, max_day_len, writer=None):
        ind = np.arange(len(x_batch))
        np.random.shuffle(ind)
        dev_size = 1000
        x_batch_ = x_batch[ind[:dev_size]]
        y_batch_ = y_batch[ind[:dev_size]]
        split_size = 200
        dev_loss = []
        dev_acc = []
        for ind, i in enumerate(range(0, len(x_batch_)-split_size, split_size)):
            feed = {input_x : x_batch_[i: i+split_size],
                    input_y : y_batch_[i: i+split_size],
                    MAX_DAY_COUNT : max_day_cnt,
                    MAX_DAY_LENGTH : max_day_len,
                    batch_size : BATCH_SIZE}
            step, summaries, cost, accuracy = sess.run([global_step, dev_summary_op, loss, acc], feed_dict=feed)
            dev_loss.append(cost)
            dev_acc.append(accuracy)
            print('eval processing..{}/{}'.format(split_size*(ind+1), dev_size))
        cost_avg = np.average(dev_loss)
        acc_avg = np.average(dev_acc)
        time_str = str(int(time.time()))
        print("{} dev {}".format("="*3, "="*3))
        print("{}, loss {:g}, acc {:g}".format(time_str, cost_avg, acc_avg))
        print("="*9)
        if writer: writer.add_summary(summaries, step)

    for epoch in range(EPOCHS):
        print('CURRENT EPOCH {}'.format(epoch+1))
        train_batch = log_util.batch_generator(train_x, train_y, BATCH_SIZE, ch_vocab, target2_vocab)
        num_batches = len(train_x) // BATCH_SIZE
        for _ in range(num_batches):
            x_batch, y_batch, day_count, day_length = next(train_batch)
            step = train_step(x_batch, y_batch, day_count, day_length)
            if step % 500 == 0:
                dev_step(dev_x, dev_y, dev_max_day_cnt, dev_max_day_length, dev_summary_wirter)

            if step % 1000 == 0:
                saver.export_meta_graph('./check/log_han_model.meta', collection_list=['train_var'])
                saver.save(sess, './check/log_han_model.ckpt', global_step=global_step)
        # shuffle
        indices = np.arange(len(train_x))
        np.random.shuffle(indices)
        train_x = itemgetter(*indices)(train_x)
        train_y = itemgetter(*indices)(train_y)

    saver.save(sess, './check/log_han.ckpt', global_step=global_step)

#sess.close()

