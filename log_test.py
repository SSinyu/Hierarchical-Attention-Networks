
import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
from log_util import BidirectionalLSTMEncoder, AttentionLayer, make_validation, batch_generator, seq_to_table

tf.reset_default_graph()

# parameter
VOCAB_SIZE = 246 #embedding.shape[0]
EMBEDDING_SIZE = 100 #embedding.shape[1]
HIDDEN_SIZE = 100
BATCH_SIZE = 32
EPOCHS = 100
N_CLASSES = 23 #len(set(channel_target2))

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
    channel_encode = BidirectionalLSTMEncoder(channel_embed, name='channel_encoder', hidden_size=HIDDEN_SIZE)
    day_vec = AttentionLayer(channel_encode, name='channel_attention', hidden_size=HIDDEN_SIZE)

with tf.name_scope('user2vec'):
    day_vec = tf.reshape(day_vec, [-1, MAX_DAY_COUNT, HIDDEN_SIZE*2])
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






### extract user vector

saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess, r'C:\Users\kimlab\PycharmProjects\userlog\check\check\log_han_model.ckpt-360000')

DATA_PATH = r"D:\USERLOG\select_user\total"
with open(os.path.join(DATA_PATH, 'total_user_dic_10m.pkl'), 'rb') as f:
    user_total_dic = pickle.load(f)
print(len(user_total_dic.keys()))
embedding = np.load(os.path.join(DATA_PATH, 'user_100dim.npy'))
with open(os.path.join(DATA_PATH, 'channel_vocab_100dim.pkl'), 'rb') as f:
    ch_vocab = pickle.load(f)

'''
total_user = []
for ind, channel in enumerate(list(user_total_dic.values())):
    total_user.append(channel)

day_count = [len(day_10) for day_10 in total_user]
day_length = [len(day_1) for day_10 in total_user for day_1 in day_10]
print(sorted(day_length)[-20:])
max_c = max(day_count)
max_l = max(day_length)


user_array = np.zeros([len(total_user) ,max_c, max_l])
for i_15, day_15_lst in enumerate(total_user):
    sub_15 = np.zeros([max_c, max_l])
    for i_1, day_1_lst in enumerate(day_15_lst):
        sub_1 = np.zeros([max_l])
        for voc_i, voc in enumerate(day_1_lst):
            sub_1[voc_i] = ch_vocab.get(voc)
        sub_15[i_1] = sub_1
    user_array[i_15] = sub_15
'''
#np.save(r'D:\USERLOG\select_user\total\user_array.npy', user_array)
user_array = np.load(r'D:\USERLOG\select_user\total\user_array.npy')

sub_user = user_array[0:38]
user_v = sess.run(user_vec, feed_dict={
    input_x : sub_user,
    MAX_DAY_COUNT : max(len(sub) for sub in sub_user),
    MAX_DAY_LENGTH : max(len(ssub) for sub in sub_user for ssub in sub)})

for i in range(38, len(user_array), 38):
    print("{}/{}".format(i, len(user_array)))
    if i != 38000:
        sub_user = user_array[i:i+38]
        uv = sess.run(user_vec, feed_dict={
            input_x : sub_user,
            MAX_DAY_COUNT :  max(len(sub) for sub in sub_user),
            MAX_DAY_LENGTH : max(len(ssub) for sub in sub_user for ssub in sub)})
        user_v = np.concatenate((user_v, uv))
    else:
        sub_user = user_array[38000:38022]
        uv = sess.run(user_vec, feed_dict={
            input_x : sub_user,
            MAX_DAY_COUNT :  max(len(sub) for sub in sub_user),
            MAX_DAY_LENGTH : max(len(ssub) for sub in sub_user for ssub in sub)})
        user_v = np.concatenate((user_v, uv))

np.save(r'D:\USERLOG\select_user\total\USERVEC.npy', user_v)


