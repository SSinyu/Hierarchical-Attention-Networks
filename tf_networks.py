import tensorflow as tf
from tf_module import BiGRU, Attention_, fc_


class HierarchicalAttentionNet(object):
    def __init__(self, vocab_size, embed_size, hidden_size, n_classes, pre_embed=None, embed_fine_tune=False, n_layer=None, dropout_prob=None):
        super(HierarchicalAttentionNet, self).__init__()
        tf.reset_default_graph()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.n_classes = n_classes
        self.pre_embed = pre_embed
        self.embed_fine_tune = embed_fine_tune
        self.n_layer = n_layer
        self.dropout_prob = dropout_prob

        # placeholder
        self.batch_size = tf.placeholder(tf.int32, name='batch_size')
        self.x = tf.placeholder(tf.int32, shape=[None, None, None], name='x')
        self.y = tf.placeholder(tf.int32, shape=[None, n_classes], name='y')
        self.max_sent_length = tf.placeholder(tf.int32, name='max_sent_length')
        self.max_doc_length = tf.placeholder(tf.int32, name='max_doc_length')
        if pre_embed:
            self.embed_placeholder = tf.placeholder(tf.float32, shape=[self.vocab_size, self.embed_size])

        # word encoding
        with tf.variable_scope('word2vec'):
            word_embed = self.word2vec()

        # sentence encoding/attention
        with tf.variable_scope('sent2vec'):
            sent_vector = self.sent2vec(word_embed)

        # document encoding/attention
        with tf.variable_scope('doc2vec'):
            self.doc_vector = self.doc2vec(sent_vector)

        # fc
        self.out = fc_(self.doc_vector, self.n_classes)

        # loss
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=self.out))

        # classification



    def word2vec(self):
        if self.pre_embed:
            embed = tf.Variable(tf.constant(0.0, shape=[self.vocab_size, self.embed_size]), trainable=self.embed_fine_tune, name='embed')
            embed_init = embed.assign(self.embed_placeholder)
        else:
            embed = tf.Variable(tf.truncated_normal((self.vocab_size, self.embed_size)))
        word_embed = tf.nn.embedding_lookup(embed, self.x)
        return word_embed

    def sent2vec(self, word_vector):
        word_vector = tf.reshape(word_vector, shape=[-1, self.max_sent_length, self.embed_size])
        sent_encode = BiGRU(word_vector, self.hidden_size, self.dropout_prob, self.n_layer)
        sent_attn = Attention_(sent_encode, self.hidden_size*2) # bidirectional
        return sent_attn

    def doc2vec(self, sent_vector):
        sent_vector = tf.reshape(sent_vector, shape=[-1, self.max_doc_length, self.hidden_size*2])
        doc_encode = BiGRU(sent_vector, self.hidden_size, self.dropout_prob, self.n_layer)
        doc_attn = Attention_(doc_encode, self.hidden_size*2)
        return doc_attn
