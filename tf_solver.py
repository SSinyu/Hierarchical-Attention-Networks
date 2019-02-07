import os
import tensorflow as tf
from tf_networks import HierarchicalAttentionNet


class Solver(object):
    def __init__(self, args, train_loader=None, eval_loader=None, test_loader=None):
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.test_loader = test_loader

        self.save_path = args.save_path

        self.max_sent = args.max_sent
        self.max_doc = args.max_doc

        self.num_epochs = args.num_epochs
        self.print_iters = args.print_iters
        self.eval_iters = args.eval_iters
        self.decay_iters = args.decay_iters
        self.save_iters = args.save_iters
        self.test_iters = args.test_iters
        self.lr = args.lr
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.clip = args.clip

        self.batch_size = args.batch_size
        self.hidden_size = args.hidden_size
        self.vocab_size = len(train_loader.word_vocab) if train_loader else len(test_loader.word_vocab)
        self.n_classes = train_loader.n_classes if train_loader else test_loader.n_classes

        self.HAN = HierarchicalAttentionNet(self.vocab_size, self.hidden_size, self.hidden_size, self.n_classes, embed_fine_tune=True)
        self.saver = tf.train.Saver(tf.global_variables())
        self.saver.export_meta_graph(os.path.join(self.save_path, 'HAN_model.meta'))

        # gradient clipping
        self.global_step = tf.Variable(0, trainable=False)
        optim = tf.train.AdamOptimizer(self.lr)
        tv_ = tf.trainable_variables()
        gradient_, _ = tf.clip_by_global_norm(tf.gradients(self.HAN.loss, tv_), self.clip)
        self.optimizer = optim.apply_gradients(tuple(zip(gradient_, tv_)), global_step=self.global_step)

    def train(self):
        with tf.Session() as sess:
            summary_loss = tf.summary.scalar('loss', self.HAN.loss)
            summary_train = tf.summary.merge([summary_loss])
            summary_eval = tf.summary.merge([summary_loss])
            summary_train_writer = tf.summary.FileWriter(os.path.join(self.save_path, 'summary', 'train'), sess.graph)
            summary_eval_writer = tf.summary.FileWriter(os.path.join(self.save_path, 'summary', 'eval'), sess.graph)

            sess.run(tf.global_variables_initializer())

            total_iters = 0
            for epoch in range(1, self.num_epochs):
                for iter_, (x, y) in enumerate(self.train_loader.get_batch()):
                    total_iters += 1
                    feed_d = {self.HAN.x:x,
                              self.HAN.y:y,
                              self.HAN.max_sent_length:self.max_sent,
                              self.HAN.max_doc_length:self.max_doc}
                    _, step, summaries = sess.run([self.optimizer, self.global_step, summary_train], feed_dict=feed_d)
                    summary_train_writer.add_summary(summaries, step)

                    if total_iters % self.print_iters == 0:
                        train_loss = sess.run(self.HAN.loss, feed_dict=feed_d)
                        print("EPOCH [{}/{}], ITER [{}/{} ({})] \nLOSS:{:.4f}".format(epoch, self.num_epochs, iter_+1, (len(self.train_loader.y)//self.batch_size), total_iters, train_loss))

                    # evaluation
                    if total_iters % self.eval_iters == 0:
                        eval_loss = 0.0
                        for e_x, e_y in self.eval_loader.get_batch():
                            e_feed_d = {self.HAN.x:e_x,
                                        self.HAN.y:e_y,
                                        self.HAN.max_sent_length:self.max_sent,
                                        self.HAN.max_doc_length:self.max_doc}
                            e_summaries, e_loss = sess.run([summary_eval, self.HAN.loss], feed_dict=e_feed_d)
                            eval_loss += e_loss
                        eval_loss /= (len(self.eval_loader.y)//self.batch_size)
                        print("==== EVALUATION ITER[{}] \n==== LOSS:{:.4f}".format(total_iters, eval_loss))
                        summary_eval_writer.add_summary(e_summaries, step)

                    # save
                    if total_iters % self.save_iters == 0:
                        self.saver.save(sess, os.path.join(self.save_path, 'HAN_iter.ckpt'), global_step=self.global_step)

    def test(self, iter_):
        del self.HAN
        del self.saver
        tf.reset_default_graph()

        HAN = HierarchicalAttentionNet(self.vocab_size, self.hidden_size, self.hidden_size, self.n_classes)

        f = os.path.join(self.save_path, 'HAN_iter.ckpt-{}'.format(iter_))
        self.saver = tf.train.Saver()

        with tf.Session() as sess:
            self.saver.restore(sess, f)

            # accuracy
            correct, total = 0, 0
            for i, (x, y) in enumerate(self.test_loader.get_batch(False)):
                feed_d = {HAN.x:x,
                          HAN.y:y,
                          HAN.max_sent_length:self.max_sent,
                          HAN.max_doc_length:self.max_doc}
                out = sess.run(HAN.out, feed_dict=feed_d)

                label = tf.argmax(y, axis=1)
                pred = tf.argmax(out, axis=1)
                acc = sess.run(tf.reduce_sum(tf.cast(tf.equal(pred, label), dtype=tf.int32)), feed_dict=feed_d)
                total += self.test_loader.batch_size
                correct += acc

                printProgressBar(i, len(self.test_loader.y)//100, prefix='calculate accuracy...', suffix='Complete', length=25)
            print('\n')
            print('Accuracy of the network on the test data: {:.4f}%'.format(100 * correct / total))


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill=' '):
    # by https://gist.github.com/snakers4/91fa21b9dda9d055a02ecd23f24fbc3d
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '=' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    if iteration == total:
        print()
