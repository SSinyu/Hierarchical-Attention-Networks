import os
import pickle
import numpy as np
from operator import itemgetter


class text_dataloader(object):
    def __init__(self, data_path, max_sent, max_doc, mode, batch_size):
        with open(os.path.join(data_path, 'word_vocab.pkl'), 'rb') as f:
            self.word_vocab = pickle.load(f)
        with open(os.path.join(data_path, '{}_data.pkl'.format(mode)), 'rb') as f:
            data_ = pickle.load(f)
        self.x, self.y = data_['x'], data_['y']
        self.n_classes = len(np.unique(self.y))

        self.max_sent = max_sent
        self.max_doc = max_doc
        self.batch_size = batch_size

        assert len(self.y) % batch_size == 0, \
            "need to (data length % batch == 0)"

    def get_batch(self, shuffle=True):
        if shuffle:
            indices = np.arange(len(self.y))
            np.random.shuffle(indices)
            x = itemgetter(*indices)(self.x)
            y = itemgetter(*indices)(self.y)
        else:
            x, y = self.x, self.y

        for i in range(0, len(y)-self.batch_size, self.batch_size):
            sub_x = x[i:i+self.batch_size]
            sub_y = y[i:i+self.batch_size]

            batch_x = np.zeros([self.batch_size, self.max_doc, self.max_sent])
            for di, doc in enumerate(sub_x):
                doc_ = np.zeros([self.max_doc, self.max_sent])
               
                if len(doc) > self.max_doc:
                    doc = doc[-self.max_doc:]
                for si, sent in enumerate(doc):
                    
                    if len(sent) > self.max_sent:
                        sent = sent[-self.max_sent:]
                    doc_[si][:len(sent)] = sent
                batch_x[di] = doc_

            batch_y = self.make_one_hot(sub_y)

            batch_x.astype(int)
            batch_y.astype(int)
            yield batch_x, batch_y

    def make_one_hot(self, y_list):
        out = []
        for y in y_list:
            yi = [0] * self.n_classes
            yi[y-1] = 1
            out.append(yi)
        return np.array(out)
