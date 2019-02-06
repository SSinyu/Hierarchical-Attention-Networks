import re
import numpy as np
import pandas as pd
from collections import defaultdict
from nltk import word_tokenize, sent_tokenize
from torch.utils.data import Dataset, DataLoader


def clean_str(string):
    # by https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def get_dataloader(data_path, 
                   min_word_count, 
                   max_sent, 
                   max_doc, 
                   mode, 
                   train_ratio, 
                   batch_size, 
                   num_workers):
    dataloader = text_dataloader(data_path, min_word_count, max_sent, max_doc, mode, train_ratio)
    data_loader = DataLoader(dataset=dataloader, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return data_loader


class text_dataloader(Dataset):
    def __init__(self, data_path, min_word_count, max_sent, max_doc, mode, train_ratio):
        self.data_path = data_path
        self.min_word_count = min_word_count
        self.max_sent = max_sent
        self.max_doc = max_doc

        self.mode = mode
        self.train_ratio = train_ratio

        self.texts, self.labels, self.word_vocab = self.prep_text()
        self.word_list = list(self.word_vocab.keys())
        self.n_classes = len(np.unique(self.labels))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx] - 1

        batch_x = np.zeros([self.max_doc, self.max_sent])
        sent_length = []
        
        if len(text) > self.max_doc:
            text = text[-self.max_doc:]

        for si, sent in enumerate(text):
            word_v = batch_x[si]
            word_lst = sent.split(' ')
            if len(word_lst) > self.max_sent:
                word_lst = word_lst[-self.max_sent:]
            sent_length.append(len(word_lst))

            for wi, word in enumerate(word_lst):
                if word in self.word_list:
                    word_v[wi] = self.word_vocab[word]
                else:
                    word_v[wi] = 0 # UNK token
            batch_x[si] = word_v

        sent_length = sent_length + [0]*(self.max_doc - len(sent_length))
        doc_length = [len(text)]

        return batch_x, np.array([label]), np.array(sent_length), np.array(doc_length)

    def prep_text(self):
        # load
        raw_text = pd.read_csv(self.data_path)

        # label
        labels = list(raw_text['class'])

        # sentence tokenize
        texts = []
        for text_ in list(raw_text['text']):
            texts.append(sent_tokenize(text_))

        # word count, build hierarchical documents
        texts_hierarchy = []
        word_count = defaultdict(int)
        for sents in texts:
            clean_sent = []
            for s in sents:
                cleaning = clean_str(s)
                clean_sent.append(cleaning)
                words = word_tokenize(cleaning)
                for w in words:
                    word_count[w] += 1
            texts_hierarchy.append(clean_sent)

        raw_text['text'] = texts_hierarchy

        # build vocabulary, remove unique words
        word_vocab = {'UNK': 0}
        i = 1
        for w, f in word_count.items():
            if f > self.min_word_count:
                word_vocab[w] = i
                i += 1

        # train/test split
        train_ind = int(len(labels)*self.train_ratio)
        eval_ind = int(train_ind * 0.1)

        if self.mode == 'train':
            texts_hierarchy = texts_hierarchy[:train_ind-eval_ind]
            labels = labels[:train_ind-eval_ind]
        elif self.mode == 'eval':
            texts_hierarchy = texts_hierarchy[train_ind:train_ind+eval_ind]
            labels = labels[train_ind:train_ind+eval_ind]
        elif self.mode == 'test':
            texts_hierarchy = texts_hierarchy[train_ind:]
            labels = labels[train_ind:]
        else:
            raise ValueError

        return texts_hierarchy, labels, word_vocab
