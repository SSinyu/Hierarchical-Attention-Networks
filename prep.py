import os
import re
import pickle
import argparse
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from nltk import word_tokenize, sent_tokenize

import nltk
nltk.download('punkt')

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


def prep_text(args):
    # load
    raw_text = pd.read_csv(args.data_path)

    # label
    labels = list(raw_text['class'])

    # sentence tokenize
    texts = []
    for text_ in list(raw_text['text']):
        texts.append(sent_tokenize(text_))

    # word count, build hierarchical documents
    texts_hierarchy = []
    word_count = defaultdict(int)
    print('text cleaning...')
    for sents in texts:
        clean_sent = []
        for s in sents:
            cleaning = clean_str(s)
            clean_sent.append(cleaning)
            words = word_tokenize(cleaning)
            for w in words:
                word_count[w] += 1
        texts_hierarchy.append(clean_sent)

    # build vocabulary, remove unique words
    word_vocab = {'UNK':0}
    i = 1
    for w, f in word_count.items():
        if f > args.min_word_count:
            word_vocab[w] = i
            i += 1
    
    # text encoding
    print('text encoding...')
    encode_text = []
    for sent in tqdm(texts_hierarchy):
        encode_doc = []
        for s in sent:
            encode_ = [word_vocab.get(word, 0) for word in word_tokenize(s)]
            encode_doc.append(encode_)
        encode_text.append(encode_doc)

    # train/eval/test split
    train_ind = int(len(labels) * args.train_ratio)
    eval_ind = int(len(labels) * args.eval_ratio)
    train_x, train_y = encode_text[:train_ind], labels[:train_ind]
    eval_x, eval_y = encode_text[train_ind:train_ind+eval_ind], labels[train_ind:train_ind+eval_ind]
    test_x, test_y = encode_text[train_ind+eval_ind:], labels[train_ind+eval_ind:]

    train_data = {'x':train_x, 'y':train_y}
    eval_data = {'x':eval_x, 'y':eval_y}
    test_data = {'x':test_x, 'y':test_y}

    # save
    if args.save_path:
        path_ = args.save_path
    else:
        path_ = os.path.split(args.data_path)[0]

    save_lst = [train_data, eval_data, test_data, word_vocab]
    save_name = ['train_data.pkl', 'eval_data.pkl', 'test_data.pkl', 'word_vocab.pkl']

    for file_, name_ in zip(save_lst, save_name):
        with open(os.path.join(path_, '{}'.format(name_)), 'wb') as f:
            pickle.dump(file_, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./sample_text.csv')
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--min_word_count', type=int, default=5)
    parser.add_argument('--train_ratio', type=float, default=.7)
    parser.add_argument('--eval_ratio', type=float, default=.1)

    args = parser.parse_args()
    prep_text(args)
