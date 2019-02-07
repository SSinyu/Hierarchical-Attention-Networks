import os
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader


def get_dataloader(data_path, max_sent, max_doc, mode, batch_size, num_workers):
    dataloader = text_dataloader(data_path, max_sent, max_doc, mode)
    data_loader = DataLoader(dataset=dataloader, 
                             batch_size=batch_size, 
                             shuffle=(True if mode=='train' else False), 
                             num_workers=num_workers,
                             drop_last=True)
    return data_loader


class text_dataloader(Dataset):
    def __init__(self, data_path, max_sent, max_doc, mode):
        self.data_path = data_path
        self.max_sent = max_sent
        self.max_doc = max_doc

        self.mode = mode

        # load
        with open(os.path.join(data_path, 'word_vocab.pkl'), 'rb') as f:
            self.word_vocab = pickle.load(f)
        with open(os.path.join(data_path, '{}_data.pkl'.format(mode)), 'rb') as f:
            data_ = pickle.load(f)
           
        self.x, self.y = data_['x'], data_['y']
        
        self.vocab_size = len(self.word_vocab)
        self.n_classes = len(np.unique(self.y))
        
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        text = self.x[idx]
        label = self.y[idx] - 1

        batch_x = np.zeros([self.max_doc, self.max_sent])
        sent_length = []
        
        if len(text) > self.max_doc:
            text = text[-self.max_doc:]

        for si, sent in enumerate(text):
            if len(sent) > self.max_sent:
                sent = sent[-self.max_sent:]
            batch_[si][:len(sent)] = sent
            sent_length.append(len(sent))
        
        sent_length = sent_length + [0]*(self.max_doc - len(sent_length))
        doc_length = [len(text)]

        return batch_x, np.array([label]), np.array(sent_length), np.array(doc_length)
