import pickle
import numpy as np
from collections import Counter
from torch.utils.data import Dataset, DataLoader

class log_dataloader(Dataset):
    def __init__(self, input_path, vocab_path, min_day_length, max_day_length, per_user, agg_min=10, target_time=3, mode='train'):
        with open(input_path, 'rb') as f:
            self.log_input = pickle.load(f)
        with open(vocab_path, 'rb') as f:
            self.vocab = pickle.load(f)

        self.vocab_size = len(self.vocab)
        self.min_day_length = min_day_length
        self.max_day_length = max_day_length
        self.per_user = per_user
        self.target_length = target_time * 6 # (10m x 6 = 1hour)
        self.day_length = (60 // agg_min) * 24

        self.mode = mode

    def __len__(self):
        return len(self.log_input)

    def __getitem__(self, idx):
        user_log = self.log_input[idx]

        if self.mode == 'train':
            batch_x, batch_y, day_lengths = [], [], []
            for _ in range(self.per_user):
                day_size = np.random.randint(self.min_day_length, self.max_day_length+1)
                day_lengths.append(day_size)

                start_ind = np.random.randint(0, len(user_log) - self.day_length * day_size - self.target_length)
                end_ind = start_ind + self.day_length * day_size

                subset_x_max = [0 for _ in range(self.day_length * self.max_day_length)]
                subset_x = user_log[start_ind:end_ind]
                subset_x_max[0:len(subset_x)] = subset_x
                subset_hx = [subset_x_max[sub_x_ind:(sub_x_ind+self.day_length)] for sub_x_ind in range(0, len(subset_x_max), self.day_length)]
                batch_x.append(subset_hx)

                subset_y = user_log[end_ind:(end_ind+self.target_length)]
                batch_y.append(subset_y)
            return np.array(batch_x), self.seq_to_ratio(batch_y), np.array(day_lengths)

        elif self.mode == 'test':
            hx = [user_log[x_ind:(x_ind+self.day_length)] for x_ind in range(0, len(user_log), self.day_length)]
            day_len = [self.max_day_length]
            return np.array(hx), np.array(day_len)


    def seq_to_ratio(self, seq):
        ratio_list = []
        for i, seq_3h in enumerate(seq):
            counter_ = Counter(seq_3h).most_common(self.vocab_size)
            labels = [0] * self.vocab_size
            for ch, cnt in counter_:
                labels[ch] = cnt/18
            ratio_list.append(labels)
        return np.array(ratio_list)


def get_loader(input_path, vocab_path, min_day_length=6, max_day_length=12, per_user=3, batch_size=5, agg_min=10, target_time=3, num_worker=7, mode='train'):
    log_dataset = log_dataloader(input_path, vocab_path, min_day_length, max_day_length, per_user, agg_min, target_time, mode)
    data_loader = DataLoader(dataset=log_dataset, batch_size=batch_size, shuffle=True, num_workers=num_worker)
    return data_loader
