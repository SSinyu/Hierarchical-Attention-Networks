import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
from pytorch_networks import HierarchicalAttentionNet


class Solver(object):
    def __init__(self, args, train_loader=None, eval_loader=None, test_loader=None):
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.test_loader = test_loader

        if args.device:
            self.device = torch.device(args.device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

        self.batch_size = args.batch_size
        self.vocab_size = (train_loader.dataset.vocab_size if train_loader else test_loader.dataset.vocab_size)
        self.n_classes = train_loader.dataset.n_classes if train_loader else test_loader.dataset.n_classes
        self.hidden_size = args.hidden_size
        self.HAN = HierarchicalAttentionNet(vocab_size=self.vocab_size, hidden_size=self.hidden_size, n_classes=self.n_classes)
        self.HAN.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.HAN.parameters(), args.lr, [self.beta1, self.beta2])

    def save_model(self, iter_):
        f = os.path.join(self.save_path, "HAN_{}iter.ckpt".format(iter_))
        torch.save(self.HAN.state_dict(), f)

    def load_model(self, iter_, multigpu_=None):
        f = os.path.join(self.save_path, "HAN_{}iter.ckpt".format(iter_))
        if multigpu_:
            state_d = OrderedDict()
            for k,v in torch.load(f):
                n = k[7:]
                state_d[n] = v
            self.HAN.load_state_dict(state_d)
        else:
            self.HAN.load_state_dict(torch.load(f))

    def sort_tensor(self, x, y, sent_len, doc_len):
        # sort sent_lengths
        sent_len, sent_idx = sent_len.sort(1, descending=True)
        sorted_sent_x = torch.zeros(x.numpy().shape)
        for batch_i in range(self.batch_size):
            sorted_sent_x[batch_i] = x[batch_i][sent_idx[batch_i]]
        sent_len = sent_len.view(-1).to(self.device)
        sent_len, sent_idx = sent_len.sort(0, descending=True)

        # sort doc_lengths
        doc_len = doc_len.view(-1).to(self.device)
        if len(doc_len) > 1:
            doc_len, doc_idx = doc_len.sort(0, descending=True)
            x, y = sorted_sent_x[doc_idx].to(self.device), y[doc_idx].to(self.device)
        else: # len(doc_len) == 1
            x, y = sorted_sent_x.to(self.device), y.to(self.device)
        return x.long(), y.long(), sent_len.long(), sent_idx.long(), doc_len.long()

    def lr_decay(self):
        lr = self.lr * 0.5
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def train(self):
        train_losses, eval_losses = [], []
        total_iters = 0
        for epoch in range(1, self.num_epochs+1):
            self.HAN.train(True)

            for iter_, (x, y, sent_lengths, doc_lengths) in enumerate(self.train_loader):
                total_iters += 1
                x, y, sent_lengths, sent_idx, doc_lengths = self.sort_tensor(x, y, sent_lengths, doc_lengths)
                out = self.HAN(x, sent_lengths, sent_idx, doc_lengths)
                loss = self.criterion(out, y.view(-1))
                self.HAN.zero_grad()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_losses.append(loss.item())

                if total_iters % self.print_iters == 0:
                    print("EPOCH [{}/{}], ITER [{}/{} ({})] \nLOSS:{:.4f}".format(epoch, self.num_epochs, iter_+1, len(self.train_loader), total_iters, loss.item()))

                # evaluation
                if total_iters % self.eval_iters == 0:
                    self.HAN.train(False)
                    self.HAN.eval()
                    eval_loss = 0.0
                    for e_x, e_y, e_sent_len, e_doc_len in self.eval_loader:
                        e_x, e_y, e_sent_len, e_sent_idx, e_doc_len = self.sort_tensor(e_x, e_y, e_sent_len, e_doc_len)
                        e_out = self.HAN(e_x, e_sent_len, e_sent_idx, e_doc_len)
                        eval_loss += self.criterion(e_out, e_y.view(-1)).item()
                    eval_loss = eval_loss/len(self.eval_loader)
                    print("==== EVALUATION ITER[{}] \n==== LOSS:{:.4f}".format(total_iters,eval_loss))
                    eval_losses.append(eval_loss)
                    self.HAN.train(True)

                # lr decay
                if total_iters % self.decay_iters == 0:
                    self.lr_decay()

                # save
                if total_iters % self.save_iters == 0:
                    self.save_model(total_iters)
                    np.save(os.path.join(self.save_path, 'train_loss_{}_iter.npy'.format(total_iters)), np.array(train_losses))
                    np.save(os.path.join(self.save_path, 'eval_loss_{}_iter.npy'.format(total_iters)), np.array(eval_losses))

    def test(self):
        del self.HAN
        self.HAN = HierarchicalAttentionNet(vocab_size=self.vocab_size, hidden_size=self.hidden_size, n_classes=self.n_classes)
        self.HAN.to(self.device)
        self.load_model(self.test_iters)

        # accuracy
        correct, total = 0, 0
        with torch.no_grad():
            for x, y, sent_len, doc_len in self.test_loader:
                x, y, sent_len, sent_idx, doc_len = self.sort_tensor(x, y, sent_len, doc_len)

                out = self.HAN(x, sent_len, sent_idx, doc_len)
                _, pred = torch.max(out.data, 1)

                total += y.size(0)
                correct += (pred == y.view(-1)).sum().item()
        print('Accuracy of the network on the test data: {}%'.format(100 * correct / total))

        # figure
        import matplotlib.pyplot as plt
        import seaborn as sns
        div = 10
        train_loss = np.load(os.path.join(self.save_path, 'train_loss_{}_iter.npy'.format(self.test_iters)))
        eval_loss = np.load(os.path.join(self.save_path, 'eval_loss_{}_iter.npy'.format(self.test_iters)))

        fig, ax = plt.subplots(2,1,figsize=(14,10))
        sns.lineplot(range(len(train_loss)//div), [train_loss[i] for i in range(len(train_loss)) if i % div == 0], ax=ax[0])
        ax[0].set_title('Train loss', fontsize=20)
        #ax[1] = ax[0].twiny()
        sns.lineplot(range(len(eval_loss)), list(eval_loss), ax=ax[1], color='red')
        ax[1].set_title('Evaluation loss', fontsize=20)

        plt.savefig(os.path.join(self.save_path, 'loss_fig.png'))
