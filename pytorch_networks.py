import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class AttentionLayer(nn.Module):
    def __init__(self, hidden_size=100, bi=True):
        super(AttentionLayer, self).__init__()
        self.hidden_size = (hidden_size*2 if bi else hidden_size)
        self.linear_ = nn.Linear(self.hidden_size, self.hidden_size)
        self.tanh_ = nn.Tanh()
        self.softmax_ = nn.Softmax(dim=1)

    def forward(self, x):
        u_context = torch.nn.Parameter(torch.FloatTensor(self.hidden_size).normal_(0, 0.01)).cuda()
        h = self.tanh_(self.linear_(x)).cuda()
        alpha = self.softmax_(torch.mul(h, u_context).sum(dim=2, keepdim=True))  # (x_dim0, x_dim1, 1)
        attention_output = torch.mul(x, alpha).sum(dim=1)  # (x_dim0, x_dim2)
        return attention_output, alpha


class HierarchicalAttentionNet(nn.Module):
    def __init__(self, vocab_size, hidden_size, n_classes, pre_embed=None, embed_fine_tune=True):
        super(HierarchicalAttentionNet, self).__init__()
        self.embed_size = hidden_size * 2
        self.hidden_size = hidden_size
        self.n_classes = n_classes

        # word level
        self.embed = nn.Embedding(vocab_size, hidden_size)
        if pre_embed:
            self.weight = nn.Parameter(pre_embed, requires_grad=(True if embed_fine_tune else False))
        else:
            init.normal_(self.embed.weight, std=0.01)
        self.word_rnn = nn.GRU(hidden_size*2, hidden_size, bidirectional=True)
        self.word_att = AttentionLayer(hidden_size)

        # sent level
        self.sent_rnn = nn.GRU(hidden_size*2, hidden_size, bidirectional=True)
        self.sent_att = AttentionLayer(hidden_size)

        # fully connected
        self.linear = nn.Linear(hidden_size*2, self.n_classes)

    def forward(self, x, sent_lengths, sent_idx, doc_lengths):
        # x shape -> (batch, max_doc, max_sent)
        # 'sent_lengths', 'doc_lengths' must sorted in decreasing order
        # sent_lengths -> (batch*max_doc, )
        # doc_lengths -> (batch, )

        max_sent_len = x.shape[2]
        max_doc_len = x.shape[1]

        # word embedding
        word_embed = self.embed(x) # (batch, max_doc, max_sent, embed_size)

        # split sent_length=0 part (length must > 0 in 'pack_padded_sequence')
        word_embed = word_embed.view(-1, max_sent_len, self.embed_size) # (batch*max_doc, max_sent, embed_size)
        word_embed = word_embed[sent_lengths] # sort length
        non_zero_idx = len(torch.nonzero(sent_lengths))
        word_embed_sltd = word_embed[:non_zero_idx]
        word_embed_remain = word_embed[non_zero_idx:]
        sent_lengths_sltd = sent_lengths[:non_zero_idx]

        # word encoding
        word_packed_input = pack_padded_sequence(word_embed_sltd, sent_lengths_sltd.cpu().numpy(), batch_first=True)
        word_packed_output, _ = self.word_rnn(word_packed_input)
        word_encode, _ = pad_packed_sequence(word_packed_output, batch_first=True) # (non_zero_idx, max_sent_len, hidden_size*2)

        # merge sent_length=0 part
        if word_encode.shape[1] != word_embed.shape[1]:
            word_encode_dim = word_encode.shape[1]
            word_embed_remain = word_embed_remain[:, :word_encode_dim, :]
            max_sent_len = word_encode_dim
        word_encode = torch.cat((word_encode, word_embed_remain), 0) # (batch*max_doc, max_sent, hidden_size*2)

        # unsort
        sort_idx = torch.LongTensor([[[i]*self.embed_size]*max_sent_len for i in sent_idx.cpu().detach().numpy()])
        unsort_word_encode = word_encode.new(*word_encode.size())
        unsort_word_encode = unsort_word_encode.scatter_(0, sort_idx.cuda(), word_encode)

        # word attention
        sent_vector, sent_alpha = self.word_att(unsort_word_encode) # (batch*max_doc, hidden_size*2)

        # sent encoding
        sent_vector = sent_vector.view(-1, max_doc_len, self.hidden_size*2) # (batch, max_doc_len, hidden_size*2)
        sent_packed_input = pack_padded_sequence(sent_vector, doc_lengths.cpu().numpy(), batch_first=True)
        sent_packed_output, _ = self.sent_rnn(sent_packed_input)
        sent_encode, _ = pad_packed_sequence(sent_packed_output, batch_first=True) # (batch, max_doc_len, hidden_size*2)

        # sent attention
        doc_vector, doc_alpha = self.sent_att(sent_encode) # (batch, hidden_size*2)

        # fc, softmax
        out = self.linear(doc_vector)

        return out
