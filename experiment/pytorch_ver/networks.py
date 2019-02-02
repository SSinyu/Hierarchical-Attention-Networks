import torch
import torch.nn as nn
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
    def __init__(self, vocab_size, embed_size, hidden_size, pre_embed=None, embed_finetune=None):
        super(HierarchicalAttentionNet, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        # word level
        self.embed = nn.Embedding(vocab_size, embed_size)
        if pre_embed:
            self.weight = nn.Parameter(pre_embed, requires_grad=(True if embed_finetune else False))
        self.word_rnn = nn.GRU(embed_size, hidden_size, bidirectional=True)
        self.word_att = AttentionLayer(hidden_size)
        # sent level
        self.sent_rnn = nn.GRU(hidden_size*2, hidden_size, bidirectional=True)
        self.sent_att = AttentionLayer(hidden_size)
        # fc, softmax
        self.linear = nn.Linear(hidden_size*2, vocab_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, sent_lengths, doc_lengths):
        # x shape -> (batch, max_doc, max_sent)
        # 'sent_lengths', 'doc_lengths' must sorted in decreasing order

        assert x.shape[2] == max(sent_lengths)
        assert x.shape[1] == max(doc_lengths)
        max_sent_len = x.shape[2]
        max_doc_len = x.shape[1]

        # word embedding
        word_embed = self.embed(x) # (batch, max_doc, max_sent, embed_size)

        # word encoding
        word_embed = word_embed.view(-1, max_sent_len, self.embed_size)  # (-1(batch), max_sent_len, embed_size)
        word_packed_input = pack_padded_sequence(word_embed, sent_lengths.cpu().numpy(), batch_first=True)
        word_packed_output, _ = self.word_rnn(word_packed_input)
        word_encode, _ = pad_packed_sequence(word_packed_output, batch_first=True) # (batch, max_sent_len, hidden_size*2)

        # word attention
        sent_vector, sent_alpha = self.word_att(word_encode) # (batch, hidden_size*2)

        # sent encoding
        sent_vector = sent_vector.view(-1, max_doc_len, self.hidden_size*2) # (batch, max_doc_len, hidden_size*2)
        sent_packed_input = pack_padded_sequence(sent_vector, doc_lengths.cpu().numpy(), batch_first=True)
        sent_packed_output, _ = self.sent_rnn(sent_packed_input)
        sent_encode, _ = pad_packed_sequence(sent_packed_output, batch_first=True) # (batch, max_doc_len, hidden_size*2)

        # sent attention
        doc_vector, doc_alpha = self.sent_att(sent_encode) # (batch, hidden_size*2)

        # fc, softmax
        out = self.softmax(self.linear(doc_vector))

        return out, doc_vector
