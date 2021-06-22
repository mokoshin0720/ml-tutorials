import torch
import torch.nn as nn
from data import char2id

# Encoderクラス
class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=char2id[" "])
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)

    def forward(self, sequence):
        embedding = self.word_embeddings(sequence)
        hs, h = self.gru(embedding)
        return hs, h

class AttentionDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, batch_size):
        super(AttentionDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=char2id[" "])
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.hidden2linear = nn.Linear(hidden_dim*2, vocab_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, sequence, hs, h):
        embedding = self.word_embeddings(sequence)
        output, state = self.gru(embedding, h)
        t_output = torch.transpose(output, 1, 2)
        s = torch.bmm(hs, t_output)
        attention_weight = self.softmax(s)
        c = torch.zeros(self.batch_size, 1, self.hidden_dim)
        for i in range(attention_weight.size()[2]):
            unsq_weight = attention_weight[:, :, i].unsqueeze(2)
            weighted_hs = hs * unsq_weight
            weight_sum = torch.sum(weighted_hs, axis=1).unsqueeze(1)
            c = torch.cat([c, weight_sum], dim=1)
        c = c[:, 1:, :]

        output = torch.cat([output, c], dim=2)
        output = self.hidden2linear(output)
        return output, state, attention_weight