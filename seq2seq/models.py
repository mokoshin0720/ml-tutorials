import torch
import torch.nn as nn
import data

embedding_dim = 200 # 文字の埋め込み次元数
hidden_dim = 128 # LSTMの隠れ層サイズ
vocab_size = len(data.char2id)

class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=data.char2id[" "])
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

    # 最後の隠れ層のみ出力する
    def forward(self, sequence):
        embedding = self.word_embeddings(sequence)
        _, state = self.lstm(embedding)
        return state

class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=data.char2id[" "])
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.hidden2linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, sequence, encoder_state):
        embedding = self.word_embeddings(sequence)
        output, state = self.lstm(embedding, encoder_state)
        output = self.hidden2linear(output)
        return output, state