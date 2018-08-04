import torch
import torch.nn as nn
import torch.nn.functional as F


class MyLSTM(nn.Module):
    def __init__(self, vocab_size, tag_size, EMBED_DIM=300, HIDDEN_DIM=1000, BATCH_SIZE=32):
        super().__init__()
        self.batch_size = BATCH_SIZE
        self.embed_dim = EMBED_DIM
        self.hidden_dim = HIDDEN_DIM
        self.embed = nn.Embedding(vocab_size, EMBED_DIM)
        self.lstm = nn.LSTM(EMBED_DIM, HIDDEN_DIM)
        self.hidden = self.init_hidden()
        self.hidden2tag = nn.Linear(HIDDEN_DIM, tag_size)
        self.tag_size = tag_size

    def init_hidden(self):
        return (torch.zeros(1, self.batch_size, self.hidden_dim),
                torch.zeros(1, self.batch_size, self.hidden_dim))

    def forward(self, x):
        self.hidden = self.init_hidden()
        embeds = self.embed(x)
        lstm_out, self.hidden = self.lstm(
            embeds.view(1, self.batch_size, self.embed_dim), self.hidden)
        tag_space = self.hidden2tag(lstm_out)
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores[0]