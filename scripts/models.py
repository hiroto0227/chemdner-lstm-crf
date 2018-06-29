import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class LSTMModel:
    def __init__(self):
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def load(self, filepath):
        pass

    def forward(self, x):
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(
        embeds.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores





class CRFModel:

    def __init__(self):
        self.crf_tagger = pycrfsuite.Tagger()

    def load(self, filepath):
        self.crf_tagger.open(filepath)

    def predict(self, features):
        return self.crf_tagger.tag(features)
