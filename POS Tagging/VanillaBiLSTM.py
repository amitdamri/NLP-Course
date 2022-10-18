import torch
import torch.nn as nn


class VanillaBiLSTM(nn.Module):
    """This class represents the Vanilla BiLSTM model.
    """

    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers, dropout, bidirectional, tagset_size, pad_idx):
        super(VanillaBiLSTM, self).__init__()
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=n_layers,
                            # dropout=dropout if n_layers > 1 else 0,
                            dropout=0,
                            bidirectional=bidirectional,
                            batch_first=True)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, tagset_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, sentence):
        embeds = self.embedding(sentence)
        outputs, (hidden, cell) = self.lstm(embeds)
        predictions = self.hidden2tag(self.dropout(outputs))
        return predictions
