import torch
import torch.nn as nn

class GloveLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.output = nn.Linear(hidden_size * 2, 1)
    def forward(self, token_ids, lengths):
        embedded = self.embedding(token_ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)
        combined = torch.cat([hidden[0], hidden[1]], dim=1)  
        return self.output(combined).squeeze(1)
