import torch.nn as nn

class GloveGRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            batch_first=True
        )
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, token_ids, lengths):
        embedded = self.embedding(token_ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.gru(packed)
        hidden = hidden[-1]
        return self.output(hidden).squeeze(1)