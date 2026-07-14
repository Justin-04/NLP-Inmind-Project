import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers=1, 
                 bidirectional=False, dropout=0.0, pretrained_weights=None, fine_tune=False):
        super().__init__()
        
        # Embedding Layer
        if pretrained_weights is not None:
            # Use pre-trained embeddings
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_weights,
                freeze=not fine_tune,
                padding_idx=0
            )
        else:
            # Train embeddings from scratch randomly
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
            
        # LSTM Layer
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0
        )
        num_directions = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_size * num_directions, 1)

    def forward(self, x, lengths=None):
        # Embeddings
        embedded = self.embedding(x)
        
        # LSTM Processing
        if lengths is not None:
            packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
            _, (hn, _) = self.lstm(packed)
        else:
            _, (hn, _) = self.lstm(embedded)
            
        if self.lstm.bidirectional:
            last_time_step = torch.cat((hn[-2], hn[-1]), dim=-1)
        else:
            last_time_step = hn[-1]

        # Final prediction score
        logits = self.fc(last_time_step)
        return logits.squeeze(-1)