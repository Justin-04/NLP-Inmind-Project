import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers=1, 
                 bidirectional=False, dropout=0.0, pretrained_weights=None, fine_tune=False):
        super().__init__()
        
        if pretrained_weights is not None:
            if fine_tune == True:
                self.embedding = nn.Embedding.from_pretrained(pretrained_weights, freeze=False)
            else:
                self.embedding = nn.Embedding.from_pretrained(pretrained_weights, freeze=True)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            
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

    def forward(self, x):
        embedded = self.embedding(x)
        out, (hn, cn) = self.lstm(embedded)
        last_time_step = out[:, -1, :] 
        logits = self.fc(last_time_step)
        return logits.squeeze(-1)