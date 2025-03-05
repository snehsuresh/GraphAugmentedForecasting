# lstm_only_model.py

import torch
import torch.nn as nn

class LSTMOnlyModel(nn.Module):
    def __init__(self, input_dim):
        super(LSTMOnlyModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size=32, batch_first=True)
        self.fc = nn.Linear(32, 1)

    def forward(self, x, adj=None):
        """
        x: Tensor of shape (batch, seq_length, features)
        The 'adj' parameter is ignored here, included for interface consistency.
        """
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])
