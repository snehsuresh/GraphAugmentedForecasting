import torch
import torch.nn as nn

class NodeLSTMModel(nn.Module):
    def __init__(self, input_len, lstm_hidden=64, forecast_horizon=5):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=lstm_hidden, batch_first=True)
        self.fc = nn.Linear(lstm_hidden, forecast_horizon)

    def forward(self, x):
        x = x.unsqueeze(-1)  # Add feature dim (needed for LSTM)
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])
