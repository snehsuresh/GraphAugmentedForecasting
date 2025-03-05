# graph_augmented_model.py

import torch
import torch.nn as nn

class GraphAugmentedModel(nn.Module):
    def __init__(self, input_dim):
        super(GraphAugmentedModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size=32, batch_first=True)
        # Placeholder for a graph-based component – later you can replace this with your actual GNN layer.
        self.graph_fc = nn.Linear(32, 32)
        self.fc = nn.Linear(32, 1)

    def forward(self, x, adj=None):
        """
        x: Tensor of shape (batch, seq_length, features)
        adj: Optional adjacency matrix for graph operations (currently not used)
        """
        out, _ = self.lstm(x)
        # Use the last time-step output
        lstm_out = out[:, -1, :]
        # Apply the placeholder graph transformation (ignoring adj for now)
        graph_out = self.graph_fc(lstm_out)
        return self.fc(graph_out)
