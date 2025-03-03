import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GraphAugmentedForecastingModel(nn.Module):
    def __init__(self, num_nodes, input_length,
                 lstm_hidden_dim=64, gcn_hidden_dim=32,
                 forecast_horizon=5, lstm_layers=2, gcn_layers=3):
        super(GraphAugmentedForecastingModel, self).__init__()
        
        self.num_nodes = num_nodes
        self.input_length = input_length
        self.forecast_horizon = forecast_horizon
        
        # Graph Convolution Stack
        self.gcn_layers = nn.ModuleList()
        self.gcn_layers.append(GCNConv(in_channels=1, out_channels=gcn_hidden_dim))
        for _ in range(gcn_layers - 1):
            self.gcn_layers.append(GCNConv(in_channels=gcn_hidden_dim, out_channels=gcn_hidden_dim))

        # Temporal (LSTM) Stack
        self.lstm = nn.LSTM(input_size=1, hidden_size=lstm_hidden_dim,
                            num_layers=lstm_layers, batch_first=True)

        # Final fully connected layer to combine LSTM + GCN output
        self.fc = nn.Linear(lstm_hidden_dim + gcn_hidden_dim, forecast_horizon)

    def forward(self, x, graph_data):
        """
        x: (batch_size, num_nodes, input_length) - Time series windows
        graph_data: PyG Data object with edge_index and node features
        """
        batch_size = x.size(0)

        # Process each node's time series with the LSTM
        x = x.unsqueeze(-1)  # (batch_size, num_nodes, input_length, 1)

        lstm_out = []
        for node in range(self.num_nodes):
            node_series = x[:, node, :, :]  # (batch_size, input_length, 1)
            _, (h_n, _) = self.lstm(node_series)
            lstm_out.append(h_n[-1])  # Final LSTM layer output

        lstm_out = torch.stack(lstm_out, dim=1)  # (batch_size, num_nodes, lstm_hidden_dim)

        # Process the graph features with GCN
        node_features = graph_data.x  # (num_nodes, 1)
        edge_index = graph_data.edge_index

        gcn_out = node_features
        for gcn in self.gcn_layers:
            gcn_out = F.relu(gcn(gcn_out, edge_index))

        # Expand node features to batch size
        gcn_out = gcn_out.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, num_nodes, gcn_hidden_dim)

        # Combine LSTM and GCN output for each node
        combined = torch.cat([lstm_out, gcn_out], dim=-1)  # (batch_size, num_nodes, lstm_hidden_dim+gcn_hidden_dim)

        logits = self.fc(combined)  # (batch_size, num_nodes, forecast_horizon)

        # Squeeze if forecasting horizon = 1 (optional for simplicity)
        if self.forecast_horizon == 1:
            logits = logits.squeeze(-1)

        return logits
