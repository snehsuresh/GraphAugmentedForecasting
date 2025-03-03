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
        
        # Graph Convolutional layers (stacked)
        self.gcn_layers = nn.ModuleList()
        # Assume initial node feature dimension is 1 (we use anomaly frequency)
        self.gcn_layers.append(GCNConv(in_channels=1, out_channels=gcn_hidden_dim))
        for _ in range(gcn_layers - 1):
            self.gcn_layers.append(GCNConv(in_channels=gcn_hidden_dim, out_channels=gcn_hidden_dim))
        
        # LSTM for temporal modeling (stacked)
        self.lstm = nn.LSTM(input_size=1, hidden_size=lstm_hidden_dim,
                            num_layers=lstm_layers, batch_first=True)
        
        # Final fully-connected layer: combining LSTM and GCN outputs
        self.fc = nn.Linear(lstm_hidden_dim + gcn_hidden_dim, forecast_horizon)
    
    def forward(self, x, graph_data):
        """
        x: tensor of shape (batch_size, num_nodes, input_length) - time series window
        graph_data: PyG Data object with x (node features) and edge_index.
        """
        batch_size = x.size(0)
        
        # --- Temporal Part: Process each node's time series with LSTM ---
        x_expanded = x.unsqueeze(-1)  # (B, num_nodes, input_length, 1)
        lstm_out = []
        for node in range(self.num_nodes):
            node_series = x_expanded[:, node, :, :]  # (B, input_length, 1)
            _, (h_n, _) = self.lstm(node_series)
            lstm_out.append(h_n[-1])  # (B, lstm_hidden_dim)
        lstm_out = torch.stack(lstm_out, dim=1)  # (B, num_nodes, lstm_hidden_dim)
        
        # --- Graph Part: Process node features with GCN stack ---
        node_features = graph_data.x  # (num_nodes, 1)
        edge_index = graph_data.edge_index
        gcn_out = node_features
        for gcn in self.gcn_layers:
            gcn_out = F.relu(gcn(gcn_out, edge_index))
        # Expand to batch dimension
        gcn_out = gcn_out.unsqueeze(0).expand(batch_size, -1, -1)  # (B, num_nodes, gcn_hidden_dim)
        
        # --- Combine ---
        combined = torch.cat([lstm_out, gcn_out], dim=-1)  # (B, num_nodes, lstm_hidden_dim+gcn_hidden_dim)
        logits = self.fc(combined)  # (B, num_nodes, forecast_horizon)
        if self.forecast_horizon == 1:
            logits = logits.squeeze(-1)
        return logits

# For testing purpose:
if __name__ == "__main__":
    import torch
    from torch_geometric.data import Data
    num_nodes = 10
    dummy_node_features = torch.ones((num_nodes, 1))
    dummy_edge_index = torch.tensor([[0,1,2,3,4],[1,2,3,4,5]], dtype=torch.long)
    dummy_graph = Data(x=dummy_node_features, edge_index=dummy_edge_index)
    
    dummy_x = torch.rand((8, num_nodes, 20))
    model = GraphAugmentedForecastingModel(num_nodes=num_nodes, input_length=20)
    out = model(dummy_x, dummy_graph)
    print("Output shape:", out.shape)
