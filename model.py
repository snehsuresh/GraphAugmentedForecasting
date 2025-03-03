import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GraphAugmentedForecastingModel(nn.Module):
    def __init__(self, 
                 num_nodes, 
                 input_length, 
                 lstm_hidden_dim=64, 
                 gcn_hidden_dim=32, 
                 combined_dim=64,
                 forecast_horizon=1):
        super(GraphAugmentedForecastingModel, self).__init__()
        self.num_nodes = num_nodes
        self.input_length = input_length
        self.forecast_horizon = forecast_horizon
        
        # Graph Convolution Network to compute node embeddings.
        # Assume initial node feature dimension is 16 (as set in preprocessing)
        self.gcn1 = GCNConv(in_channels=16, out_channels=gcn_hidden_dim)
        self.gcn2 = GCNConv(in_channels=gcn_hidden_dim, out_channels=gcn_hidden_dim)
        
        # LSTM for temporal modeling of each node's time series.
        # We process each node's sequence independently.
        self.lstm = nn.LSTM(input_size=1, hidden_size=lstm_hidden_dim, batch_first=True)
        
        # Final fully connected layer to combine LSTM output and GCN embedding.
        self.fc = nn.Linear(lstm_hidden_dim + gcn_hidden_dim, forecast_horizon)
        
    def forward(self, x, graph_data):
        """
        x: tensor of shape (batch_size, num_nodes, input_length) -> time series inputs
        graph_data: PyTorch Geometric data object containing x (node features) and edge_index.
        """
        batch_size = x.size(0)
        num_nodes = self.num_nodes
        
        # --- Graph Part ---
        # Compute node embeddings using the static graph.
        node_features = graph_data.x  # shape: (num_nodes, 16)
        edge_index = graph_data.edge_index
        
        gcn_out = F.relu(self.gcn1(node_features, edge_index))
        node_emb = F.relu(self.gcn2(gcn_out, edge_index))  # shape: (num_nodes, gcn_hidden_dim)
        # Expand node embeddings to have batch dimension.
        node_emb_expanded = node_emb.unsqueeze(0).expand(batch_size, -1, -1)  # shape: (batch_size, num_nodes, gcn_hidden_dim)
        
        # --- Temporal Part ---
        # Process each node’s time series with LSTM.
        # First, we need to add a feature dimension to x.
        x_lstm = x.unsqueeze(-1)  # now shape: (batch_size, num_nodes, input_length, 1)
        # We'll process each node independently.
        lstm_out_list = []
        for node in range(num_nodes):
            # Get time series for one node over the batch: shape (batch_size, input_length, 1)
            node_series = x_lstm[:, node, :, :]
            # LSTM expects (batch_size, seq_length, feature_dim)
            _, (h_n, _) = self.lstm(node_series)
            lstm_out_list.append(h_n[-1])  # last layer hidden state, shape: (batch_size, lstm_hidden_dim)
        
        # Stack over nodes -> shape: (batch_size, num_nodes, lstm_hidden_dim)
        lstm_out = torch.stack(lstm_out_list, dim=1)
        
        # --- Combine ---
        combined = torch.cat([lstm_out, node_emb_expanded], dim=-1)  # shape: (batch_size, num_nodes, lstm_hidden_dim+gcn_hidden_dim)
        output = self.fc(combined)  # shape: (batch_size, num_nodes, forecast_horizon)
        
        # For forecast_horizon = 1, squeeze the last dimension.
        if self.forecast_horizon == 1:
            output = output.squeeze(-1)  # shape: (batch_size, num_nodes)
        
        return output

# For testing the model architecture (optional)
if __name__ == "__main__":
    import torch_geometric
    # Dummy graph data loading
    from torch_geometric.data import Data
    num_nodes = 50
    dummy_node_features = torch.ones((num_nodes, 16))
    dummy_edge_index = torch.tensor([[0, 1, 2, 3],
                                     [1, 2, 3, 4]], dtype=torch.long)
    dummy_graph = Data(x=dummy_node_features, edge_index=dummy_edge_index)
    
    # Dummy time series input: batch_size=8, num_nodes=50, input_length=20
    dummy_x = torch.rand((8, num_nodes, 20))
    model = GraphAugmentedForecastingModel(num_nodes=num_nodes, input_length=20)
    pred = model(dummy_x, dummy_graph)
    print("Model output shape:", pred.shape)
