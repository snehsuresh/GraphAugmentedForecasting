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
        
        # Graph Convolution Network
        self.gcn1 = GCNConv(in_channels=16, out_channels=gcn_hidden_dim)
        self.gcn2 = GCNConv(in_channels=gcn_hidden_dim, out_channels=gcn_hidden_dim)
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(input_size=1, hidden_size=lstm_hidden_dim, batch_first=True)
        
        # Fully connected layer: combines LSTM output + GCN embedding
        self.fc = nn.Linear(lstm_hidden_dim + gcn_hidden_dim, forecast_horizon)
        
    def forward(self, x, graph_data):
        """
        x: (batch_size, num_nodes, input_length)
        graph_data: PyTorch Geometric data object with x (node feats) & edge_index
        """
        batch_size = x.size(0)
        
        # --- Graph Part ---
        node_features = graph_data.x  # shape: (num_nodes, 16)
        edge_index = graph_data.edge_index
        
        gcn_out = F.relu(self.gcn1(node_features, edge_index))
        node_emb = F.relu(self.gcn2(gcn_out, edge_index))  # (num_nodes, gcn_hidden_dim)
        
        # Expand node embeddings for each sample in the batch
        node_emb_expanded = node_emb.unsqueeze(0).expand(batch_size, -1, -1)  # (B, num_nodes, gcn_hidden_dim)
        
        # --- LSTM Part ---
        # x is (B, num_nodes, input_length) => we process each node's sequence with LSTM
        x_lstm = x.unsqueeze(-1)  # (B, num_nodes, input_length, 1)
        
        lstm_out_list = []
        for node in range(self.num_nodes):
            # node_series: (B, input_length, 1)
            node_series = x_lstm[:, node, :, :]
            _, (h_n, _) = self.lstm(node_series)
            # h_n[-1] = final hidden state for last LSTM layer => (B, lstm_hidden_dim)
            lstm_out_list.append(h_n[-1])
        
        # Stack => (B, num_nodes, lstm_hidden_dim)
        lstm_out = torch.stack(lstm_out_list, dim=1)
        
        # --- Combine ---
        combined = torch.cat([lstm_out, node_emb_expanded], dim=-1)  # (B, num_nodes, lstm_hidden_dim+gcn_hidden_dim)
        logits = self.fc(combined)  # (B, num_nodes, forecast_horizon)
        
        # For horizon=1, we can squeeze the last dimension => (B, num_nodes)
        if self.forecast_horizon == 1:
            logits = logits.squeeze(-1)
        
        return logits

# Optional test
if __name__ == "__main__":
    import torch
    from torch_geometric.data import Data
    num_nodes = 50
    dummy_node_features = torch.ones((num_nodes, 16))
    dummy_edge_index = torch.tensor([[0, 1, 2, 3],
                                     [1, 2, 3, 4]], dtype=torch.long)
    dummy_graph = Data(x=dummy_node_features, edge_index=dummy_edge_index)
    
    dummy_x = torch.rand((8, num_nodes, 20))
    model = GraphAugmentedForecastingModel(num_nodes=num_nodes, input_length=20)
    out = model(dummy_x, dummy_graph)
    print("Output shape:", out.shape)
