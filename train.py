import torch
import torch.nn as nn
import torch.optim as optim
from model import GraphAugmentedForecastingModel
from torch_geometric.data import Data
import os

# Load preprocessed data
train_X, train_Y = torch.load("processed/train.pt", weights_only=False)
val_X, val_Y = torch.load("processed/val.pt", weights_only=False)

# Load graph data
graph_data = torch.load("processed/graph.pt", weights_only=False)

num_nodes = graph_data.num_nodes if hasattr(graph_data, 'num_nodes') else graph_data.x.shape[0]
input_length = train_X.shape[-1]

# Hyperparameters
lstm_hidden_dim = 64
gcn_hidden_dim = 32
combined_dim = 64  # not used explicitly here since fc layer takes the concatenated vector
forecast_horizon = 1
learning_rate = 0.001
num_epochs = 20  # Increase if needed

# Instantiate the model and move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GraphAugmentedForecastingModel(num_nodes=num_nodes, input_length=input_length,
                                        lstm_hidden_dim=lstm_hidden_dim,
                                        gcn_hidden_dim=gcn_hidden_dim,
                                        combined_dim=combined_dim,
                                        forecast_horizon=forecast_horizon)
model = model.to(device)
graph_data = graph_data.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
best_val_loss = float("inf")
for epoch in range(1, num_epochs + 1):
    model.train()
    optimizer.zero_grad()
    # Move batch to device
    batch_X = train_X.to(device)
    batch_Y = train_Y.to(device)
    
    outputs = model(batch_X, graph_data)
    loss = criterion(outputs, batch_Y)
    loss.backward()
    optimizer.step()
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(val_X.to(device), graph_data)
        val_loss = criterion(val_outputs, val_Y.to(device))
    
    print(f"Epoch {epoch}: Train Loss = {loss.item():.6f}, Val Loss = {val_loss.item():.6f}")
    
    # Save best model
    if val_loss.item() < best_val_loss:
        best_val_loss = val_loss.item()
        torch.save(model.state_dict(), "processed/best_model.pt")
        print("Best model updated.")

print("Training complete. Best model saved to processed/best_model.pt")
