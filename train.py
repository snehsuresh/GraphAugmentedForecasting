import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import GraphAugmentedForecastingModel

train_X, train_Y = torch.load("processed/train.pt", weights_only=False)
val_X, val_Y = torch.load("processed/val.pt", weights_only=False)
graph_data = torch.load("processed/graph.pt", weights_only=False)

num_nodes = graph_data.x.shape[0]
input_length = train_X.shape[-1]
forecast_horizon = train_Y.shape[-1]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model: Deeper LSTM & GCN
model = GraphAugmentedForecastingModel(num_nodes, input_length,
                                       lstm_hidden_dim=64, gcn_hidden_dim=32,
                                       forecast_horizon=forecast_horizon,
                                       lstm_layers=2, gcn_layers=3).to(device)

graph_data = graph_data.to(device)

# Balanced BCE Loss (no oversampling anymore)
total_events = train_Y.sum().item()
total_no_events = train_Y.numel() - total_events
pos_weight = torch.tensor([total_no_events / (total_events + 1e-6)], device=device)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=0.001)

best_val_loss = float('inf')

for epoch in range(20):
    model.train()
    optimizer.zero_grad()

    logits = model(train_X.to(device), graph_data)
    loss = criterion(logits, train_Y.to(device))

    loss.backward()
    optimizer.step()

    with torch.no_grad():
        val_logits = model(val_X.to(device), graph_data)
        val_loss = criterion(val_logits, val_Y.to(device))

    print(f"Epoch {epoch+1}: Train Loss = {loss.item():.4f}, Val Loss = {val_loss.item():.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "processed/best_model.pt")
        print("Best model updated.")

print("Training complete.")
