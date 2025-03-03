import torch
import torch.nn as nn
import torch.optim as optim
from model import GraphAugmentedForecastingModel

# Load processed data
train_X, train_Y = torch.load("processed/train.pt")
val_X, val_Y = torch.load("processed/val.pt")
graph_data = torch.load("processed/graph.pt")

num_nodes = graph_data.x.shape[0]
input_length = train_X.shape[-1]
forecast_horizon = train_Y.shape[-1]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate the model with deeper LSTM & GCN (as defined in model.py)
model = GraphAugmentedForecastingModel(num_nodes, input_length,
                                       lstm_hidden_dim=64, gcn_hidden_dim=32,
                                       forecast_horizon=forecast_horizon,
                                       lstm_layers=2, gcn_layers=3).to(device)
graph_data = graph_data.to(device)

# Compute dynamic pos_weight based on training data distribution
total_events = train_Y.sum().item()
total_no_events = train_Y.numel() - total_events
pos_weight = torch.tensor([total_no_events / (total_events + 1e-6)], device=device)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=0.001)

best_val_loss = float('inf')
num_epochs = 20

for epoch in range(1, num_epochs+1):
    model.train()
    optimizer.zero_grad()
    
    logits = model(train_X.to(device), graph_data)
    loss = criterion(logits, train_Y.to(device))
    
    loss.backward()
    optimizer.step()
    
    model.eval()
    with torch.no_grad():
        val_logits = model(val_X.to(device), graph_data)
        val_loss = criterion(val_logits, val_Y.to(device))
    
    print(f"Epoch {epoch}: Train Loss = {loss.item():.4f}, Val Loss = {val_loss.item():.4f}")
    
    if val_loss.item() < best_val_loss:
        best_val_loss = val_loss.item()
        torch.save(model.state_dict(), "processed/best_model.pt")
        print("Best model updated.")
        
print("Training complete. Best model saved to processed/best_model.pt")
