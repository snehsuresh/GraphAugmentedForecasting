import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from model import GraphAugmentedForecastingModel
import os

# Load preprocessed data
train_X, train_Y = torch.load("processed/train.pt", weights_only=False)
val_X, val_Y = torch.load("processed/val.pt", weights_only=False)

# Oversampling: Keep more realistic balance (80% no-event, 20% event)
def balanced_oversample_events(X, Y, target_event_ratio=0.2):
    has_event = Y.sum(dim=1) > 0
    X_with_events = X[has_event]
    Y_with_events = Y[has_event]
    X_without_events = X[~has_event]
    Y_without_events = Y[~has_event]

    num_event_samples = X_with_events.shape[0]
    num_no_event_samples = int(num_event_samples * (1 - target_event_ratio) / target_event_ratio)

    # If not enough no-event windows exist, just take all of them.
    if len(X_without_events) < num_no_event_samples:
        num_no_event_samples = len(X_without_events)

    if num_no_event_samples > 0:
        idx = np.random.choice(len(X_without_events), num_no_event_samples, replace=False)
        X_without_events = X_without_events[idx]
        Y_without_events = Y_without_events[idx]

    X_balanced = torch.cat([X_with_events, X_without_events], dim=0)
    Y_balanced = torch.cat([Y_with_events, Y_without_events], dim=0)

    indices = torch.randperm(X_balanced.shape[0])
    return X_balanced[indices], Y_balanced[indices]


train_X, train_Y = balanced_oversample_events(train_X, train_Y)

# Load graph data
graph_data = torch.load("processed/graph.pt", weights_only=False)
num_nodes = graph_data.num_nodes if hasattr(graph_data, 'num_nodes') else graph_data.x.shape[0]
input_length = train_X.shape[-1]

# Hyperparameters
lstm_hidden_dim = 64
gcn_hidden_dim = 32
forecast_horizon = 1
learning_rate = 0.001
num_epochs = 20

# Instantiate model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GraphAugmentedForecastingModel(num_nodes=num_nodes,
                                       input_length=input_length,
                                       lstm_hidden_dim=lstm_hidden_dim,
                                       gcn_hidden_dim=gcn_hidden_dim,
                                       forecast_horizon=forecast_horizon)
model.to(device)
graph_data = graph_data.to(device)

# Focal Loss Definition (moderate pos_weight now)
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, pos_weight=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        pt = targets * probs + (1 - targets) * (1 - probs)
        focal_weight = (self.alpha * targets + (1 - self.alpha) * (1 - targets)) * (1 - pt).pow(self.gamma)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none', pos_weight=self.pos_weight)
        return (focal_weight * bce).mean()

# Adjusted pos weight — not extreme anymore
pos_weight = torch.tensor([50.0], device=device)

criterion = FocalLoss(alpha=0.25, gamma=2, pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

best_val_loss = float("inf")

for epoch in range(1, num_epochs + 1):
    model.train()
    optimizer.zero_grad()

    batch_X = train_X.to(device)
    batch_Y = train_Y.to(device)

    logits = model(batch_X, graph_data)
    loss = criterion(logits, batch_Y)

    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_logits = model(val_X.to(device), graph_data)
        val_loss = criterion(val_logits, val_Y.to(device))

    print(f"Epoch {epoch}: Train Loss = {loss.item():.6f}, Val Loss = {val_loss.item():.6f}")

    if val_loss.item() < best_val_loss:
        best_val_loss = val_loss.item()
        torch.save(model.state_dict(), "processed/best_model.pt")
        print("Best model updated.")

print("Training complete. Best model saved to processed/best_model.pt")
