import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
import os

# Parameters
window_size = 20
forecast_horizon = 5  # Expanded to capture patterns across time
train_ratio = 0.7
val_ratio = 0.15

# Load data
df = pd.read_csv("data/time_series.csv")
ts_data = df.drop(columns=["time"]).values

num_timesteps, num_nodes = ts_data.shape

# Build co-occurrence matrix
co_occurrence = np.zeros((num_nodes, num_nodes), dtype=np.float32)

for t in range(1, num_timesteps):
    active_nodes = np.where(ts_data[t] == 1)[0]
    for i in active_nodes:
        for j in active_nodes:
            if i != j:
                co_occurrence[i, j] += 1

# Normalize into probabilities
co_occurrence = co_occurrence / (np.sum(co_occurrence, axis=1, keepdims=True) + 1e-6)

# Sliding window samples
X_samples, Y_samples = [], []
for t in range(num_timesteps - window_size - forecast_horizon + 1):
    X_samples.append(ts_data[t:t+window_size].T)
    Y_samples.append(ts_data[t+window_size:t+window_size+forecast_horizon].T)

X_samples = np.array(X_samples)
Y_samples = np.array(Y_samples)

# Train/val/test split
train_end = int(len(X_samples) * train_ratio)
val_end = int(len(X_samples) * (train_ratio + val_ratio))

X_train, Y_train = X_samples[:train_end], Y_samples[:train_end]
X_val, Y_val = X_samples[train_end:val_end], Y_samples[train_end:val_end]
X_test, Y_test = X_samples[val_end:], Y_samples[val_end:]

# Convert to tensors
def to_tensor(arr):
    return torch.tensor(arr, dtype=torch.float32)

torch.save((to_tensor(X_train), to_tensor(Y_train)), "processed/train.pt")
torch.save((to_tensor(X_val), to_tensor(Y_val)), "processed/val.pt")
torch.save((to_tensor(X_test), to_tensor(Y_test)), "processed/test.pt")

# Graph Data - Co-occurrence Graph
edge_index = torch.tensor(np.stack(np.where(co_occurrence > 0)), dtype=torch.long)
node_features = torch.tensor(co_occurrence.sum(axis=1, keepdims=True), dtype=torch.float32)

graph_data = Data(x=node_features, edge_index=edge_index)
torch.save(graph_data, "processed/graph.pt")

print("Preprocessing complete. Data and graph saved.")
