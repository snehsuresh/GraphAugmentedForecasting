import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data

# --- SETTINGS ---
# Folder where you have placed the Yahoo S5 CSV files.
# Each CSV should contain at least these columns: "timestamp", "value", "is_anomaly"
data_dir = "data/yahoo"
# Use the first 10 CSV files (adjust as needed)
file_list = sorted(os.listdir(data_dir))[:10]

# --- LOAD & MERGE TIME SERIES ---
# We assume all files have the same number of rows and aligned timestamps.
time_series_list = []
for file in file_list:
    df = pd.read_csv(os.path.join(data_dir, file))
    # Ensure anomaly column is binary (0/1)
    ts = df['is_anomaly'].astype(int).values
    time_series_list.append(ts)

# Truncate to the minimum length (in case they differ)
min_len = min(len(ts) for ts in time_series_list)
time_series_list = [ts[:min_len] for ts in time_series_list]

# Create a matrix of shape (min_len, num_nodes) where each column is a node’s binary event series.
data_matrix = np.stack(time_series_list, axis=1)
num_timesteps, num_nodes = data_matrix.shape
print(f"Loaded data with {num_timesteps} timesteps and {num_nodes} nodes.")

# --- SLIDING WINDOW SPLITTING ---
# We forecast over a horizon to capture temporal patterns.
window_size = 20
forecast_horizon = 5

X_samples = []
Y_samples = []
for t in range(num_timesteps - window_size - forecast_horizon + 1):
    # Input window: shape (num_nodes, window_size)
    X_samples.append(data_matrix[t:t+window_size, :].T)
    # Forecast window: shape (num_nodes, forecast_horizon)
    Y_samples.append(data_matrix[t+window_size:t+window_size+forecast_horizon, :].T)

X_samples = np.array(X_samples)
Y_samples = np.array(Y_samples)
print(f"Created {X_samples.shape[0]} samples.")

# --- SPLIT INTO TRAIN/VAL/TEST ---
num_samples = X_samples.shape[0]
train_end = int(num_samples * 0.7)
val_end = int(num_samples * 0.85)

X_train = torch.tensor(X_samples[:train_end], dtype=torch.float32)
Y_train = torch.tensor(Y_samples[:train_end], dtype=torch.float32)
X_val = torch.tensor(X_samples[train_end:val_end], dtype=torch.float32)
Y_val = torch.tensor(Y_samples[train_end:val_end], dtype=torch.float32)
X_test = torch.tensor(X_samples[val_end:], dtype=torch.float32)
Y_test = torch.tensor(Y_samples[val_end:], dtype=torch.float32)

os.makedirs("processed", exist_ok=True)
torch.save((X_train, Y_train), "processed/train.pt")
torch.save((X_val, Y_val), "processed/val.pt")
torch.save((X_test, Y_test), "processed/test.pt")
print(f"Saved train ({X_train.shape[0]} samples), val ({X_val.shape[0]}), test ({X_test.shape[0]}) datasets.")

# --- BUILD THE CO-OCCURRENCE GRAPH ---
# Here we compute, over the training samples, how often pairs of nodes are anomalous together.
# For each sample, if any anomaly occurs in the forecast window for a node, we mark it as an event.
train_binary = (Y_train.sum(dim=2) > 0).numpy()  # shape: (num_samples, num_nodes)
co_occurrence = np.zeros((num_nodes, num_nodes))
for sample in train_binary:
    for i in range(num_nodes):
        for j in range(num_nodes):
            if sample[i] == 1 and sample[j] == 1:
                co_occurrence[i, j] += 1
co_occurrence /= train_binary.shape[0]

# Build edge list: Only add an edge if co-occurrence > 0.1 (adjust threshold as needed)
edges = []
for i in range(num_nodes):
    for j in range(num_nodes):
        if i != j and co_occurrence[i, j] > 0.1:
            edges.append([i, j])
edges = np.array(edges).T  # shape: (2, num_edges)
edge_index = torch.tensor(edges, dtype=torch.long)

# Node features: Use each node's anomaly frequency in training.
node_freq = Y_train.sum(dim=(0,2)).unsqueeze(1) / Y_train.shape[0]
node_features = node_freq  # shape: (num_nodes, 1)

graph_data = Data(x=node_features, edge_index=edge_index)
torch.save(graph_data, "processed/graph.pt")
print("Graph data saved to processed/graph.pt")
