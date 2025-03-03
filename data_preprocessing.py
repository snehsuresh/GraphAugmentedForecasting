import numpy as np
import pandas as pd
import torch
import os
from torch_geometric.data import Data

# Parameters
window_size = 20  # length of input sequence
forecast_horizon = 1  # forecast next time step
train_ratio = 0.7
val_ratio = 0.15
# test_ratio = remaining portion

# Create processed data directory
os.makedirs("processed", exist_ok=True)

# Load time series data
df = pd.read_csv("data/time_series.csv")
# Remove the time column (we only need the values)
ts_data = df.drop(columns=["time"]).values  # shape: (num_timesteps, num_nodes)
num_timesteps, num_nodes = ts_data.shape

# Prepare sliding-window samples for all nodes simultaneously.
# Each sample is an array of shape (num_nodes, window_size) and target is (num_nodes,)
X_samples = []
Y_samples = []

# Create samples such that the input covers t:t+window_size and the target is at t+window_size
for t in range(num_timesteps - window_size - forecast_horizon + 1):
    X_samples.append(ts_data[t: t + window_size, :].T)  # shape (num_nodes, window_size)
    Y_samples.append(ts_data[t + window_size: t + window_size + forecast_horizon, :].T.squeeze())

X_samples = np.array(X_samples)  # shape: (num_samples, num_nodes, window_size)
Y_samples = np.array(Y_samples)  # shape: (num_samples, num_nodes)

num_samples = X_samples.shape[0]
train_end = int(num_samples * train_ratio)
val_end = int(num_samples * (train_ratio + val_ratio))

X_train = torch.tensor(X_samples[:train_end], dtype=torch.float32)
Y_train = torch.tensor(Y_samples[:train_end], dtype=torch.float32)
X_val = torch.tensor(X_samples[train_end:val_end], dtype=torch.float32)
Y_val = torch.tensor(Y_samples[train_end:val_end], dtype=torch.float32)
X_test = torch.tensor(X_samples[val_end:], dtype=torch.float32)
Y_test = torch.tensor(Y_samples[val_end:], dtype=torch.float32)

torch.save((X_train, Y_train), "processed/train.pt")
torch.save((X_val, Y_val), "processed/val.pt")
torch.save((X_test, Y_test), "processed/test.pt")

print(f"Preprocessing complete. Samples: Train={X_train.shape[0]}, Val={X_val.shape[0]}, Test={X_test.shape[0]}")

# Now build the PyTorch Geometric graph data object from the edge list
edge_df = pd.read_csv("data/graph_edges.csv")
# Convert to torch tensor and transpose to shape [2, num_edges]
edge_index = torch.tensor(edge_df.values.T, dtype=torch.long)

# For node features, we can simply use an identity matrix or ones.
# Here, we use a learnable embedding so the initial features can be ones.
node_features = torch.ones((num_nodes, 16))  # 16-dimensional initial features

graph_data = Data(x=node_features, edge_index=edge_index)
torch.save(graph_data, "processed/graph.pt")
print("Graph data saved to processed/graph.pt")
