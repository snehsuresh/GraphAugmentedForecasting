import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data

# Settings
data_dir = "data/yahoo"
file_list = sorted(os.listdir(data_dir))[:10]

# Load & merge Yahoo S5 data
time_series = []
for file in file_list:
    df = pd.read_csv(os.path.join(data_dir, file))
    series = df['is_anomaly'].astype(int).values
    time_series.append(series)

# Ensure aligned length across nodes
min_len = min(len(s) for s in time_series)
time_series = [s[:min_len] for s in time_series]
data_matrix = np.stack(time_series, axis=1)

# Sliding window processing
window_size = 20
forecast_horizon = 5

X_samples, Y_samples = [], []
for t in range(len(data_matrix) - window_size - forecast_horizon + 1):
    X_samples.append(data_matrix[t:t+window_size].T)
    Y_samples.append(data_matrix[t+window_size:t+window_size+forecast_horizon].T)

X_samples, Y_samples = np.array(X_samples), np.array(Y_samples)

# Train/Val/Test Split
train_end = int(len(X_samples) * 0.7)
val_end = int(len(X_samples) * 0.15) + train_end

torch.save((torch.tensor(X_samples[:train_end]), torch.tensor(Y_samples[:train_end])), "processed/train.pt")
torch.save((torch.tensor(X_samples[train_end:val_end]), torch.tensor(Y_samples[train_end:val_end])), "processed/val.pt")
torch.save((torch.tensor(X_samples[val_end:]), torch.tensor(Y_samples[val_end:])), "processed/test.pt")

# Co-occurrence graph (train set only)
train_Y_binary = (Y_samples[:train_end].sum(axis=2) > 0).astype(int)
co_occurrence = np.zeros((len(file_list), len(file_list)))

for sample in train_Y_binary:
    for i in range(len(file_list)):
        for j in range(len(file_list)):
            if sample[i] and sample[j]:
                co_occurrence[i, j] += 1

co_occurrence /= train_Y_binary.shape[0]
co_occurrence[range(len(file_list)), range(len(file_list))] = 0  # no self-edges

# Graph edges (force minimum connectivity)
edges = []
for i in range(len(file_list)):
    if co_occurrence[i].sum() == 0:  # isolated node fix
        nearest = np.argsort(-co_occurrence[:, i])[:2]
        for j in nearest:
            edges.append([i, j])
            edges.append([j, i])
    else:
        for j in range(len(file_list)):
            if co_occurrence[i, j] > 0:
                edges.append([i, j])

edge_index = torch.tensor(edges, dtype=torch.long).T

# Node features (historical event rate + degree)
event_rate = train_Y_binary.sum(axis=0) / train_Y_binary.shape[0]
degree = np.zeros(len(file_list))
for edge in edges:
    degree[edge[0]] += 1

node_features = torch.tensor(np.stack([event_rate, degree], axis=1), dtype=torch.float32)
graph_data = Data(x=node_features, edge_index=edge_index)

torch.save(graph_data, "processed/graph.pt")
print("Preprocessing complete with graph fixes and node features.")
