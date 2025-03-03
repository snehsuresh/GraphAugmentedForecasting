import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data

data_dir = "data/yahoo"
file_list = sorted(os.listdir(data_dir))[:10]

time_series = []
for file in file_list:
    df = pd.read_csv(os.path.join(data_dir, file))
    series = df['is_anomaly'].astype(int).values
    time_series.append(series)

min_len = min(len(s) for s in time_series)
time_series = [s[:min_len] for s in time_series]
data_matrix = np.stack(time_series, axis=1)

window_size = 20
forecast_horizon = 5

X, Y = [], []
for t in range(len(data_matrix) - window_size - forecast_horizon + 1):
    X.append(data_matrix[t:t+window_size].T)
    Y.append(data_matrix[t+window_size:t+window_size+forecast_horizon].T)

X, Y = np.array(X), np.array(Y)

train_end = int(len(X) * 0.7)
val_end = int(len(X) * 0.85)

torch.save((torch.tensor(X[:train_end]), torch.tensor(Y[:train_end])), "processed/train.pt")
torch.save((torch.tensor(X[train_end:val_end]), torch.tensor(Y[train_end:val_end])), "processed/val.pt")
torch.save((torch.tensor(X[val_end:]), torch.tensor(Y[val_end:])), "processed/test.pt")

train_Y_binary = (Y[:train_end].sum(axis=2) > 0).astype(int)
co_occurrence = np.zeros((len(file_list), len(file_list)))

for sample in train_Y_binary:
    for i in range(len(file_list)):
        for j in range(len(file_list)):
            if sample[i] and sample[j]:
                co_occurrence[i, j] += 1

edges = []
for i in range(len(file_list)):
    if co_occurrence[i].sum() == 0:
        nearest = np.argsort(-co_occurrence[:, i])[:2]
        for j in nearest:
            edges.append([i, j])
            edges.append([j, i])
    else:
        for j in range(len(file_list)):
            if co_occurrence[i, j] > 0:
                edges.append([i, j])

edge_index = torch.tensor(edges, dtype=torch.long).T
event_rate = train_Y_binary.sum(axis=0) / train_Y_binary.shape[0]
degree = np.zeros(len(file_list))
for edge in edges:
    degree[edge[0]] += 1

node_features = torch.tensor(np.stack([event_rate, degree], axis=1), dtype=torch.float32)
graph_data = Data(x=node_features, edge_index=edge_index)

torch.save(graph_data, "processed/graph.pt")
print("Preprocessing complete.")
