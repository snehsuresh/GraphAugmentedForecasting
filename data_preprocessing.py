import os
import pandas as pd
import numpy as np
import torch

data_dir = "data/yahoo"
file_list = sorted(os.listdir(data_dir))[:10]

window_size = 20
forecast_horizon = 5

for node_id, file in enumerate(file_list):
    df = pd.read_csv(os.path.join(data_dir, file))
    series = df['is_anomaly'].astype(int).values

    X, Y = [], []
    for t in range(len(series) - window_size - forecast_horizon + 1):
        X.append(series[t:t+window_size])
        Y.append(series[t+window_size:t+window_size+forecast_horizon])

    X, Y = np.array(X), np.array(Y)

    train_end = int(len(X) * 0.7)
    val_end = int(len(X) * 0.85)

    os.makedirs("processed", exist_ok=True)

    torch.save((torch.tensor(X[:train_end], dtype=torch.float32), 
                torch.tensor(Y[:train_end], dtype=torch.float32)),
               f"processed/train_node_{node_id}.pt")

    torch.save((torch.tensor(X[train_end:val_end], dtype=torch.float32), 
                torch.tensor(Y[train_end:val_end], dtype=torch.float32)),
               f"processed/val_node_{node_id}.pt")

    torch.save((torch.tensor(X[val_end:], dtype=torch.float32), 
                torch.tensor(Y[val_end:], dtype=torch.float32)),
               f"processed/test_node_{node_id}.pt")

print("Per-node preprocessing complete.")
