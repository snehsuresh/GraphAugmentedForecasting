import os
import torch
import matplotlib.pyplot as plt
import numpy as np

num_nodes = 10
train_counts = []
test_counts = []

for node_id in range(num_nodes):
    train_X, train_Y = torch.load(f"processed/train_node_{node_id}.pt",  weights_only=False)
    test_X, test_Y = torch.load(f"processed/test_node_{node_id}.pt", weights_only=False)

    train_counts.append(train_Y.sum().item())
    test_counts.append(test_Y.sum().item())

plt.figure(figsize=(12, 6))
x = np.arange(num_nodes)

bar_width = 0.4
plt.bar(x - bar_width/2, train_counts, width=bar_width, label="Train Anomalies")
plt.bar(x + bar_width/2, test_counts, width=bar_width, label="Test Anomalies")

plt.xlabel("Node ID")
plt.ylabel("Anomaly Count")
plt.title("Per-Node Anomaly Count (Train vs Test)")
plt.xticks(x)
plt.legend()
plt.grid(True)

plt.savefig("diagnostic_anomaly_counts.png")
plt.show()

print("Saved plot to diagnostic_anomaly_counts.png")
