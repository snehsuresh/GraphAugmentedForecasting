import torch
import numpy as np
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from model import NodeLSTMModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_nodes = 10
best_f1s = []
best_thresholds = []

for node_id in range(num_nodes):
    test_X, test_Y = torch.load(f"processed/test_node_{node_id}.pt")
    test_X, test_Y = test_X.to(device), test_Y.to(device)

    model = NodeLSTMModel(test_X.shape[1]).to(device)
    model.load_state_dict(torch.load(f"processed/best_model_node_{node_id}.pt"))
    model.eval()

    with torch.no_grad():
        logits = model(test_X)
        probabilities = torch.sigmoid(logits).cpu().numpy()

    y_true = test_Y.cpu().numpy().flatten()
    y_prob = probabilities.flatten()

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    best_f1, best_threshold = 0, 0
    for p, r, t in zip(precision, recall, thresholds):
        if p + r > 0:
            f1 = 2 * p * r / (p + r)
            if f1 > best_f1:
                best_f1, best_threshold = f1, t

    best_f1s.append(best_f1)
    best_thresholds.append(best_threshold)

x = np.arange(num_nodes)
bar_width = 0.4

plt.figure(figsize=(12, 6))

plt.bar(x - bar_width/2, best_f1s, width=bar_width, label="Best F1")
plt.bar(x + bar_width/2, best_thresholds, width=bar_width, label="Best Threshold")

plt.xlabel("Node ID")
plt.ylabel("Value")
plt.title("Per-Node Best F1 & Threshold")
plt.xticks(x)
plt.legend()
plt.grid(True)

plt.savefig("threshold_tuning_dashboard.png")
plt.show()

print("Saved plot to threshold_tuning_dashboard.png")
