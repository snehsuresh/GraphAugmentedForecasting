import torch
import matplotlib.pyplot as plt
import numpy as np
from model import GraphAugmentedForecastingModel
from torch_geometric.data import Data
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve

# Load test data and graph
test_X, test_Y = torch.load("processed/test.pt", weights_only=False)
graph_data = torch.load("processed/graph.pt", weights_only=False)
num_nodes = graph_data.x.shape[0]
input_length = test_X.shape[-1]

lstm_hidden_dim = 64
gcn_hidden_dim = 32
forecast_horizon = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GraphAugmentedForecastingModel(num_nodes=num_nodes,
                                       input_length=input_length,
                                       lstm_hidden_dim=lstm_hidden_dim,
                                       gcn_hidden_dim=gcn_hidden_dim,
                                       forecast_horizon=forecast_horizon)
model.load_state_dict(torch.load("processed/best_model.pt", map_location=device, weights_only=False))
model.to(device)
model.eval()

with torch.no_grad():
    logits = model(test_X.to(device), graph_data.to(device))
    probabilities = torch.sigmoid(logits).cpu()

y_true = test_Y.numpy().flatten()
y_prob = probabilities.numpy().flatten()

# Precision-Recall Curve
precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

plt.figure(figsize=(8, 6))
plt.plot(thresholds, precision[:-1], label="Precision")
plt.plot(thresholds, recall[:-1], label="Recall")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.legend()
plt.grid(True)
plt.title("Precision-Recall Tradeoff")
plt.show()

# Optimal Threshold — set this after reviewing the PR curve
threshold = 0.7

pred_binary = (probabilities > threshold).float()

# Streak Filter (optional — keeps only longer sequences of events)
def apply_streak_filter(binary, min_streak=2):
    filtered = binary.clone()
    for batch in range(binary.shape[0]):
        for node in range(binary.shape[1]):
            streak = 0
            for t in range(binary.shape[2] if binary.ndim == 3 else binary.shape[0]):
                if binary[batch, node, t] if binary.ndim == 3 else binary[batch, node]:
                    streak += 1
                else:
                    if streak < min_streak:
                        filtered[batch, node, t-streak:t] = 0
                    streak = 0
    return filtered

# Enable if desired
# pred_binary = apply_streak_filter(pred_binary.unsqueeze(0)).squeeze(0)

y_pred = pred_binary.numpy().flatten()

precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

print("==== Final Evaluation Metrics ====")
print(f"Threshold: {threshold:.2f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print("==================================")

# Plot for Node 0
sample_node = 0
num_samples_to_plot = min(50, test_X.shape[0])

actual = test_Y[:num_samples_to_plot, sample_node].numpy()
prob = probabilities[:num_samples_to_plot, sample_node].numpy()
binary = pred_binary[:num_samples_to_plot, sample_node].numpy()

plt.figure(figsize=(10, 5))
plt.plot(actual, label="Actual (0/1)", color='tab:blue')
plt.plot(prob, label="Forecast Probability", linestyle="--", color='tab:orange')
plt.plot(binary, label=f"Forecast Binary (>{threshold})", linestyle=":", color='tab:green')
plt.ylim([-0.1, 1.1])
plt.xlabel("Test Sample Index")
plt.ylabel("Event Value")
plt.title(f"Forecast vs Actual for Node {sample_node}")
plt.legend()
plt.grid(True)
plt.show()

plt.savefig("forecast_vs_actual.png", dpi=300)
# plt.show()
