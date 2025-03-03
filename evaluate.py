import torch
import matplotlib.pyplot as plt
import numpy as np
from model import GraphAugmentedForecastingModel
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve

# Load data
test_X, test_Y = torch.load("processed/test.pt", weights_only=False)
graph_data = torch.load("processed/graph.pt", weights_only=False)

num_nodes = graph_data.x.shape[0]
input_length = test_X.shape[-1]
forecast_horizon = test_Y.shape[-1]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = GraphAugmentedForecastingModel(num_nodes, input_length,
                                       lstm_hidden_dim=64, gcn_hidden_dim=32,
                                       forecast_horizon=forecast_horizon,
                                       lstm_layers=2, gcn_layers=3).to(device)

model.load_state_dict(torch.load("processed/best_model.pt", map_location=device, weights_only=False))
model.eval()

# Predict
with torch.no_grad():
    logits = model(test_X.to(device), graph_data.to(device))
    probabilities = torch.sigmoid(logits).cpu()

# Flatten for global metrics
y_true = test_Y.numpy().flatten()
y_prob = probabilities.numpy().flatten()

# Precision-Recall Curve (Global)
precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

plt.figure(figsize=(8, 6))
plt.plot(thresholds, precision[:-1], label="Precision")
plt.plot(thresholds, recall[:-1], label="Recall")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.legend()
plt.grid(True)
plt.title("Global Precision-Recall Tradeoff")
plt.show()

# Set global threshold
optimal_threshold = 0.5  # You can tune this based on the curve

# Apply threshold globally
pred_binary = (probabilities > optimal_threshold).float()

# ---- Per-Node Diagnostics ----
per_node_metrics = []
node_degrees = []

# Compute degree per node from the graph
degree_dict = {i: 0 for i in range(num_nodes)}
for edge in graph_data.edge_index.T.cpu().numpy():
    degree_dict[edge[0]] += 1
    degree_dict[edge[1]] += 1
node_degrees = np.array([degree_dict[i] for i in range(num_nodes)])

# Per-node metrics
for node in range(num_nodes):
    y_true_node = test_Y[:, node, :].numpy().flatten()
    y_pred_node = pred_binary[:, node, :].numpy().flatten()

    precision = precision_score(y_true_node, y_pred_node, zero_division=0)
    recall = recall_score(y_true_node, y_pred_node, zero_division=0)
    f1 = f1_score(y_true_node, y_pred_node, zero_division=0)

    per_node_metrics.append((node, precision, recall, f1))

# Convert to numpy for easier analysis
per_node_metrics = np.array(per_node_metrics)

# ---- Summary Table ----
print("\n===== Per-Node Diagnostics =====")
print(f"{'Node':<6}{'Degree':<8}{'Precision':<10}{'Recall':<10}{'F1':<10}")
for node, precision, recall, f1 in per_node_metrics:
    degree = node_degrees[int(node)]
    print(f"{int(node):<6}{degree:<8}{precision:<10.4f}{recall:<10.4f}{f1:<10.4f}")
print("================================")

# ---- Global Metrics ----
y_pred = pred_binary.numpy().flatten()
global_precision = precision_score(y_true, y_pred, zero_division=0)
global_recall = recall_score(y_true, y_pred, zero_division=0)
global_f1 = f1_score(y_true, y_pred, zero_division=0)

print("\n==== Final Global Evaluation Metrics ====")
print(f"Threshold: {optimal_threshold:.2f}")
print(f"Precision: {global_precision:.4f}")
print(f"Recall:    {global_recall:.4f}")
print(f"F1 Score:  {global_f1:.4f}")
print("=========================================")

# ---- Visualization for Each Node (Optional - Auto-Generated) ----
for node in range(num_nodes):
    actual = test_Y[:, node, :].numpy().flatten()
    prob = probabilities[:, node, :].numpy().flatten()
    binary = pred_binary[:, node, :].numpy().flatten()

    plt.figure(figsize=(12, 5))
    plt.plot(actual, label=f"Actual (Node {node})", color='tab:blue')
    plt.plot(prob, label=f"Forecast Probability (Node {node})", linestyle="--", color='tab:orange')
    plt.plot(binary, label=f"Forecast Binary (>{optimal_threshold})", linestyle=":", color='tab:green')
    plt.ylim([-0.1, 1.1])
    plt.xlabel("Test Sample Index")
    plt.ylabel("Event Value")
    plt.title(f"Forecast vs Actual for Node {node} (Degree={node_degrees[node]})")
    plt.legend()
    plt.grid(True)
    plt.savefig("forecast_vs_actual.png", dpi=300)
    plt.show()


# plt.show()
