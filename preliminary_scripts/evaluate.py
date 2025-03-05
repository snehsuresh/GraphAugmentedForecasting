import torch
import numpy as np
import pandas as pd
import os
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve
import matplotlib.pyplot as plt
from model import NodeLSTMModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create directory to save per-node predictions and probabilities
os.makedirs("results/per_node_predictions", exist_ok=True)

for node_id in range(10):
    # Load data
    test_X, test_Y = torch.load(f"processed/test_node_{node_id}.pt", weights_only=False)
    test_X, test_Y = test_X.to(device), test_Y.to(device)

    # Load model
    model = NodeLSTMModel(test_X.shape[1]).to(device)
    model.load_state_dict(torch.load(f"processed/best_model_node_{node_id}.pt", weights_only=False))
    model.eval()

    # Predict
    with torch.no_grad():
        logits = model(test_X)
        probabilities = torch.sigmoid(logits).cpu().numpy()

    y_true = test_Y.cpu().numpy().flatten()
    y_prob = probabilities.flatten()

    # Precision-Recall curve to find best threshold
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    best_f1, best_threshold = 0, 0
    for p, r, t in zip(precision, recall, thresholds):
        if p + r > 0:
            f1 = 2 * p * r / (p + r)
            if f1 > best_f1:
                best_f1, best_threshold = f1, t

    print(f"Node {node_id}: Best Threshold = {best_threshold:.3f}, Best F1 = {best_f1:.3f}")

    # Binary predictions using best threshold
    pred = (y_prob > best_threshold).astype(int)

    # Evaluate with binary metrics
    print(f"  Precision = {precision_score(y_true, pred):.3f}")
    print(f"  Recall    = {recall_score(y_true, pred):.3f}")
    print(f"  F1 Score  = {f1_score(y_true, pred):.3f}")

    # Plot forecast vs actual vs predicted
    plt.figure(figsize=(12, 5))
    plt.plot(y_true, label="Actual")
    plt.plot(y_prob, label="Probability")
    plt.plot(pred, label=f"Prediction (>{best_threshold:.2f})")
    plt.legend()
    plt.grid()
    plt.title(f"Forecast for Node {node_id}")
    plt.savefig(f"results/forecast_vs_actual_node_{node_id}.png", dpi=300)
    plt.close()

    # === SAVE FOR LATER ANALYSIS ===
    # Save per-node probabilities (for anomaly visualization later)
    prob_df = pd.DataFrame({'probability': y_prob})
    prob_df.to_csv(f'results/per_node_predictions/node_{node_id}_probabilities.csv', index=False)

    # Save actual labels and predicted labels
    actual_pred_df = pd.DataFrame({'actual': y_true, 'predicted': pred})
    actual_pred_df.to_csv(f'results/per_node_predictions/node_{node_id}_actual_vs_predicted.csv', index=False)

    print(f"Saved probabilities and predictions for Node {node_id}.")

