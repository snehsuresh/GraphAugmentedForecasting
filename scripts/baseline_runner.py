import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import logging
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
# baseline_runner.py (FINAL - Works with new arima_baseline)

import os
import pandas as pd
import numpy as np
from models.arima_baseline import run_arima
from models.prophet_baseline import run_prophet

def compute_dynamic_threshold(errors, multiplier=3):
    """Computes a robust dynamic threshold using median and median absolute deviation (MAD)."""
    median = np.median(errors)
    mad = np.median(np.abs(errors - median))
    return median + multiplier * mad

datasets = ['yahoo_s5', 'metrla']
save_dir = 'results'
os.makedirs(save_dir, exist_ok=True)

metrics = []

for dataset in datasets:
    for node in range(25):
        if dataset == 'yahoo_s5':
            train_csv = f'processed/{dataset}/real_{node+1}.csv_train.csv'
            test_csv = f'processed/{dataset}/real_{node+1}.csv_test.csv'
            relabel_flag = False
        else:
            train_csv = f'processed/{dataset}/node_{node}_train.csv'
            test_csv = f'processed/{dataset}/node_{node}_test.csv'
            relabel_flag = True  # Apply relabeling for MetrLA

        # Run ARIMA baseline (note: we now pass dataset and node for diagnostics logging)
        arima_errors, arima_labels = run_arima(train_csv, test_csv, dataset, node, relabel=relabel_flag)
        arima_labels = np.array(arima_labels)

        min_length = min(len(arima_errors), len(arima_labels))
        arima_errors = arima_errors[:min_length]
        arima_labels = arima_labels[:min_length]

        arima_threshold = compute_dynamic_threshold(arima_errors)
        arima_preds = (arima_errors > arima_threshold).astype(int)

        tp = np.sum((arima_preds == 1) & (arima_labels == 1))
        fp = np.sum((arima_preds == 1) & (arima_labels == 0))
        fn = np.sum((arima_preds == 0) & (arima_labels == 1))

        arima_precision = tp / (tp + fp + 1e-6)
        arima_recall = tp / (tp + fn + 1e-6)
        arima_f1 = 2 * (arima_precision * arima_recall) / (arima_precision + arima_recall + 1e-6)

        metrics.append([dataset, node, 'ARIMA', arima_precision, arima_recall, arima_f1])

        # Run Prophet baseline (unchanged)
        from models.prophet_baseline import run_prophet  # Prophet does not need dataset/node, so no change here
        prophet_errors, prophet_labels = run_prophet(train_csv, test_csv, relabel=relabel_flag)
        prophet_labels = np.array(prophet_labels)

        min_length = min(len(prophet_errors), len(prophet_labels))
        prophet_errors = prophet_errors[:min_length]
        prophet_labels = prophet_labels[:min_length]

        prophet_threshold = compute_dynamic_threshold(prophet_errors)
        prophet_preds = (prophet_errors > prophet_threshold).astype(int)

        tp = np.sum((prophet_preds == 1) & (prophet_labels == 1))
        fp = np.sum((prophet_preds == 1) & (prophet_labels == 0))
        fn = np.sum((prophet_preds == 0) & (prophet_labels == 1))

        prophet_precision = tp / (tp + fp + 1e-6)
        prophet_recall = tp / (tp + fn + 1e-6)
        prophet_f1 = 2 * (prophet_precision * prophet_recall) / (prophet_precision + prophet_recall + 1e-6)

        metrics.append([dataset, node, 'Prophet', prophet_precision, prophet_recall, prophet_f1])

df = pd.DataFrame(metrics, columns=['Dataset', 'Node', 'Model', 'Precision', 'Recall', 'F1'])
df.to_csv(f'{save_dir}/baseline_metrics.csv', index=False)

print("✅ Saved baseline evaluation results to CSV.")
