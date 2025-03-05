import os
import pickle
import pandas as pd
import numpy as np

# Paths
data_dir = 'data/metrla/4/'
output_dir = 'processed/metrla/'
os.makedirs(output_dir, exist_ok=True)

# Load adjacency matrix
with open(os.path.join(data_dir, 'adj_METR-LA.pkl'), 'rb') as f:
    sensor_ids, sensor_id_to_index, adj_matrix = pickle.load(f, encoding='latin1')

adj_matrix = np.array(adj_matrix)  # Make sure it's a NumPy array
print(f"✅ Loaded adjacency matrix with shape: {adj_matrix.shape}")

# Load traffic speed data using pandas (correct for this Kaggle version)
df = pd.read_hdf(os.path.join(data_dir, 'METR-LA.h5'), key='df')
print(f"✅ Loaded METR-LA traffic data with shape: {df.shape}")

# Convert to numpy array (time x sensors)
traffic_speeds = df.values

# Train/test split (70% train, 30% test)
train_series = traffic_speeds[:int(0.7 * len(traffic_speeds))]
test_series = traffic_speeds[int(0.7 * len(traffic_speeds)):]

# Generate fake anomaly labels (simple z-score heuristic per sensor)
train_mean = train_series.mean(axis=0)
train_std = train_series.std(axis=0)

train_labels = (train_series > train_mean + 2 * train_std).astype(int)
test_labels = (test_series > train_mean + 2 * train_std).astype(int)

# Save per-sensor (node) train/test files
for node in range(traffic_speeds.shape[1]):
    pd.DataFrame({
        'value': train_series[:, node],
        'label': train_labels[:, node]
    }).to_csv(f'{output_dir}/node_{node}_train.csv', index=False)

    pd.DataFrame({
        'value': test_series[:, node],
        'label': test_labels[:, node]
    }).to_csv(f'{output_dir}/node_{node}_test.csv', index=False)

print("✅ Preprocessed METR-LA into node-wise train/test splits (Kaggle version).")
