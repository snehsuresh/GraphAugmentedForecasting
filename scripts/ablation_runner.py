# ablation_runner.py — Investigate Graph Benefits (Real Graph vs Random Graph)

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from models.graph_augmented_model import GraphAugmentedModel
from models.lstm_model import LSTMOnlyModel
from models.relabeling import relabel_df
import warnings

warnings.simplefilter('ignore', category=FutureWarning)

def compute_dynamic_threshold(errors, multiplier=3):
    median = np.median(errors)
    mad = np.median(np.abs(errors - median))
    return median + multiplier * mad

class TimeSeriesDataset(Dataset):
    def __init__(self, csv_file, seq_length=24, relabel=False):
        self.data = pd.read_csv(csv_file)
        if relabel:
            self.data = relabel_df(self.data)
        self.values = self.data['value'].values
        self.labels = self.data['label'].values
        self.seq_length = seq_length

    def __len__(self):
        return len(self.values) - self.seq_length

    def __getitem__(self, idx):
        x = self.values[idx:idx+self.seq_length]
        y = self.values[idx+self.seq_length]
        label = self.labels[idx+self.seq_length]
        return torch.tensor(x, dtype=torch.float32).unsqueeze(-1), torch.tensor(y, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

def load_random_graph(num_nodes=25):
    adj = torch.rand(num_nodes, num_nodes)
    adj = (adj > 0.8).float()  # random sparse graph
    torch.diagonal(adj).fill_(0)  # no self-loops
    return adj

def train_and_evaluate(train_csv, test_csv, model_type, seq_length=24, epochs=5, relabel=False, use_random_graph=False):
    train_ds = TimeSeriesDataset(train_csv, seq_length, relabel)
    test_ds = TimeSeriesDataset(test_csv, seq_length, relabel)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    input_dim = 1
    if model_type == 'graph':
        model = GraphAugmentedModel(input_dim)
    elif model_type == 'lstm':
        model = LSTMOnlyModel(input_dim)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    adj = load_random_graph() if use_random_graph else None

    # Training
    model.train()
    for epoch in range(epochs):
        for x, y, _ in train_loader:
            optimizer.zero_grad()
            if model_type == 'graph':
                outputs = model(x, adj)
            else:
                outputs = model(x)
            loss = criterion(outputs.squeeze(), y)
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    errors, true_labels = [], []
    with torch.no_grad():
        for x, y, label in test_loader:
            if model_type == 'graph':
                outputs = model(x, adj)
            else:
                outputs = model(x)
            errors.extend((outputs.squeeze() - y).abs().cpu().numpy())
            true_labels.extend(label.cpu().numpy())

    errors = np.array(errors)
    true_labels = np.array(true_labels)
    threshold = compute_dynamic_threshold(errors)

    preds = (errors > threshold).astype(int)

    tp = np.sum((preds == 1) & (true_labels == 1))
    fp = np.sum((preds == 1) & (true_labels == 0))
    fn = np.sum((preds == 0) & (true_labels == 1))
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

    return precision, recall, f1

if __name__ == "__main__":
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
                relabel_flag = True
            relabel_flag = (dataset == 'metrla')

            # Real Graph
            p, r, f1 = train_and_evaluate(train_csv, test_csv, 'graph', relabel=relabel_flag, use_random_graph=False)
            metrics.append([dataset, node, 'Graph-Real', p, r, f1])

            # Random Graph
            p, r, f1 = train_and_evaluate(train_csv, test_csv, 'graph', relabel=relabel_flag, use_random_graph=True)
            metrics.append([dataset, node, 'Graph-Random', p, r, f1])

    pd.DataFrame(metrics, columns=['Dataset', 'Node', 'Model', 'Precision', 'Recall', 'F1']).to_csv(f'{save_dir}/ablation_metrics.csv', index=False)
    print("✅ Saved ablation results.")
