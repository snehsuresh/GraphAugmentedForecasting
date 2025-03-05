import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from models.graph_augmented_model import GraphAugmentedModel
from models.lstm_model import LSTMOnlyModel
from models.relabeling import relabel_df
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

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

def train_model(model, train_loader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for x, y, _ in train_loader:
            optimizer.zero_grad()
            try:
                outputs = model(x, None)
            except TypeError:
                outputs = model(x)
            loss = criterion(outputs.squeeze(), y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    return model

def evaluate_model(model, test_loader):
    model.eval()
    errors = []
    true_labels = []
    with torch.no_grad():
        for x, y, label in test_loader:
            try:
                outputs = model(x, None)
            except TypeError:
                outputs = model(x)
            error = torch.abs(outputs.squeeze() - y)
            errors.extend(error.cpu().numpy())
            true_labels.extend(label.cpu().numpy())
    errors = np.array(errors)
    true_labels = np.array(true_labels)
    threshold = compute_dynamic_threshold(errors)
    preds = (errors > threshold).astype(int)
    return errors, preds, true_labels

def compute_metrics(preds, labels):
    tp = np.sum((preds == 1) & (labels == 1))
    fp = np.sum((preds == 1) & (labels == 0))
    fn = np.sum((preds == 0) & (labels == 1))
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    return precision, recall, f1

def run_neural_models(train_csv, test_csv, model_type='graph', seq_length=24, epochs=10, relabel=False):
    train_dataset = TimeSeriesDataset(train_csv, seq_length=seq_length, relabel=relabel)
    test_dataset = TimeSeriesDataset(test_csv, seq_length=seq_length, relabel=relabel)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    input_dim = 1
    if model_type == 'graph':
        model = GraphAugmentedModel(input_dim)
    elif model_type == 'lstm':
        model = LSTMOnlyModel(input_dim)
    else:
        raise ValueError("Unknown model type")
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print(f"Training {model_type} model on {train_csv} ...")
    model = train_model(model, train_loader, criterion, optimizer, epochs=epochs)
    errors, preds, true_labels = evaluate_model(model, test_loader)
    precision, recall, f1 = compute_metrics(preds, true_labels)
    
    return errors, preds, true_labels, precision, recall, f1

if __name__ == "__main__":
    datasets = ['yahoo_s5', 'metrla']
    save_dir = 'results'
    os.makedirs(save_dir, exist_ok=True)

    metrics = []

    for dataset in datasets:
        for node in range(25):
            if dataset == 'yahoo_s5':
                train_csv = f'../processed/{dataset}/real_{node+1}.csv_train.csv'
                test_csv = f'../processed/{dataset}/real_{node+1}.csv_test.csv'
                relabel_flag = False
            else:
                train_csv = f'../processed/{dataset}/node_{node}_train.csv'
                test_csv = f'../processed/{dataset}/node_{node}_test.csv'
                relabel_flag = True

            _, _, _, precision, recall, f1 = run_neural_models(train_csv, test_csv, model_type='graph', epochs=5, relabel=relabel_flag)
            metrics.append([dataset, node, 'Graph-Augmented', precision, recall, f1])

            _, _, _, precision, recall, f1 = run_neural_models(train_csv, test_csv, model_type='lstm', epochs=5, relabel=relabel_flag)
            metrics.append([dataset, node, 'LSTM-Only', precision, recall, f1])

    df = pd.DataFrame(metrics, columns=['Dataset', 'Node', 'Model', 'Precision', 'Recall', 'F1'])
    df.to_csv(f'{save_dir}/neural_metrics.csv', index=False)
    print("✅ Saved neural model evaluation results to CSV.")
