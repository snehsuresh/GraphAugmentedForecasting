# train.py

import os
import torch
import numpy as np
import pandas as pd
from models import GraphAugmentedModel, LSTMOnlyModel
from torch.optim import Adam
import torch.nn.functional as F

datasets = ['yahoo_s5', 'metrla']
save_dir = 'results'
os.makedirs(save_dir, exist_ok=True)

def load_data(dataset, node):
    train = pd.read_csv(f'processed/{dataset}/node_{node}_train.csv')
    test = pd.read_csv(f'processed/{dataset}/node_{node}_test.csv')
    return train, test

def train_model(model, train_x, train_y, epochs=20):
    optimizer = Adam(model.parameters(), lr=0.001)
    best_loss = float('inf')

    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(train_x)
        loss = F.binary_cross_entropy_with_logits(output, train_y)
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), f'{save_dir}/{dataset}_node{node}_best_model.pt')
        
        print(f'Node {node}, Epoch {epoch+1}: Loss = {loss.item():.4f}')

for dataset in datasets:
    for node in range(10):  # Adjust for actual node count
        train, test = load_data(dataset, node)
        train_x = torch.tensor(train['value'].values, dtype=torch.float32).unsqueeze(1)
        train_y = torch.tensor(train['label'].values, dtype=torch.float32).unsqueeze(1)

        if dataset == 'yahoo_s5':
            model = GraphAugmentedModel(input_dim=1)
        else:
            model = LSTMOnlyModel(input_dim=1)

        train_model(model, train_x, train_y)
