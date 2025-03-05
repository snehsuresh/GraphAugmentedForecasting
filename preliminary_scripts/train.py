import torch
import torch.nn as nn
import torch.optim as optim
from model import NodeLSTMModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for node_id in range(10):
    train_X, train_Y = torch.load(f"processed/train_node_{node_id}.pt", weights_only=False)
    val_X, val_Y = torch.load(f"processed/val_node_{node_id}.pt", weights_only=False)

    train_X, train_Y = train_X.to(device), train_Y.to(device)
    val_X, val_Y = val_X.to(device), val_Y.to(device)

    model = NodeLSTMModel(train_X.shape[1]).to(device)

    pos_weight = (train_Y.numel() - train_Y.sum()) / (train_Y.sum() + 1e-6)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_val_loss = float('inf')

    for epoch in range(20):
        model.train()
        logits = model(train_X)
        loss = criterion(logits, train_Y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            val_logits = model(val_X)
            val_loss = criterion(val_logits, val_Y)

        print(f"Node {node_id}, Epoch {epoch+1}: Train Loss = {loss:.4f}, Val Loss = {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"processed/best_model_node_{node_id}.pt")
            print(f"Best model saved for Node {node_id}")
