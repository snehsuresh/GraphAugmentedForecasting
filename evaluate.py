import torch
import matplotlib.pyplot as plt
from model import GraphAugmentedForecastingModel
from torch_geometric.data import Data

# Load test data and graph
test_X, test_Y = torch.load("processed/test.pt", weights_only=False)
graph_data = torch.load("processed/graph.pt", weights_only=False)
num_nodes = graph_data.x.shape[0]
input_length = test_X.shape[-1]

# Hyperparameters must match training
lstm_hidden_dim = 64
gcn_hidden_dim = 32
combined_dim = 64
forecast_horizon = 1

# Instantiate model and load best weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GraphAugmentedForecastingModel(num_nodes=num_nodes, input_length=input_length,
                                        lstm_hidden_dim=lstm_hidden_dim,
                                        gcn_hidden_dim=gcn_hidden_dim,
                                        combined_dim=combined_dim,
                                        forecast_horizon=forecast_horizon)
model.load_state_dict(torch.load("processed/best_model.pt", map_location=device, weights_only=False))
model = model.to(device)
model.eval()

# Make predictions on test set
with torch.no_grad():
    predictions = model(test_X.to(device), graph_data.to(device)).cpu()

# For visualization, select a sample node (e.g., node_0) and a range of samples.
sample_node = 0
num_samples_to_plot = 50  # number of test samples to visualize

actual = test_Y[:num_samples_to_plot, sample_node].numpy()
pred = predictions[:num_samples_to_plot, sample_node].numpy()

plt.figure(figsize=(10, 5))
plt.plot(actual, label="Actual")
plt.plot(pred, label="Forecast", linestyle="--")
plt.xlabel("Test Sample Index")
plt.ylabel("Event Value")
plt.title(f"Forecast vs Actual for Node {sample_node}")
plt.legend()
plt.tight_layout()

# Save the plot as a PNG file
plt.savefig("forecast_vs_actual.png", dpi=300)

# Display the plot (optional)
# plt.show()
