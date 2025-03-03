import numpy as np
import pandas as pd
import networkx as nx
import os

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
num_nodes = 50          # number of event types (nodes)
num_timesteps = 1000    # length of the time series
base_prob = 0.01        # base probability for an event at any time
bonus_prob = 0.05       # additional probability if any neighbor had an event in the previous timestep

# Generate a random graph using networkx (Erdős-Rényi graph)
graph_prob = 0.1  # probability of edge creation
G = nx.erdos_renyi_graph(n=num_nodes, p=graph_prob, seed=42, directed=False)

# Get edge list in a DataFrame
edges = list(G.edges())
edge_df = pd.DataFrame(edges, columns=["source", "target"])

# Create a time series data frame: rows = timesteps, columns = node_0, node_1, ... node_N
data = np.zeros((num_timesteps, num_nodes), dtype=np.int32)

# Initialize first timestep
data[0, :] = np.random.binomial(1, base_prob, size=num_nodes)

# Precompute neighbors for efficiency
neighbors = {node: list(G.neighbors(node)) for node in G.nodes()}

# Generate events over time
for t in range(1, num_timesteps):
    for node in range(num_nodes):
        # Check if any neighbor had an event in the previous timestep
        neighbor_events = any(data[t-1, nbr] == 1 for nbr in neighbors[node])
        prob = base_prob + (bonus_prob if neighbor_events else 0)
        # Ensure probability does not exceed 1
        prob = min(prob, 1.0)
        data[t, node] = np.random.binomial(1, prob)

# Create a DataFrame and add a time column
time_steps = np.arange(num_timesteps)
df = pd.DataFrame(data, columns=[f"node_{i}" for i in range(num_nodes)])
df.insert(0, "time", time_steps)

# Create output directory if needed
os.makedirs("data", exist_ok=True)

# Save time series and graph
df.to_csv("data/time_series.csv", index=False)
edge_df.to_csv("data/graph_edges.csv", index=False)

print("Data generation complete. Files saved to the 'data' folder.")
