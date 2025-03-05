import pandas as pd
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import haversine_distances

# Load sensor locations
df = pd.read_csv('processed/metrla/graph_sensor_locations.csv')
coords = np.radians(df[['latitude', 'longitude']].values)

# Compute pairwise haversine distances (in radians)
distances = haversine_distances(coords, coords) * 6371  # Convert to km

# Threshold graph (connect sensors within 0.5 km)
adj_matrix = (distances <= 0.5).astype(int)
np.fill_diagonal(adj_matrix, 0)

# Build graph and compute degree
G = nx.from_numpy_array(adj_matrix)
degrees = dict(G.degree())

# Save degree list
degree_df = pd.DataFrame.from_dict(degrees, orient='index', columns=['degree'])
degree_df.index.name = 'node'
degree_df.to_csv('processed/metrla/node_degrees.csv')
