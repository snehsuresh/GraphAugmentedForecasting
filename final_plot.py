import pandas as pd
import matplotlib.pyplot as plt

# Load data
metrics = pd.read_csv('scripts/results/neural_metrics.csv')
degrees = pd.read_csv('processed/metrla/node_degrees.csv')

# Ensure column names are aligned (in case your degree CSV has different format)
degrees.rename(columns={'node': 'Node', 'degree': 'Degree'}, inplace=True)

# Filter to METR-LA dataset (since degree is only available for METR-LA)
metrla_metrics = metrics[metrics['Dataset'] == 'metrla']

# Compute F1 improvement (Graph-Augmented vs LSTM-Only)
f1_graph = metrla_metrics[metrla_metrics['Model'] == 'Graph-Augmented'].set_index('Node')['F1']
f1_lstm = metrla_metrics[metrla_metrics['Model'] == 'LSTM-Only'].set_index('Node')['F1']

f1_improvement = (f1_graph - f1_lstm).reset_index()
f1_improvement.columns = ['Node', 'F1_Improvement']

# Merge with node degrees
f1_with_degree = pd.merge(f1_improvement, degrees, on='Node')

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(f1_with_degree['Degree'], f1_with_degree['F1_Improvement'], s=80, edgecolor='k', alpha=0.7)
plt.axhline(0, color='gray', linestyle='--')

plt.xlabel('Node Degree (Neighbor Count)')
plt.ylabel('F1 Score Improvement (Graph-Augmented - LSTM-Only)')
plt.title('Per-Node F1 Improvement vs Node Degree (METR-LA)')
plt.grid(True)

# Save the plot
plt.savefig('results/f1_vs_degree.png', dpi=300)

# Show it if needed
# plt.show()

print("Saved plot to results/f1_vs_degree.png")
