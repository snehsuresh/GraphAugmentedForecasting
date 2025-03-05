import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('results/ablation_metrics.csv')
df = df.pivot_table(index=['Dataset', 'Node'], columns='Model', values='F1')

for dataset in df.index.get_level_values(0).unique():
    plt.figure(figsize=(12, 6))
    sns.heatmap(df.loc[dataset], annot=True, fmt=".2f", cmap='coolwarm')
    plt.title(f'Node-wise F1 Comparison: {dataset}')
    plt.savefig(f'results/{dataset}_nodewise_f1_heatmap.png')
    plt.close()

# Optional: Visualize Gain from Real Graph
df['Graph Benefit'] = df['Graph-Real'] - df['Graph-Random']

for dataset in df.index.get_level_values(0).unique():
    plt.figure(figsize=(12, 6))
    sns.heatmap(df.loc[dataset][['Graph Benefit']], annot=True, fmt=".2f", cmap='RdYlGn')
    plt.title(f'Graph Benefit Per Node: {dataset}')
    plt.savefig(f'results/{dataset}_graph_benefit_heatmap.png')
    plt.close()
