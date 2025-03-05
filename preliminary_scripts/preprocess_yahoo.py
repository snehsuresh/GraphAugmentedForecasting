# preprocess_yahoo.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split

data_dir = 'data/yahoo/'
output_dir = 'processed/yahoo_s5'
os.makedirs(output_dir, exist_ok=True)

all_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

for file in all_files:
    df = pd.read_csv(os.path.join(data_dir, file))
    series = df['value'].values
    labels = df['is_anomaly'].values

    train_series, test_series, train_labels, test_labels = train_test_split(
        series, labels, test_size=0.3, shuffle=False
    )

    pd.DataFrame({'value': train_series, 'label': train_labels}).to_csv(f'{output_dir}/{file}_train.csv', index=False)
    pd.DataFrame({'value': test_series, 'label': test_labels}).to_csv(f'{output_dir}/{file}_test.csv', index=False)

print("Preprocessed Yahoo S5 (A1Benchmark) into train/test splits.")
