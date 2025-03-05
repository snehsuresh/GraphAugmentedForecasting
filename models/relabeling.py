# relabeling.py
import pandas as pd
from sklearn.ensemble import IsolationForest
import numpy as np

def relabel_df(df, contamination=0.01):
    """
    Applies Isolation Forest to the 'value' column and creates a new binary 'label' column.
    Anomalies (predicted as -1) are mapped to 1, and normals to 0.
    """
    clf = IsolationForest(contamination=contamination, random_state=42)
    values = df['value'].values.reshape(-1, 1)
    preds = clf.fit_predict(values)  # 1 for normal, -1 for anomaly
    df['label'] = (preds == -1).astype(int)
    return df
