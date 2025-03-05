# arima_baseline.py (FINAL CLEAN with explicit frequency)
# arima_baseline.py (Research-Grade)
import os
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import pmdarima as pm
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# Prepare diagnostics output
diagnostics_file = 'results/arima_diagnostics.csv'
if not os.path.exists('results'):
    os.makedirs('results')

if not os.path.exists(diagnostics_file):
    with open(diagnostics_file, 'w') as f:
        f.write("dataset,node,converged,non_stationary_ar,non_invertible_ma\n")

# Suppress other ARIMA warnings we don’t care about
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

def check_stationarity(series):
    result = adfuller(series.dropna())
    return result[1] < 0.05  # p-value < 0.05 means stationary

def log_diagnostics(dataset, node, converged, non_stationary_ar, non_invertible_ma):
    with open(diagnostics_file, 'a') as f:
        f.write(f"{dataset},{node},{converged},{non_stationary_ar},{non_invertible_ma}\n")

def run_arima(train_csv, test_csv, dataset, node, relabel=False):
    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)

    if relabel:
        from models.relabeling import relabel_df
        train = relabel_df(train)
        test = relabel_df(test)

    # Set proper datetime index to avoid freq warnings
    train['timestamp'] = pd.date_range(start='2020-01-01', periods=len(train), freq='H')
    train = train.set_index('timestamp')
    train.index.freq = 'H'

    test['timestamp'] = pd.date_range(start=train.index[-1] + pd.Timedelta(hours=1), periods=len(test), freq='H')
    test = test.set_index('timestamp')
    test.index.freq = 'H'

    train_series = train['value']
    test_series = test['value']

    if not check_stationarity(train_series):
        train_series = train_series.diff().dropna()
        test_series = test_series.diff().dropna()

    # Track diagnostics
    converged = True
    non_stationary_ar = False
    non_invertible_ma = False

    def custom_warn(message, category, filename, lineno, file=None, line=None):
        nonlocal non_stationary_ar, non_invertible_ma
        msg = str(message)
        if "Non-stationary starting autoregressive parameters" in msg:
            non_stationary_ar = True
        elif "Non-invertible starting MA parameters" in msg:
            non_invertible_ma = True
        elif "Maximum Likelihood optimization failed to converge" in msg:
            converged = False

    with warnings.catch_warnings():
        warnings.simplefilter('always')
        warnings.showwarning = custom_warn

        # AutoARIMA to detect best order
        model = pm.auto_arima(train_series, seasonal=False, stepwise=True, suppress_warnings=True)

        # Fit ARIMA using detected order
        model_fit = ARIMA(train_series, order=model.order).fit()

    # Forecast future values
    forecast = model_fit.forecast(steps=len(test_series))

    # Compute absolute errors
    errors = (forecast - test_series).abs()

    # Log diagnostics per node
    log_diagnostics(dataset, node, converged, non_stationary_ar, non_invertible_ma)

    return errors, test['label'].values
