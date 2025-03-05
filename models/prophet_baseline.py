# prophet_baseline.py (with relabeling support)

from prophet import Prophet
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

def check_seasonality(series):
    """Detects seasonality by decomposing the series."""
    decomposition = seasonal_decompose(series, period=24, model='additive', extrapolate_trend='freq')
    seasonal_std = decomposition.seasonal.std()
    return seasonal_std > 0.1

def run_prophet(train_csv, test_csv, relabel=False):
    """
    Preprocess & run Prophet model.
    If relabel=True, applies Isolation Forest relabeling.
    """
    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)
    
    if relabel:
        from models.relabeling import relabel_df
        train = relabel_df(train)
        test = relabel_df(test)

    train['ds'] = pd.date_range(start='2020-01-01', periods=len(train), freq='H')
    test['ds'] = pd.date_range(start='2020-01-01', periods=len(test), freq='H')
    train['y'] = train['value']

    disable_seasonality = not check_seasonality(train['y'])

    model = Prophet(
        yearly_seasonality=not disable_seasonality,
        weekly_seasonality=not disable_seasonality,
        daily_seasonality=not disable_seasonality
    )
    
    model.fit(train[['ds', 'y']])

    forecast = model.predict(test[['ds']])
    forecast['yhat'] = forecast['yhat'].clip(lower=0)

    errors = (forecast['yhat'] - test['value']).abs()
    
    return errors, test['label'].values
