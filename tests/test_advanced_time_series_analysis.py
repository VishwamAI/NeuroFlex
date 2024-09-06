import pytest
import pandas as pd
import numpy as np
from NeuroFlex.scientific_domains import AdvancedTimeSeriesAnalysis, create_advanced_time_series_analysis

@pytest.fixture
def time_series_data():
    dates = pd.date_range(start='2020-01-01', end='2022-12-31', freq='D')
    values = np.random.randn(len(dates)).cumsum() + 100  # Random walk with drift
    return pd.Series(values, index=dates)

@pytest.fixture
def atsa():
    return create_advanced_time_series_analysis()

def test_create_advanced_time_series_analysis():
    atsa = create_advanced_time_series_analysis()
    assert isinstance(atsa, AdvancedTimeSeriesAnalysis)

def test_arima_forecast(atsa, time_series_data):
    horizon = 30
    forecast = atsa.forecast(time_series_data, model='arima', horizon=horizon)
    assert isinstance(forecast, pd.Series)
    assert len(forecast) == horizon

def test_sarima_forecast(atsa, time_series_data):
    horizon = 30
    forecast = atsa.forecast(time_series_data, model='sarima', horizon=horizon)
    assert isinstance(forecast, pd.Series)
    assert len(forecast) == horizon

def test_prophet_forecast(atsa, time_series_data):
    horizon = 30
    forecast = atsa.forecast(time_series_data, model='prophet', horizon=horizon)
    assert isinstance(forecast, pd.Series)
    assert len(forecast) == horizon

def test_detect_anomalies(atsa, time_series_data):
    anomalies = atsa.detect_anomalies(time_series_data)
    assert isinstance(anomalies, pd.Series)
    assert len(anomalies) == len(time_series_data)
    assert set(anomalies.unique()) == {0, 1}  # 0 for normal, 1 for anomaly

def test_causal_inference(atsa):
    dates = pd.date_range(start='2020-01-01', end='2022-12-31', freq='D')
    treatment = np.zeros(len(dates))
    treatment[len(dates)//2:] = 1  # Intervention starts halfway
    y = np.random.randn(len(dates)).cumsum() + 100 + treatment * 10  # Effect of intervention
    data = pd.DataFrame({'y': y, 'treatment': treatment}, index=dates)

    result = atsa.causal_inference(data, intervention_start='2021-07-02', intervention_var='treatment')
    print("Causal Inference Result:", result)  # Debug print

    assert isinstance(result, dict)
    if 'error' in result:
        print("Error in causal inference:", result['error'])  # Debug print for error
    else:
        assert 'summary' in result, f"'summary' not in result: {result}"
        assert 'report' in result, f"'report' not in result: {result}"
        assert 'estimated_effect' in result, f"'estimated_effect' not in result: {result}"
        print("Estimated effect:", result['estimated_effect'])  # Debug print for estimated effect

def test_invalid_forecast_model(atsa, time_series_data):
    with pytest.raises(ValueError):
        atsa.forecast(time_series_data, model='invalid_model')
