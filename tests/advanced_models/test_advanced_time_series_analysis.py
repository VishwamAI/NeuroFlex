import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from NeuroFlex.advanced_models.advanced_time_series_analysis import AdvancedTimeSeriesAnalysis
from statsmodels.tools.sm_exceptions import ValueWarning, EstimationWarning

@pytest.fixture
def sample_data():
    dates = pd.date_range(start='2020-01-01', end='2022-12-31', freq='D')
    values = np.random.randn(len(dates)).cumsum() + 100
    return pd.DataFrame({'ds': dates, 'y': values})

@pytest.fixture
def time_series_analyzer():
    return AdvancedTimeSeriesAnalysis()

def test_analyze_invalid_method(time_series_analyzer, sample_data):
    with pytest.raises(ValueError, match="Unsupported method: invalid_method"):
        time_series_analyzer.analyze('invalid_method', sample_data)

@pytest.mark.parametrize("method", ['arima', 'sarima', 'prophet'])
def test_analyze_valid_methods(time_series_analyzer, sample_data, method):
    result = time_series_analyzer.analyze(method, sample_data)
    assert 'forecast' in result
    assert 'actual' in result
    assert isinstance(result['forecast'], (pd.Series, pd.DataFrame))
    assert isinstance(result['actual'], (pd.Series, pd.DataFrame))

def test_arima_forecast(time_series_analyzer, sample_data):
    # Test with Series input
    result = time_series_analyzer.arima_forecast(sample_data['y'])
    assert 'forecast' in result
    assert 'actual' in result
    assert 'model_summary' in result

    # Test with DataFrame input
    result_df = time_series_analyzer.arima_forecast(sample_data)
    assert 'forecast' in result_df
    assert 'actual' in result_df
    assert 'model_summary' in result_df

    # Test with non-numeric data
    with pytest.raises(ValueError, match="Input data must be numeric for ARIMA forecast"):
        time_series_analyzer.arima_forecast(pd.Series(['a', 'b', 'c']))

    # Test with DataFrame missing 'y' column
    with pytest.raises(ValueError, match="DataFrame must contain a 'y' column for ARIMA forecast"):
        time_series_analyzer.arima_forecast(pd.DataFrame({'x': [1, 2, 3]}))

def test_sarima_forecast(time_series_analyzer, sample_data):
    # Test with Series input
    result_series = time_series_analyzer.sarima_forecast(sample_data['y'])
    assert 'forecast' in result_series
    assert 'actual' in result_series
    assert 'model_summary' in result_series

    # Test with DataFrame input
    result_df = time_series_analyzer.sarima_forecast(sample_data)
    assert 'forecast' in result_df
    assert 'actual' in result_df
    assert 'model_summary' in result_df

    # Test with non-numeric data
    with pytest.raises(ValueError, match="Input data must be numeric for SARIMA forecast"):
        time_series_analyzer.sarima_forecast(pd.Series(['a', 'b', 'c']))

    # Test with DataFrame missing 'y' column
    with pytest.raises(ValueError, match="DataFrame must contain a 'y' column for SARIMA forecast"):
        time_series_analyzer.sarima_forecast(pd.DataFrame({'x': [1, 2, 3]}))

def test_prophet_forecast(time_series_analyzer, sample_data):
    result = time_series_analyzer.prophet_forecast(sample_data)
    assert 'forecast' in result
    assert 'actual' in result
    assert 'model' in result

def test_calculate_performance(time_series_analyzer):
    result = {
        'forecast': pd.Series([1, 2, 3, 4, 5]),
        'actual': pd.Series([1.1, 2.1, 3.1, 4.1, 5.1])
    }
    performance = time_series_analyzer._calculate_performance(result)
    assert 0 <= performance <= 1

def test_self_heal(time_series_analyzer):
    time_series_analyzer.performance = 0.5
    time_series_analyzer._self_heal()
    assert time_series_analyzer.performance >= 0.5

@pytest.mark.parametrize("method, exception, error_message, test_data", [
    ('arima', ValueError, "DataFrame must contain a 'y' column for ARIMA forecast", pd.DataFrame({'x': [1, 2, 3]})),
    ('arima', ValueError, "Input data must be numeric for ARIMA forecast", pd.DataFrame({'y': ['a', 'b', 'c']})),
    ('sarima', ValueError, "DataFrame must contain a 'y' column for SARIMA forecast", pd.DataFrame({'x': [1, 2, 3]})),
    ('sarima', ValueError, "Input data must be numeric for SARIMA forecast", pd.DataFrame({'y': ['a', 'b', 'c']})),
    ('prophet', ValueError, "Invalid DataFrame for Prophet. It must have columns 'ds' and 'y' with the dates and values respectively.", pd.DataFrame({'x': [1, 2, 3]})),
])
def test_analyze_exceptions(time_series_analyzer, method, exception, error_message, test_data):
    with pytest.raises(exception) as exc_info:
        time_series_analyzer.analyze(method, test_data)
    assert error_message in str(exc_info.value)

@pytest.mark.parametrize("method, exception", [
    ('arima', (ValueWarning, EstimationWarning)),
    ('sarima', (ValueWarning, EstimationWarning)),
])
def test_analyze_warnings(time_series_analyzer, sample_data, method, exception):
    with pytest.warns(exception):
        time_series_analyzer.analyze(method, sample_data)

def test_update_performance(time_series_analyzer):
    initial_performance = time_series_analyzer.performance
    time_series_analyzer._update_performance({'forecast': pd.Series([1, 2, 3]), 'actual': pd.Series([1.1, 2.1, 3.1])})
    assert time_series_analyzer.performance != initial_performance
    assert len(time_series_analyzer.performance_history) > 0

if __name__ == "__main__":
    pytest.main()
