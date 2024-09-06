import numpy as np
import pytest
from NeuroFlex.utils import calculate_descriptive_statistics, preprocess_data, analyze_bci_data

def test_calculate_descriptive_statistics():
    # Test with 1D array
    data_1d = np.array([1, 2, 3, 4, 5])
    result_1d = calculate_descriptive_statistics(data_1d)
    assert np.isclose(result_1d['mean'], 3.0)
    assert np.isclose(result_1d['median'], 3.0)
    assert np.isclose(result_1d['variance'], 2.5)  # Sample variance
    assert np.isclose(result_1d['std_dev'], np.sqrt(2.5))  # Sample standard deviation

    # Test with 2D array
    data_2d = np.array([[1, 2, 3], [4, 5, 6]])
    result_2d = calculate_descriptive_statistics(data_2d, axis=1)
    assert np.allclose(result_2d['mean'], np.array([2.0, 5.0]))
    assert np.allclose(result_2d['median'], np.array([2.0, 5.0]))
    assert np.allclose(result_2d['variance'], np.array([1.0, 1.0]))  # Sample variance
    assert np.allclose(result_2d['std_dev'], np.array([1.0, 1.0]))  # Sample standard deviation

    # Test with empty array
    with pytest.raises(ValueError):
        calculate_descriptive_statistics(np.array([]))

def test_preprocess_data():
    data = np.array([1, 2, np.nan, 4, np.inf, 6])
    result = preprocess_data(data)
    assert np.array_equal(result, np.array([1, 2, 0, 4, 0, 6]))
    assert result.shape == data.shape

    # Test with 2D array
    data_2d = np.array([[1, 2, np.nan], [4, np.inf, 6]])
    result_2d = preprocess_data(data_2d)
    assert np.array_equal(result_2d, np.array([[1, 2, 0], [4, 0, 6]]))
    assert result_2d.shape == data_2d.shape

def test_analyze_bci_data():
    bci_data = np.array([[1, 2, np.nan], [4, 5, 6], [7, np.inf, 9]])
    result = analyze_bci_data(bci_data, axis=0)
    # Adjusted expected values after preprocessing (replacing NaN and Inf with 0)
    assert np.allclose(result['mean'], np.array([4.0, 2.33333333, 5.0]), rtol=1e-5)
    assert np.allclose(result['median'], np.array([4.0, 2.0, 6.0]), rtol=1e-5)
    assert np.allclose(result['variance'], np.array([9.0, 6.33333333, 21.0]), rtol=1e-5)  # Sample variance
    assert np.allclose(result['std_dev'], np.array([3.0, 2.51661148, 4.58257569]), rtol=1e-5)  # Sample standard deviation

    # Test with no axis specified
    result_no_axis = analyze_bci_data(bci_data)
    assert np.isclose(result_no_axis['mean'], 3.777777777777778)
    assert np.isclose(result_no_axis['median'], 4.0)
    assert np.isclose(result_no_axis['variance'], 10.444444444444445)  # Sample variance
    assert np.isclose(result_no_axis['std_dev'], 3.2317806001093755)  # Sample standard deviation

    # Test with all NaN or Inf values
    all_nan_inf_data = np.array([[np.nan, np.inf], [np.inf, np.nan]])
    result_all_nan_inf = analyze_bci_data(all_nan_inf_data)
    assert np.allclose(result_all_nan_inf['mean'], np.array([0.0, 0.0]))
    assert np.allclose(result_all_nan_inf['median'], np.array([0.0, 0.0]))
    assert np.allclose(result_all_nan_inf['variance'], np.array([0.0, 0.0]))
    assert np.allclose(result_all_nan_inf['std_dev'], np.array([0.0, 0.0]))
