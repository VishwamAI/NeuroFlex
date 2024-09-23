import pytest
import numpy as np
from NeuroFlex.bci_integration.bci_processing import BCIProcessor


@pytest.fixture
def sample_eeg_data():
    # Create a sample EEG data for testing
    return np.random.rand(10, 64, 1000)  # 10 trials, 64 channels, 1000 time points


@pytest.fixture
def sample_labels():
    # Create sample labels for testing
    return np.array([0, 1] * 5)  # Ensure equal distribution of classes for 10 trials


@pytest.fixture
def bci_processor():
    return BCIProcessor(sampling_rate=250, num_channels=64)


def test_preprocess(bci_processor, sample_eeg_data, sample_labels):
    # Ensure the input data is correctly shaped as 3D (trials x channels x samples)
    n_trials, n_channels, n_samples = sample_eeg_data.shape
    assert (
        sample_eeg_data.ndim == 3
    ), "Input data should be 3D (trials x channels x samples)"
    # Ensure at least two unique classes in the labels
    labels = np.array([0, 1] * (n_trials // 2 + 1))[:n_trials]
    assert len(np.unique(labels)) >= 2, "Not enough unique classes for CSP"
    # Ensure labels array has the correct shape and alignment with trials
    assert len(labels) == n_trials, "Number of labels must match number of trials"
    preprocessed_data = bci_processor.preprocess(sample_eeg_data, labels)
    assert (
        preprocessed_data.shape[0] == n_trials
    ), "Number of trials should be preserved"
    assert preprocessed_data.shape[1] <= n_channels, "CSP may reduce dimensionality"
    assert (
        preprocessed_data.shape != sample_eeg_data.shape
    ), "Data should be transformed"


def test_apply_filters(bci_processor, sample_eeg_data):
    # Reshape the input data to 2D
    reshaped_data = sample_eeg_data.reshape(-1, sample_eeg_data.shape[-1])
    filtered_data = bci_processor.apply_filters(reshaped_data)
    assert isinstance(filtered_data, dict)
    for band, data in filtered_data.items():
        assert data.shape == reshaped_data.shape
        assert not np.allclose(data, reshaped_data)  # Ensure data has been filtered


def test_extract_features(bci_processor, sample_eeg_data):
    # Reshape the input data to 2D
    reshaped_data = sample_eeg_data.reshape(-1, sample_eeg_data.shape[-1])
    filtered_data = bci_processor.apply_filters(reshaped_data)
    features = bci_processor.extract_features(filtered_data)
    assert isinstance(features, dict)
    for feature_name, feature_data in features.items():
        if "wavelet" in feature_name:
            # For wavelet features, the shape should be (n_channels, n_wavelet_coeffs)
            expected_shape = (64, 6)
        else:
            # For power features, the shape should be (n_freq_bins, n_channels)
            expected_shape = (129, 64)
        print(f"Shape of {feature_name}: {feature_data.shape}")  # Debug print
        assert (
            feature_data.shape == expected_shape
        ), f"Mismatch in {feature_name} shape. Expected {expected_shape}, got {feature_data.shape}"


def test_process(bci_processor, sample_eeg_data, sample_labels):
    # Reshape the input data to 3D (trials x channels x samples)
    n_trials, n_channels, n_samples = sample_eeg_data.shape
    # Ensure at least two unique classes in the labels
    labels = np.array([0, 1] * (n_trials // 2 + 1))[:n_trials]
    assert len(np.unique(labels)) >= 2, "Not enough unique classes for CSP"
    # Ensure labels array has the correct shape and alignment with trials
    assert len(labels) == n_trials, "Number of labels must match number of trials"
    # Process the data without reshaping
    processed_data = bci_processor.process(sample_eeg_data, labels)
    assert isinstance(processed_data, dict)
    for feature_name, feature_data in processed_data.items():
        if "wavelet" in feature_name:
            # For wavelet features, the shape should be (n_channels, n_wavelet_coeffs)
            assert (
                feature_data.shape[0] == n_channels
            ), f"Expected {n_channels} channels, but got {feature_data.shape[0]} for {feature_name}"
        else:
            # For power features, the shape should be (n_channels, n_freq_bins)
            assert (
                feature_data.shape[0] == n_channels
            ), f"Expected {n_channels} channels, but got {feature_data.shape[0]} for {feature_name}"
            assert (
                feature_data.shape[1] > 0
            ), f"Expected non-zero frequency bins, but got {feature_data.shape[1]} for {feature_name}"
        print(f"Shape of {feature_name}: {feature_data.shape}")  # Debug print


if __name__ == "__main__":
    pytest.main([__file__])
