import pytest
import numpy as np
from NeuroFlex.bci_integration.neuro_data_integration import NeuroDataIntegrator
from NeuroFlex.bci_integration.bci_processing import BCIProcessor


@pytest.fixture
def sample_eeg_data():
    return np.random.rand(10, 64, 1000)  # 10 trials, 64 channels, 1000 time points


@pytest.fixture
def sample_fmri_data():
    return np.random.rand(10, 50, 50, 20)  # 10 trials, 50x50x20 voxels


@pytest.fixture
def sample_behavioral_data():
    return np.random.rand(10, 5)  # 10 trials, 5 behavioral measures


@pytest.fixture
def bci_processor():
    return BCIProcessor(sampling_rate=256, num_channels=64)


@pytest.fixture
def neuro_data_integrator(bci_processor):
    return NeuroDataIntegrator(bci_processor)


def test_neuro_data_integrator_initialization(neuro_data_integrator):
    assert isinstance(neuro_data_integrator, NeuroDataIntegrator)


def test_integrate_eeg_data(neuro_data_integrator, sample_eeg_data):
    # Create labels with at least two unique classes
    labels = np.array([0, 1] * (sample_eeg_data.shape[0] // 2))
    if len(labels) < sample_eeg_data.shape[0]:
        labels = np.append(labels, [0])
    processed_data = neuro_data_integrator.integrate_eeg_data(sample_eeg_data, labels)
    assert processed_data is not None
    assert isinstance(processed_data, dict)
    assert "eeg" in neuro_data_integrator.get_integrated_data()


def test_integrate_external_data(neuro_data_integrator, sample_fmri_data):
    neuro_data_integrator.integrate_external_data("fmri", sample_fmri_data)
    integrated_data = neuro_data_integrator.get_integrated_data()
    assert "fmri" in integrated_data
    assert np.array_equal(integrated_data["fmri"], sample_fmri_data)


def test_get_integrated_data(neuro_data_integrator, sample_eeg_data, sample_fmri_data):
    # Create labels with at least two unique classes
    labels = np.array([0, 1] * (sample_eeg_data.shape[0] // 2))
    if len(labels) < sample_eeg_data.shape[0]:
        labels = np.append(labels, [0])
    neuro_data_integrator.integrate_eeg_data(sample_eeg_data, labels)
    neuro_data_integrator.integrate_external_data("fmri", sample_fmri_data)
    integrated_data = neuro_data_integrator.get_integrated_data()
    assert isinstance(integrated_data, dict)
    assert "eeg" in integrated_data
    assert "fmri" in integrated_data


def test_perform_multimodal_analysis(
    neuro_data_integrator, sample_eeg_data, sample_fmri_data
):
    # Create labels with at least two unique classes
    labels = np.array([0, 1] * (sample_eeg_data.shape[0] // 2))
    if len(labels) < sample_eeg_data.shape[0]:
        labels = np.append(labels, [0])
    neuro_data_integrator.integrate_eeg_data(sample_eeg_data, labels)
    neuro_data_integrator.integrate_external_data("fmri", sample_fmri_data)
    analysis_result = neuro_data_integrator.perform_multimodal_analysis()
    assert analysis_result is not None
    assert isinstance(analysis_result, dict)
    assert "eeg" in analysis_result
    assert "fmri" in analysis_result


if __name__ == "__main__":
    pytest.main([__file__])
