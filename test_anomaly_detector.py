import pytest
import jax
import jax.numpy as jnp
from edge_ai_optimization import EdgeAIOptimization, AnomalyDetector

@pytest.fixture
def edge_ai():
    return EdgeAIOptimization()

@pytest.fixture
def anomaly_detector():
    return AnomalyDetector()

def test_anomaly_detector(edge_ai, anomaly_detector):
    # Create dummy data
    key = jax.random.PRNGKey(0)
    dummy_data = jax.random.normal(key, (10, 784))  # 10 samples, 784 features

    # Detect anomalies
    anomaly_scores = edge_ai.detect_anomalies(dummy_data)

    # Check output shape
    assert anomaly_scores.shape == (10, 1)

    # Check if values are within expected range
    assert jnp.all(jnp.isfinite(anomaly_scores))  # Check for NaN or infinity
    assert jnp.min(anomaly_scores) >= -1e6 and jnp.max(anomaly_scores) <= 1e6  # Reasonable range

    # Test the AnomalyDetector directly
    params = anomaly_detector.init(jax.random.PRNGKey(0), dummy_data)
    detector_output = anomaly_detector.apply(params, dummy_data)
    assert detector_output.shape == (10, 1)
    assert jnp.all(jnp.isfinite(detector_output))
    assert jnp.min(detector_output) >= -1e6 and jnp.max(detector_output) <= 1e6

def test_anomaly_detector_training(edge_ai, anomaly_detector):
    # Create dummy training data
    key = jax.random.PRNGKey(1)
    train_data = jax.random.normal(key, (100, 784))  # 100 samples, 784 features

    # Train the anomaly detector
    edge_ai.train_anomaly_detector(train_data)

    # Create dummy test data with an anomaly
    test_data = jax.random.normal(key, (10, 784))
    anomaly = jnp.ones((1, 784)) * 10  # Create an obvious anomaly
    test_data = jnp.vstack([test_data, anomaly])

    # Detect anomalies
    anomaly_scores = edge_ai.detect_anomalies(test_data)

    # Check if the anomaly is detected
    assert anomaly_scores[-1] > anomaly_scores[:-1].mean()

def test_self_healing_mechanism(edge_ai):
    # Create dummy data with anomalies
    key = jax.random.PRNGKey(2)
    normal_data = jax.random.normal(key, (100, 784))
    anomalies = jnp.ones((10, 784)) * 10
    data = jnp.vstack([normal_data, anomalies])

    # Train the anomaly detector
    edge_ai.train_anomaly_detector(normal_data)

    # Test self-healing mechanism
    healed_data = edge_ai.self_healing_mechanism(data)

    # Check if anomalies are mitigated
    anomaly_scores_before = edge_ai.detect_anomalies(data)
    anomaly_scores_after = edge_ai.detect_anomalies(healed_data)

    assert jnp.mean(anomaly_scores_after) < jnp.mean(anomaly_scores_before)

if __name__ == "__main__":
    pytest.main([__file__])
