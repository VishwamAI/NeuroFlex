import pytest
import numpy as np
import jax
import jax.numpy as jnp
from NeuroFlex.bci_integration.real_time_feedback import RealTimeFeedback, NeurofeedbackApplication, ExampleDecoder

@pytest.fixture
def example_signal_processor():
    def processor(raw_signal):
        return jnp.array(raw_signal)
    return processor

@pytest.fixture
def example_feedback_generator():
    def generator(decoded_signal):
        return float(jnp.asarray(decoded_signal).item())
    return generator

@pytest.fixture
def real_time_feedback(example_signal_processor, example_feedback_generator):
    decoder = ExampleDecoder()
    key = jax.random.PRNGKey(0)
    params = decoder.init(key, jnp.ones((1, 64)))  # Initialize with dummy input
    decoder = decoder.bind(params)
    return RealTimeFeedback(example_signal_processor, decoder, example_feedback_generator)

@pytest.fixture
def neurofeedback_application(real_time_feedback):
    return NeurofeedbackApplication(real_time_feedback)

def test_real_time_feedback_initialization(real_time_feedback):
    assert isinstance(real_time_feedback, RealTimeFeedback)
    assert callable(real_time_feedback.signal_processor)
    assert callable(real_time_feedback.feedback_generator)
    assert isinstance(real_time_feedback.decoder, ExampleDecoder)

def test_closed_loop_feedback(real_time_feedback):
    raw_signal = np.random.rand(64)
    feedback = real_time_feedback.closed_loop_feedback(raw_signal)
    assert isinstance(feedback, float)
    assert len(real_time_feedback.feedback_history) == 1

def test_neurofeedback_application_initialization(neurofeedback_application):
    assert isinstance(neurofeedback_application, NeurofeedbackApplication)
    assert isinstance(neurofeedback_application.real_time_feedback, RealTimeFeedback)
    assert neurofeedback_application.adaptation_rate == 0.1

def test_neurofeedback_application_update_performance(neurofeedback_application):
    initial_performance = 0.5
    neurofeedback_application.update_user_performance(initial_performance)
    assert len(neurofeedback_application.user_performance) == 1
    assert neurofeedback_application.user_performance[0] == initial_performance

def test_neurofeedback_application_adapt_feedback(neurofeedback_application):
    neurofeedback_application.update_user_performance(0.5)
    neurofeedback_application.update_user_performance(0.6)
    initial_rate = neurofeedback_application.adaptation_rate
    neurofeedback_application.adapt_feedback()
    assert neurofeedback_application.adaptation_rate > initial_rate

def test_neurofeedback_application_run_session(neurofeedback_application):
    def mock_signal_stream():
        while True:
            yield np.random.rand(64)

    session_duration = 10
    neurofeedback_application.run_session(mock_signal_stream(), session_duration)
    assert len(neurofeedback_application.user_performance) == session_duration

if __name__ == "__main__":
    pytest.main([__file__])
