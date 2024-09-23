import jax
import jax.numpy as jnp
from NeuroFlex.bci_integration.neuroscience_models import SpikingNeuralNetwork, LSTMNetwork, TemporalConvNet

def test_spiking_neural_network():
    print("Testing SpikingNeuralNetwork...")
    key = jax.random.PRNGKey(0)
    input_size, hidden_size, output_size = 32, 64, 2
    model = SpikingNeuralNetwork(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
    params = model.init(key, jnp.ones((1, 10, input_size)))

    # Test forward pass
    x = jax.random.normal(key, (1, 10, input_size))
    output = model.apply(params, x)
    print(f"Input shape: {x.shape}, Output shape: {output.shape}")
    print("SpikingNeuralNetwork test completed.")

def test_lstm_network():
    print("Testing LSTMNetwork...")
    key = jax.random.PRNGKey(0)
    input_size, hidden_size, output_size = 32, 64, 2
    model = LSTMNetwork(hidden_size=hidden_size, output_size=output_size)
    params = model.init(key, jnp.ones((1, 10, input_size)))

    # Test forward pass
    x = jax.random.normal(key, (1, 10, input_size))
    output = model.apply(params, x)
    print(f"Input shape: {x.shape}, Output shape: {output.shape}")
    print("LSTMNetwork test completed.")

def test_temporal_conv_net():
    print("Testing TemporalConvNet...")
    key = jax.random.PRNGKey(0)
    input_size, hidden_size, output_size = 32, 64, 2
    model = TemporalConvNet(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
    params = model.init(key, jnp.ones((1, 10, input_size)))

    # Test forward pass
    x = jax.random.normal(key, (1, 10, input_size))
    output = model.apply(params, x)
    print(f"Input shape: {x.shape}, Output shape: {output.shape}")
    print("TemporalConvNet test completed.")

if __name__ == "__main__":
    test_spiking_neural_network()
    print("\n")
    test_lstm_network()
    print("\n")
    test_temporal_conv_net()
