import pytest
import jax
import jax.numpy as jnp
from flax import linen as nn
from NeuroFlex.generative_models.vae import VAE

@pytest.fixture
def vae_model():
    return VAE(latent_dim=10, hidden_dim=64, input_shape=(28, 28, 1))

@pytest.fixture
def initialized_vae(vae_model):
    key = jax.random.PRNGKey(0)
    input_shape = (1,) + vae_model.input_shape
    params = vae_model.init(key, jnp.ones(input_shape), key)
    return params, vae_model

def test_vae_creation(vae_model):
    assert isinstance(vae_model, VAE)
    assert vae_model.latent_dim == 10
    assert vae_model.hidden_dim == 64
    assert vae_model.input_shape == (28, 28, 1)

def test_vae_encode(initialized_vae):
    params, vae_model = initialized_vae
    x = jnp.ones((1, 28, 28, 1))
    mean, logvar = vae_model.apply(params, x, method=vae_model.encode)
    assert mean.shape == (1, 10)
    assert logvar.shape == (1, 10)

def test_vae_decode(initialized_vae):
    params, vae_model = initialized_vae
    z = jnp.ones((1, 10))
    x_recon = vae_model.apply(params, z, method=vae_model.decode)
    assert x_recon.shape == (1, 28, 28, 1)

def test_vae_forward_pass(initialized_vae):
    params, vae_model = initialized_vae
    x = jnp.ones((1, 28, 28, 1))
    rng = jax.random.PRNGKey(0)
    x_recon, mean, logvar = vae_model.apply(params, x, rng, method=vae_model.__call__)
    assert x_recon.shape == (1, 28, 28, 1)
    assert mean.shape == (1, 10)
    assert logvar.shape == (1, 10)

def test_vae_loss_calculation(initialized_vae):
    params, vae_model = initialized_vae
    x = jnp.ones((1, 28, 28, 1))
    x_recon = jnp.ones((1, 28, 28, 1))
    mean = jnp.zeros((1, 10))
    logvar = jnp.zeros((1, 10))
    loss = vae_model.apply(params, x_recon, x, mean, logvar, method=vae_model.loss_function)
    assert isinstance(loss, jnp.ndarray)
    assert loss.shape == ()

if __name__ == "__main__":
    pytest.main([__file__])
