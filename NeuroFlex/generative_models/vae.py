import jax
import jax.numpy as jnp
import flax.linen as nn

class VAE(nn.Module):
    latent_dim: int  # Dimension of the latent space (z)
    hidden_dim: int  # Dimension of the hidden layers in encoder/decoder
    input_shape: tuple  # Shape of the input data (e.g., (28, 28, 1) for MNIST)

    def setup(self):
        """
        Setup the encoder and decoder networks.

        This function defines the structure of the encoder and decoder using
        sequential layers. The encoder outputs the mean and log variance, which
        are used for reparameterization, while the decoder reconstructs the
        input data from the latent representation.
        """
        self.encoder = nn.Sequential([
            nn.Dense(self.hidden_dim),  # First hidden layer
            nn.relu,                    # ReLU activation
            nn.Dense(self.hidden_dim),  # Second hidden layer
            nn.relu,                    # ReLU activation
            nn.Dense(self.latent_dim * 2)  # Output layer for mean and log variance (2 * latent_dim)
        ])

        # Convert input_shape to JAX ndarray
        input_size = jnp.prod(jnp.array(self.input_shape))

        self.decoder = nn.Sequential([
            nn.Dense(self.hidden_dim),  # First hidden layer
            nn.relu,                    # ReLU activation
            nn.Dense(self.hidden_dim),  # Second hidden layer
            nn.relu,                    # ReLU activation
            nn.Dense(input_size),  # Output layer to match input size
            lambda x: x.reshape((-1,) + self.input_shape)  # Reshape to the original input shape
        ])

    def __call__(self, x, rng):
        """
        Forward pass through the VAE.

        Args:
            x (jnp.ndarray): Input data
            rng (jax.random.PRNGKey): Random key for sampling

        Returns:
            Tuple: (Reconstructed data, mean of latent variables, log variance of latent variables)
        """
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar, rng)
        return self.decode(z), mean, logvar

    def encode(self, x):
        """
        Encode the input data into latent variables.

        Args:
            x (jnp.ndarray): Input data

        Returns:
            Tuple: (mean, log variance) of the latent variables
        """
        h = self.encoder(x.reshape((x.shape[0], -1)))
        return jnp.split(h, 2, axis=-1)  # Split into mean and log variance

    def decode(self, z):
        """
        Decode the latent variables to reconstruct the input data.

        Args:
            z (jnp.ndarray): Latent variables

        Returns:
            jnp.ndarray: Reconstructed data
        """
        return self.decoder(z)

    def reparameterize(self, mean, logvar, rng):
        """
        Reparameterize to sample from the latent space.

        Args:
            mean (jnp.ndarray): Mean of the latent variables
            logvar (jnp.ndarray): Log variance of the latent variables
            rng (jax.random.PRNGKey): Random key for sampling

        Returns:
            jnp.ndarray: Sampled latent variables
        """
        std = jnp.exp(0.5 * logvar)
        eps = jax.random.normal(rng, mean.shape)
        return mean + eps * std  # Reparameterization trick

    def kl_divergence(self, mean, logvar):
        """
        Calculate the Kullback-Leibler divergence between the latent distribution and a standard normal distribution.

        Args:
            mean (jnp.ndarray): Mean of the latent variables
            logvar (jnp.ndarray): Log variance of the latent variables

        Returns:
            jnp.ndarray: KL divergence for each sample
        """
        return -0.5 * jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar), axis=-1)

    def reconstruction_loss(self, x_recon, x):
        """
        Calculate the reconstruction loss (Mean Squared Error) between the original and reconstructed data.

        Args:
            x_recon (jnp.ndarray): Reconstructed data
            x (jnp.ndarray): Original input data

        Returns:
            jnp.ndarray: Reconstruction loss for each sample
        """
        return jnp.sum(jnp.square(x_recon - x), axis=(1, 2, 3))

    def loss_function(self, x_recon, x, mean, logvar):
        """
        Calculate the total loss for the VAE (KL divergence + Reconstruction loss).

        Args:
            x_recon (jnp.ndarray): Reconstructed data
            x (jnp.ndarray): Original input data
            mean (jnp.ndarray): Mean of the latent variables
            logvar (jnp.ndarray): Log variance of the latent variables

        Returns:
            jnp.ndarray: Total loss for the VAE
        """
        kl_div = self.kl_divergence(mean, logvar)
        recon_loss = self.reconstruction_loss(x_recon, x)
        return jnp.mean(recon_loss + kl_div)
