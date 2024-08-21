import jax
import jax.numpy as jnp
import flax.linen as nn

class VisionTransformer(nn.Module):
    num_classes: int
    patch_size: int = 16
    hidden_size: int = 768
    num_heads: int = 12
    num_layers: int = 12
    mlp_dim: int = 3072
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, train: bool = True):
        # Assuming input shape is (batch_size, height, width, channels)
        B, H, W, C = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, 'Image dimensions must be divisible by patch size.'

        # Split image into patches
        x = jnp.reshape(x, (B, H // self.patch_size, W // self.patch_size, self.patch_size * self.patch_size * C))
        x = jnp.reshape(x, (B, -1, self.patch_size * self.patch_size * C))

        # Embed patches
        x = nn.Dense(self.hidden_size)(x)

        # Add position embeddings
        n_patches = x.shape[1]
        pos_embed = self.param('pos_embed', nn.initializers.normal(stddev=0.02), (1, n_patches, self.hidden_size))
        x = x + pos_embed

        # Transformer encoder
        for _ in range(self.num_layers):
            y = nn.LayerNorm()(x)
            y = nn.MultiHeadDotProductAttention(num_heads=self.num_heads)(y, y)
            y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=not train)
            x = x + y

            y = nn.LayerNorm()(x)
            y = nn.Dense(self.mlp_dim)(y)
            y = nn.gelu(y)
            y = nn.Dense(self.hidden_size)(y)
            y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=not train)
            x = x + y

        # Global average pooling
        x = jnp.mean(x, axis=1)

        # Classification head
        x = nn.LayerNorm()(x)
        x = nn.Dense(self.num_classes)(x)

        return x
