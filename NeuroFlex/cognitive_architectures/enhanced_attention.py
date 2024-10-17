import jax.numpy as jnp
import flax.linen as nn

class EnhancedAttention(nn.Module):
    num_heads: int
    qkv_features: int
    out_features: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, mask=None, deterministic=False):
        attention = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.qkv_features,
            out_features=self.out_features,
            dropout_rate=self.dropout_rate
        )
        x = attention(x, x, mask=mask, deterministic=deterministic)
        x = nn.LayerNorm()(x)
        return x

def create_enhanced_attention(num_heads, qkv_features, out_features, dropout_rate=0.1):
    return EnhancedAttention(num_heads=num_heads, qkv_features=qkv_features, out_features=out_features, dropout_rate=dropout_rate)
