import jax
import jax.numpy as jnp
from NeuroFlex.cognitive_architectures.enhanced_attention import EnhancedAttention
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_enhanced_attention():
    logger.info("Starting enhanced attention test")

    try:
        rng = jax.random.PRNGKey(0)
        rng, dropout_rng = jax.random.split(rng)
        batch_size = 1
        seq_length = 10
        input_dim = 64
        num_heads = 4
        qkv_features = 32
        out_features = 64
        dropout_rate = 0.1

        # Initialize the EnhancedAttention
        attention = EnhancedAttention(
            num_heads=num_heads,
            qkv_features=qkv_features,
            out_features=out_features,
            dropout_rate=dropout_rate
        )

        # Create a random input
        x = jax.random.normal(rng, (batch_size, seq_length, input_dim))
        logger.debug(f"Input shape: {x.shape}")

        # Initialize parameters
        params = attention.init({'params': rng, 'dropout': dropout_rng}, x)

        # Apply the EnhancedAttention
        output = attention.apply(params, x, rngs={'dropout': dropout_rng})

        logger.debug(f"Output shape: {output.shape}")

        # Assertions
        assert output.shape == (batch_size, seq_length, out_features), f"Expected shape {(batch_size, seq_length, out_features)}, but got {output.shape}"

        logger.info("Enhanced attention test passed successfully")
    except Exception as e:
        logger.error(f"Enhanced attention test failed with error: {str(e)}")
        logger.exception("Traceback for the error:")
        raise

if __name__ == "__main__":
    test_enhanced_attention()
