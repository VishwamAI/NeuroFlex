import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple, Dict, Any
from .nlp_integration import NLPIntegration


class TextToImageGenerator(nn.Module):
    features: Tuple[int, ...]
    image_size: Tuple[int, int, int]
    text_embedding_size: int

    def setup(self):
        self.nlp_integration = NLPIntegration()
        self.text_encoder = nn.Dense(self.text_embedding_size)
        self.image_generator = nn.Sequential(
            [nn.Dense(feat) for feat in self.features]
            + [nn.Dense(jnp.prod(self.image_size))]
        )

    def __call__(self, text: str, train: bool = False):
        text_embedding = self.nlp_integration.encode_text(text)
        encoded_text = self.text_encoder(text_embedding)
        generated_image = self.image_generator(encoded_text)
        return generated_image.reshape((-1,) + self.image_size)

    def generate(self, text: str):
        return self(text, train=False)


def create_text_to_image_generator(
    features: Tuple[int, ...],
    image_size: Tuple[int, int, int],
    text_embedding_size: int,
):
    return TextToImageGenerator(
        features=features,
        image_size=image_size,
        text_embedding_size=text_embedding_size,
    )
