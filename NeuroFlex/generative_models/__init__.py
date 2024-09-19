"""
NeuroFlex Generative Models Module

This module provides generative models and tools for AI, including GANs, VAEs, and diffusion models.

Recent updates:
- Enhanced support for text-to-image generation
- Improved integration with NLP models
- Added support for latent diffusion techniques
"""

from .generative_ai import GAN
from .vae import VAE
from .nlp_integration import NLPIntegration
from .text_to_image import TextToImageGenerator
from .latent_diffusion import LatentDiffusionModel
from .cognitive_architecture import CognitiveArchitecture
from .ddim import DDIMSampler

__all__ = [
    'GAN',
    'VAE',
    'NLPIntegration',
    'TextToImageGenerator',
    'LatentDiffusionModel',
    'CognitiveArchitecture',
    'DDIMSampler',
    'get_generative_models_version',
    'SUPPORTED_GENERATIVE_MODELS',
    'generate_random_noise',
    'initialize_generative_models'
]

def get_generative_models_version():
    return "1.0.0"

SUPPORTED_GENERATIVE_MODELS = ['GAN', 'VAE', 'Diffusion']

def generate_random_noise(shape):
    import numpy as np
    return np.random.normal(0, 1, shape)

def initialize_generative_models():
    print("Initializing Generative Models...")
    # Add any necessary initialization code here

# Add any other Generative Models-specific utility functions or constants as needed
