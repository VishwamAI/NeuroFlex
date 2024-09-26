# MIT License
# 
# Copyright (c) 2024 VishwamAI
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
NeuroFlex Generative AI Models Example

This script demonstrates the usage of generative AI models in NeuroFlex.
It includes initialization, configuration, and basic operations for TransformerModel, GenerativeAIModel, and GAN.
"""

from NeuroFlex.generative_models import TransformerModel, GenerativeAIModel, GAN
from NeuroFlex.utils import data_loader

def demonstrate_transformer_model():
    print("Demonstrating Transformer Model:")
    transformer = TransformerModel(vocab_size=10000, d_model=512, nhead=8, num_encoder_layers=6)
    input_sequence = data_loader.load_example_text_sequence()
    output = transformer.generate(input_sequence, max_length=50)
    print(f"Input sequence: {input_sequence[:30]}...")
    print(f"Generated output: {output[:50]}...")
    print()

def demonstrate_generative_ai_model():
    print("Demonstrating Generative AI Model:")
    gen_ai = GenerativeAIModel(input_dim=100, output_dim=784)  # For generating 28x28 images
    latent_vector = data_loader.generate_random_latent_vector(100)
    generated_image = gen_ai.generate(latent_vector)
    print(f"Latent vector shape: {latent_vector.shape}")
    print(f"Generated image shape: {generated_image.shape}")
    print()

def demonstrate_gan():
    print("Demonstrating GAN:")
    gan = GAN(latent_dim=100, img_shape=(28, 28, 1))
    real_images = data_loader.load_example_images(batch_size=32)
    gan.train(real_images, epochs=1)  # Train for 1 epoch as an example
    generated_images = gan.generate(num_samples=5)
    print(f"Number of generated images: {len(generated_images)}")
    print(f"Shape of each generated image: {generated_images[0].shape}")
    print()

def main():
    demonstrate_transformer_model()
    demonstrate_generative_ai_model()
    demonstrate_gan()

if __name__ == "__main__":
    main()
