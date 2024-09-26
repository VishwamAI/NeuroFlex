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

# Import necessary modules from NeuroFlex
from neuroflex import NeuroFlexNN

# Environment setup
import torch
import pytest

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize NeuroFlex model for image generation
neuroflex_model = NeuroFlexNN(
    input_dim=768,  # Assuming BERT-like embedding size for text input
    hidden_dims=[512, 256, 128],
    output_dim=3 * 256 * 256,  # 3 channels, 256x256 image size
)
neuroflex_model.to(device)

# Image generation logic
def generate_image(prompt):
    # Simple tokenization (replace with more sophisticated method if needed)
    tokenized_prompt = torch.tensor([ord(c) for c in prompt]).unsqueeze(0)

    # Generate image using NeuroFlex
    with torch.no_grad():
        generated_image = neuroflex_model(tokenized_prompt)

    # Reshape and normalize the output
    generated_image = generated_image.view(3, 256, 256)
    generated_image = torch.clamp(generated_image, 0, 1)

    return generated_image

# Test cases for image generating agent
def test_integration():
    test_prompt = "A serene landscape with mountains and a lake"

    result = generate_image(test_prompt)

    # Test output type
    assert result is not None, "Image generation failed"
    assert isinstance(result, torch.Tensor), "Output should be a tensor"

    # Test output dimensions
    expected_shape = (3, 256, 256)  # RGB image of size 256x256
    assert result.shape == expected_shape, f"Expected shape {expected_shape}, got {result.shape}"

    # Test output range (normalized pixel values)
    assert torch.all(result >= 0) and torch.all(result <= 1), "Pixel values should be in range [0, 1]"

    # Test with different prompts
    diverse_prompts = [
        "A futuristic cityscape at night",
        "A close-up of a colorful butterfly on a flower",
        "An abstract painting with vibrant colors"
    ]
    for prompt in diverse_prompts:
        diverse_result = generate_image(prompt)
        assert diverse_result.shape == expected_shape, f"Shape mismatch for prompt: {prompt}"

    # Test that different prompts produce different outputs
    results = [generate_image(prompt) for prompt in diverse_prompts]
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            assert not torch.allclose(results[i], results[j]), "Different prompts should produce different outputs"

    # Test error handling (assuming generate_image raises ValueError for empty prompts)
    with pytest.raises(ValueError):
        generate_image("")

    print("All integration tests passed successfully!")

if __name__ == "__main__":
    test_integration()
