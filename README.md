# NeuroFlex: Advanced Neural Network Framework

[![NeuroFlex CI/CD](https://github.com/VishwamAI/NeuroFlex/actions/workflows/ci_cd.yml/badge.svg)](https://github.com/VishwamAI/NeuroFlex/actions/workflows/ci_cd.yml)
[![PyPI version](https://badge.fury.io/py/neuroflex.svg)](https://badge.fury.io/py/neuroflex)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/VishwamAI/NeuroFlex/blob/main/LICENSE)
[![GitHub release](https://img.shields.io/github/release/VishwamAI/NeuroFlex.svg)](https://GitHub.com/VishwamAI/NeuroFlex/releases/)
[![GitHub stars](https://img.shields.io/github/stars/VishwamAI/NeuroFlex.svg)](https://GitHub.com/VishwamAI/NeuroFlex/stargazers/)
[![Dependencies](https://img.shields.io/librariesio/release/pypi/neuroflex)](https://libraries.io/pypi/neuroflex)
[![GitHub issues](https://img.shields.io/github/issues/VishwamAI/NeuroFlex.svg)](https://GitHub.com/VishwamAI/NeuroFlex/issues/)


NeuroFlex is a cutting-edge neural network framework built on JAX, Flax, and TensorFlow, designed to address key challenges in modern machine learning: interpretability, generalization, robustness, and fairness. This project integrates state-of-the-art techniques and methodologies for creating more transparent, reliable, and ethical AI systems.

![NeuroFlex Architecture](neuroflex-architecture-svg.svg)

## Key Features

- **Advanced Neural Architectures**: Support for CNN, RNN, LSTM, GAN, and Spiking Neural Networks.
- **Multi-Backend Integration**: Seamless integration with JAX, Flax, and TensorFlow.
- **2D and 3D Convolutions**: Efficient implementation of both 2D and 3D convolutional layers.
- **Quantum Computing**: Quantum Neural Network module for next-generation AI.
- **Reinforcement Learning**: Advanced capabilities with enhanced self-curing algorithms.
- **Brain-Computer Interface (BCI)**: Cutting-edge integration for neurotechnology applications.
- **Ethical AI**: Fairness constraints and bias mitigation techniques.
- **Robustness**: Improved adversarial training, interpretability tools (SHAP), and adaptive learning rate adjustment.
- **Bioinformatics**: AlphaFold integration for protein structure prediction and drug discovery.
- **Generative AI**: Creative problem-solving and content generation.
- **Natural Language Processing**: Sentence piece integration and advanced NLP tasks.
- **Neuromorphic Computing**: Energy-efficient spiking neural networks.
- **Self-Healing**: Advanced diagnostic and healing processes for improved model performance and stability.
- **Consciousness Simulation**: Integration of **Integrated Information Theory (IIT)** and **Global Workspace Theory (GWT)** for advanced consciousness modeling.
  
## Latest Updates (v0.1.3)

- **Consciousness Simulation**: Achieved integration of **Integrated Information Theory (IIT)** and **Global Workspace Theory (GWT)** for cognitive and consciousness modeling.
- TensorFlow Integration: Added support for TensorFlow backend alongside JAX/Flax.
- 2D and 3D Convolutions: Implemented efficient convolutional layers for both 2D and 3D inputs.
- Enhanced Compatibility: Improved interoperability between JAX/Flax and TensorFlow.
- Comprehensive Testing: Developed and passed pytest tests for all components, including TensorFlow integration.
- Enhanced self-curing mechanism with adaptive learning rate adjustment.
- Improved model robustness against gradient explosions and local minima.
- Advanced diagnostic checks for model performance and training issues.
- Whisper API integration for advanced speech recognition.
- Enhanced AlphaFold integration for protein structure prediction.
- Support for Python 3.9, 3.10, 3.11, 3.12.
- Resolved dependency issues and improved stability.

## Installation

To install NeuroFlex, please follow these steps:

1. Ensure you have Python 3.9 or later installed.
2. It's recommended to use a virtual environment:
   ```bash
   python -m venv neuroflex-env
   source neuroflex-env/bin/activate  # On Windows use `neuroflex-env\Scripts\activate`
   ```
3. Install NeuroFlex using pip:
   ```bash
   pip install neuroflex
   ```
4. If you encounter any issues, ensure that your pip is up to date:
   ```bash
   pip install --upgrade pip
   ```

## Quick Start Guide

1. **Import NeuroFlex**

```python
from NeuroFlex.core_neural_networks.advanced_nn import NeuroFlexNN, train_model, create_neuroflex_nn
from NeuroFlex.bci_integration import BCIIntegration
```

2. **Define Your Model**

```python
model = create_neuroflex_nn(
    features=[128, 64, 32, 10],  # Updated feature sizes for better performance
    input_shape=(1, 28, 28, 1),
    output_shape=(1, 10),
    use_cnn=True,
    conv_dim=2,
    backend='tensorflow'  # or 'jax' for JAX/Flax backend
)
```

3. **Integrate BCI Components**

```python
bci_integration = BCIIntegration(model)
bci_integration.setup_bci()
```

4. **Train Your Model**

```python
trained_state, trained_model = train_model(
    model, train_data, val_data,
    num_epochs=20,  # Increased epochs for improved training
    batch_size=32,
    learning_rate=1e-3
)
```

5. **Make Predictions**

```python
predictions = trained_model(test_data)
```

6. **Evaluate Model Performance**

```python
accuracy = evaluate_model(trained_model, test_data, test_labels)
print(f"Model Accuracy: {accuracy:.2f}%")
```

## Advanced Usage

### TensorFlow Convolutions

```python
from neuroflex import TensorFlowConvolutions

tf_conv = TensorFlowConvolutions(
    features=(32, 64, 128),
    input_shape=(1, 28, 28, 1),
    conv_dim=2  # Use 3 for 3D convolutions
)
model = tf_conv.create_model()
```

### JAX/Flax and TensorFlow Compatibility

```python
from neuroflex.utils import convert_array

jax_array = jax.random.uniform(jax.random.PRNGKey(0), shape=(1, 28, 28, 1))
tf_tensor = convert_array(jax_array, 'tensorflow')
```

### Adversarial Training

```python
from neuroflex import adversarial_training

perturbed_data = adversarial_training(model, params, input_data, epsilon=0.1, step_size=0.01)
```

## Testing

We have developed comprehensive pytest tests for all components, including the TensorFlow integration. To run the tests:

```bash
pytest tests/
```

Key test files:
- `test_neuroflex_nn.py`: Tests for the core NeuroFlexNN functionality.
- `test_tensorflow_convolutions.py`: Tests for TensorFlow convolution implementations.

## Environment Setup

NeuroFlex supports multiple operating systems:
- Ubuntu
- Windows
- macOS

To set up your environment:

1. Clone the repository:
   ```bash
   git clone https://github.com/neuroflex/neuroflex.git
   ```
2. Create a virtual environment:
   ```bash
   python -m venv neuroflex-env
   ```
3. Activate the environment:
   - Ubuntu/macOS: `source neuroflex-env/bin/activate`
   - Windows: `neuroflex-env\Scripts\activate`
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Documentation and Project Structure

The NeuroFlex project is organized into several key directories and files, each serving a specific purpose:

- **NeuroFlex.py**: Integrates various AI and neural network components, providing a unified interface for core neural networks, quantum neural networks, ethical frameworks, explainable AI, BCI integration, consciousness simulation, protein structure prediction, and more.
- **Prompt_Agent**: Contains agentic behavior models and related functionalities.
- **Transformers**: Includes advanced transformer models for text processing and classification.
- **Advanced Models**: Contains advanced math solving, time series analysis, and multi-modal learning models.
- **AI Ethics**: Implements ethical frameworks and guidelines for AI, including explainable AI and self-curing algorithms.
- **BCI Integration**: Facilitates integration with brain-computer interfaces, providing tools for BCI data processing and neuroscience model integration.
- **Cognitive Architectures**: Implements advanced cognitive architectures with self-healing mechanisms and adaptive algorithms.
- **Config**: Contains configuration files and constants for the project.
- **Core Neural Networks**: Provides core neural network models and functionalities.
- **Edge AI**: Focuses on edge AI optimization and deployment.
- **Generative Models**: Includes models for generative AI and creative problem-solving.
- **Main.py**: Demonstrates various components of the NeuroFlex framework.
- **Model Monitoring**: Contains tools for monitoring and analyzing model performance and health.
- **NeuroFlex Integration**: Integrates various components of the NeuroFlex system.
- **Threat Detection**: Provides methods for detecting and analyzing threats using anomaly detection, deep learning, and pattern recognition.
- **Utils**: Offers utility functions for data handling, preprocessing, and analysis.

For detailed documentation, including API references and tutorials, visit our [official documentation](https://neuroflex.readthedocs.io).

## Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md) for more information on how to get started.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use NeuroFlex in your research, please cite:

```bibtex
@software{neuroflex2024,
  author = {kasinadhsarma},
  title = {NeuroFlex: Advanced Neural Network Framework},
  year = {2024},
  url = {https://github.com/VishwamAI/NeuroFlex}
}
```

## Contact

For questions, feedback, or support, please open an issue on our [GitHub repository](https://github.com/VishwamAI/NeuroFlex/issues) or contact us at [gmail](mailto:kasinadhsarma@gmail.com).

---

### **Changes**:
- Added **Consciousness Simulation** section under Key Features and Latest Updates to reflect the integration of **IIT** and **GWT** theories.
- Updated the version to **v0.1.3**.
- General improvements and formatting to ensure clarity and consistency with the latest features.
