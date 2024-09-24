# NeuroFlex Testing Instructions

## Table of Contents
1. [Environment Setup](#environment-setup)
2. [Running Tests](#running-tests)
3. [Core Components Tests](#core-components-tests)
4. [AlphaFold Integration Tests](#alphafold-integration-tests)
5. [Cognitive Architectures Tests](#cognitive-architectures-tests)
6. [Expected Outcomes](#expected-outcomes)
7. [Troubleshooting](#troubleshooting)

## Environment Setup

Before running the tests, ensure you have set up your environment correctly:

1. Clone the NeuroFlex repository:
   ```bash
   git clone https://github.com/VishwamAI/NeuroFlex.git
   cd NeuroFlex
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv neuroflex-env
   source neuroflex-env/bin/activate  # On Windows, use: neuroflex-env\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Verify the installation of key dependencies:
   ```bash
   python -c "import alphafold, jax, haiku, openmm; print('All dependencies installed successfully')"
   ```

## Running Tests

NeuroFlex uses pytest for running tests. To run all tests:

```bash
pytest tests/
```

To run specific test files:

```bash
pytest tests/test_core_components.py
pytest tests/scientific_domains/test_alphafold_integration.py
pytest tests/cognitive_architectures/test_cognitive_architectures.py
```

## Core Components Tests

The core components tests are located in `tests/test_core_components.py`. These tests cover:

- NeuroFlex initialization
- Core model setup (PyTorch and TensorFlow backends)
- Quantum model setup
- Ethical framework
- Explainable AI
- BCI processor
- Consciousness simulation
- Math solver
- Edge optimization

Key test cases include:

- Initialization of all NeuroFlex components
- Backend-specific model creation (PyTorch/TensorFlow)
- Ethical evaluation of actions
- Explanation of model predictions
- BCI data processing
- Consciousness simulation
- Mathematical problem-solving
- Model optimization for edge devices

## AlphaFold Integration Tests

The AlphaFold integration tests are located in `tests/scientific_domains/test_alphafold_integration.py`. These tests cover:

- Version compatibility checks for AlphaFold, JAX, Haiku, and OpenMM
- AlphaFold model setup with OpenMM support
- Protein structure prediction and refinement process
- OpenMM simulation setup and execution

Key test cases include:

- Verifying correct versions of dependencies (AlphaFold 2.0.0, JAX 0.3.25, Haiku 0.0.9, OpenMM 7.7.0)
- Setting up the AlphaFold model with OpenMM support
- Preparing features for protein structure prediction
- Predicting protein structure using AlphaFold
- Refining the predicted structure using OpenMM molecular dynamics simulation
- Retrieving pLDDT scores and predicted aligned errors
- Testing fallback strategies for incompatible versions

### Usage Examples

To use the AlphaFold integration in your NeuroFlex project:

```python
from NeuroFlex.scientific_domains.alphafold_integration import AlphaFoldIntegration

# Initialize AlphaFold integration
af_integration = AlphaFoldIntegration()

# Set up the model
af_integration.setup_model()

# Prepare features for a protein sequence
sequence = "MKFLILLFNILCLFPVLAADNHGVGPQGASGVDPITFDINSNQTGPAFLTAVEMAGVKYLQVQHGSNVNIHRLVEGNVVIWENASTPLYTGAIVTNNDGPYMAYVEVLGDPNLQFFIKSGDAWVTLNTTFTDVATLLNTAETPSGTVGVGAFAAEQPGQNVAPQTFSTPEGAVTFPNFNGSWTSVNWTFTVTLPDTNGVVTSAPTIKSSGPSTGLLVGVKGDKPLPELVVGAVILSGINLNNQNVSAVAKTQATGNAVGFQVKQGDVVIATYMPMAPGVFVQDHGVVMTSNATVNTIDLDPNNIVFVNNDHNNSQNWQPVSMKPFSTPEGAVLFKNFNGSWTSVNWTFTVTLPDTNGVVTSAPTIKSSGPSTGLLVGVKGDKPLPELVVGAVILSGINLNNQNVSAVAKTQATGNAVGFQVKQGDVVIATYMPMAPGVFVQDHGVVMTSNATVNTIDLDPNNIVFVNNDHNNSQNWQPVSM"

af_integration.prepare_features(sequence)

# Predict protein structure
predicted_structure = af_integration.predict_structure()

# Get pLDDT scores
plddt_scores = af_integration.get_plddt_scores()

# Get predicted aligned error
predicted_aligned_error = af_integration.get_predicted_aligned_error()
```

This example demonstrates how to initialize the AlphaFold integration, set up the model, prepare features for a protein sequence, predict the structure, and retrieve additional information such as pLDDT scores and predicted aligned errors.

## Cognitive Architectures Tests

The cognitive architectures tests are located in `tests/cognitive_architectures/test_cognitive_architectures.py`. These tests cover:

- Basic cognitive architecture functionality
- Consciousness creation and simulation
- Feedback mechanism
- Extended cognitive architecture
- Working memory

Key test cases include:

- Cognitive architecture update with multi-modal inputs
- Consciousness state and feedback generation
- Consciousness simulation with working memory
- Extended cognitive model with BCI integration
- Working memory operations and updates

## Expected Outcomes

When running the tests, you should expect:

1. All tests in `test_core_components.py` to pass successfully.
2. All tests in `test_alphafold_integration.py` to pass, with possible warnings about version compatibility.
3. Some tests in `test_cognitive_architectures.py` are currently skipped due to failures and need updates. Specifically:
   - `test_consciousness_simulation`
   - `test_extended_cognitive_architecture`
   - `test_working_memory`

A successful test run should show passing tests for all core components, AlphaFold integration, and the non-skipped cognitive architecture tests.

## Troubleshooting

If you encounter any issues while running the tests:

1. Ensure all dependencies are correctly installed and at the correct versions:
   - AlphaFold: 2.0.0
   - JAX: 0.3.25
   - Haiku: 0.0.9
   - OpenMM: 7.7.0
2. Check that you're using a supported Python version (3.9, 3.10, 3.11, or 3.12).
3. Verify that your environment variables are set correctly.
4. For skipped tests or version compatibility warnings, refer to the latest documentation or GitHub issues for known problems and potential workarounds.
5. If you encounter OpenMM-related issues, ensure that your system supports CUDA for GPU acceleration. If not, the tests will fall back to CPU, which may be slower.

If problems persist, please open an issue on the [NeuroFlex GitHub repository](https://github.com/VishwamAI/NeuroFlex/issues) with details about the error, your environment, and the versions of key dependencies.
