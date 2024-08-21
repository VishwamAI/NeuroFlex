# Import main components
from .advanced_thinking import NeuroFlex, data_augmentation, create_train_state, select_action, adversarial_training
from .machinelearning import NeuroFlexClassifier
from .array_libraries import ArrayLibraries
from .art_integration import ARTIntegration
from .lale_integration import LaleIntegration
from .aif360_integration import AIF360Integration
from .neunets_integration import NeuNetSIntegration
from .jax_module import JAXModel, train_jax_model
from .tensorflow_module import TensorFlowModel, train_tf_model
from .pytorch_module import PyTorchModel, train_pytorch_model
from .xarray_integration import XarrayIntegration
from .bioinformatics_integration import BioinformaticsIntegration
from .scikit_bio_integration import ScikitBioIntegration
from .ete_integration import ETEIntegration
from .alphafold_integration import AlphaFoldIntegration
from .detectron2_integration import Detectron2Integration
from .vision_transformer import VisionTransformer
from .quantum_nn_module import QuantumNeuralNetwork
from .rl_module import RLAgent, RLEnvironment, train_rl_agent
from .advanced_nn import NeuroFlexNN
from .tokenisation import tokenize_text
from .correctgrammer import correct_grammar
from .tokenizer import Tokenizer
from .destroy_button import DestroyButton

# Ensure compatibility with latest JAX and Flax versions
import jax
import flax

# Define what should be available when importing the package
__all__ = [
    'NeuroFlex',
    'data_augmentation',
    'create_train_state',
    'select_action',
    'adversarial_training',
    'NeuroFlexClassifier',
    'PyTorchModel',
    'train_pytorch_model',
    'JAXModel',
    'train_jax_model',
    'TensorFlowModel',
    'train_tf_model',
    'ArrayLibraries',
    'ARTIntegration',
    'LaleIntegration',
    'AIF360Integration',
    'NeuNetSIntegration',
    'XarrayIntegration',
    'BioinformaticsIntegration',
    'ScikitBioIntegration',
    'ETEIntegration',
    'AlphaFoldIntegration',
    'Detectron2Integration',
    'VisionTransformer',
    'QuantumNeuralNetwork',
    'RLAgent',
    'RLEnvironment',
    'train_rl_agent',
    'NeuroFlexNN',
    'tokenize_text',
    'correct_grammar',
    'Tokenizer',
    'DestroyButton',
]

# Check JAX and Flax versions
print(f"JAX version: {jax.__version__}")
print(f"Flax version: {flax.__version__}")
