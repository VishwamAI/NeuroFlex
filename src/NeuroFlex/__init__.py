# Import main components
from .advanced_thinking import NeuroFlex, data_augmentation, create_train_state, select_action, adversarial_training
from .machinelearning import NeuroFlexClassifier
from .array_libraries import ArrayLibraries
from .wml_ce_integration import WMLCEIntegration
from .art_integration import ARTIntegration
from .lale_integration import LaleIntegration
from .aif360_integration import AIF360Integration
# from .ffdl_integration import FfDLIntegration  # Commented out due to missing setup
from .neunets_integration import NeuNetSIntegration
from NeuroFlex.modules.jax_module import JAXModel, train_jax_model
from NeuroFlex.modules.tensorflow import TensorFlowModel, train_tf_model
from NeuroFlex.modules.pytorch import PyTorchModel, train_pytorch_model
from .xarray_integration import XarrayIntegration
from .bioinformatics_integration import BioinformaticsIntegration
from .scikit_bio_integration import ScikitBioIntegration
from .ete_integration import ETEIntegration
from .alphafold_integration import AlphaFoldIntegration
from .detectron2_integration import Detectron2Integration
from .vision_transformer import VisionTransformer

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
    'WMLCEIntegration',
    'ARTIntegration',
    'LaleIntegration',
    'AIF360Integration',
    # 'FfDLIntegration',  # Commented out due to missing setup
    'NeuNetSIntegration',
    'XarrayIntegration',
    'BioinformaticsIntegration',
    'ScikitBioIntegration',
    'ETEIntegration',
    'AlphaFoldIntegration',
    'Detectron2Integration',
    'VisionTransformer',
]

# Check JAX and Flax versions
print(f"JAX version: {jax.__version__}")
print(f"Flax version: {flax.__version__}")
