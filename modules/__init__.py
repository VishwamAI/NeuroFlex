# Import main components from each module
from .jax_module import JAXModel, train_jax_model
from .pytorch import PyTorchModel, train_pytorch_model
from .tensorflow_module import TensorFlowModel, train_tf_model
from .scientific_domains.quantum_domains import QuantumDomains
from .advanced_thinking import NeuroFlexNN, create_train_state
from .rl_module import RLAgent, RLEnvironment, train_rl_agent

# Define what should be available when importing the package
__all__ = [
    'JAXModel',
    'train_jax_model',
    'PyTorchModel',
    'train_pytorch_model',
    'TensorFlowModel',
    'train_tf_model',
    'QuantumDomains',
    'NeuroFlexNN',
    'create_train_state',
    'RLAgent',
    'RLEnvironment',
    'train_rl_agent'
]
