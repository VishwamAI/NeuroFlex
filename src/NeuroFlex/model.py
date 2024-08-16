import NeuroFlex
import ddpm
from neuroflex import NeuroFlex, train_model
from neuroflex.jax_module import JAXModel
from neuroflex.tensorflow_module import TensorFlowModel
from neuroflex.pytorch_module import PyTorchModel
from neuroflex.quantum_module import QuantumModel

# Define your model
model = NeuroFlex(
    features=[64, 32, 10],
    use_cnn=True,
    use_rnn=True,
    use_gan=True,
    fairness_constraint=0.1,
    use_quantum=True,  # Enable quantum neural network
    backend='jax',  # Choose from 'jax', 'tensorflow', 'pytorch'
    jax_model=JAXModel,
    tensorflow_model=TensorFlowModel,
    pytorch_model=PyTorchModel,
    quantum_model=QuantumModel
)

# Train your model
trained_state, trained_model = train_model(
    model, train_data, val_data,
    num_epochs=10, batch_size=32, learning_rate=1e-3
)
