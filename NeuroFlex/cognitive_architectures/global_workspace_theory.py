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
Global Workspace Theory (GWT) Model

This module implements a basic representation of the Global Workspace Theory,
which proposes that consciousness arises from a global workspace that integrates
and broadcasts information from various specialized cognitive processes.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.core.frozen_dict import FrozenDict
from jax import random
from collections.abc import Callable

class GWTModel(nn.Module):
    num_processes: int
    workspace_size: int

    def setup(self):
        self.specialized_processes = [nn.Dense(self.workspace_size, kernel_init=nn.initializers.normal(stddev=0.01)) for _ in range(self.num_processes)]
        self.global_workspace = nn.Dense(self.workspace_size, kernel_init=nn.initializers.normal(stddev=0.01))
        self.weights = self.param('weights', nn.initializers.uniform(), (self.num_processes,))
        self.consciousness_layer = nn.Dense(self.workspace_size, kernel_init=nn.initializers.normal(stddev=0.01))
        self.bias_mitigation_layer = nn.Dense(self.workspace_size, kernel_init=nn.initializers.normal(stddev=0.01))

    def __call__(self, inputs):
        # Process inputs without assuming specific attributes
        if callable(inputs):
            return inputs(self)
        inputs = jnp.atleast_2d(inputs)  # Ensure inputs are at least 2D
        specialized_outputs = [process(inputs) for process in self.specialized_processes]
        integrated_workspace = self.integrate_workspace(specialized_outputs)
        broadcasted_workspace = self.broadcast_workspace(integrated_workspace, specialized_outputs)
        return broadcasted_workspace, integrated_workspace

    def integrate_workspace(self, specialized_outputs):
        """
        Integrate information from specialized processes into the global workspace.
        """
        weights = self.variables['params']['weights']
        weighted_sum = jnp.sum(jnp.stack(specialized_outputs) * weights[:, jnp.newaxis], axis=0)
        integrated = self.global_workspace(weighted_sum)
        print(f"Integrated shape: {integrated.shape}, Specialized output shape: {specialized_outputs[0].shape}")
        print(f"Weights shape: {weights.shape}, Weighted sum shape: {weighted_sum.shape}")
        # Ensure the integrated output has shape (1, workspace_size)
        integrated = integrated.mean(axis=0, keepdims=True)  # Average across processes and keep dims
        print(f"Final integrated shape: {integrated.shape}")
        return integrated

    def broadcast_workspace(self, integrated_workspace, specialized_outputs):
        """
        Broadcast the contents of the global workspace to all specialized processes.
        """
        print("Integrated workspace shape:", integrated_workspace.shape)
        print("Specialized outputs shapes:", [output.shape for output in specialized_outputs])
        broadcasted = [jnp.broadcast_to(integrated_workspace[i], output.shape) for i, output in enumerate(specialized_outputs)]
        return broadcasted

    def apply_gwt_formula(self, input_stimulus):
        """
        Apply the GWT formula: G(x) = sum(w_i * f(x_i))

        Args:
            input_stimulus (jax.numpy.array): Input stimulus to the model.

        Returns:
            jax.numpy.array: The result of applying the GWT formula.
        """
        specialized_outputs = [process(input_stimulus) for process in self.specialized_processes]
        weights = self.variables['params']['weights']
        return jnp.sum(jnp.stack(specialized_outputs) * weights[:, jnp.newaxis], axis=0)

    def update_weights(self, new_weights):
        """
        Update the weights for each specialized process.

        Args:
            new_weights (jax.numpy.array or callable): New weights for the specialized processes or a function to update them.
        """
        current_weights = self.variables['params']['weights']
        if callable(new_weights):
            updated_weights = new_weights(current_weights)
        else:
            updated_weights = new_weights

        if isinstance(updated_weights, jnp.ndarray):
            if updated_weights.shape != (self.num_processes,):
                raise ValueError("Number of weights must match number of processes")
            normalized_weights = updated_weights / jnp.sum(updated_weights)  # Normalize weights
        else:
            raise ValueError("Updated weights must be a JAX numpy array")

        # Return a new FrozenDict with updated weights
        return FrozenDict({'params': {'weights': normalized_weights}})

    @property
    def current_weights(self):
        return self.variables['params']['weights']

    @property
    def current_weights(self):
        return self.variables['params']['weights']

# Example usage:
if __name__ == "__main__":
    import numpy as np

    # Initialize the model
    model = GWTModel(num_processes=5, workspace_size=100)

    # Generate some dummy input data
    inputs = jnp.array(np.random.randn(1, 100))

    # Initialize parameters
    key = jax.random.PRNGKey(0)
    params = model.init(key, inputs)

    # Run the model
    broadcasted_workspace, integrated_workspace = model.apply(params, inputs)
    print("Broadcasted workspace shape:", [bw.shape for bw in broadcasted_workspace])
    print("Integrated workspace shape:", integrated_workspace.shape)

    # Apply GWT formula
    gwt_output = model.apply(params, inputs, method=model.apply_gwt_formula)
    print("GWT formula output shape:", gwt_output.shape)
