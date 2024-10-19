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
from flax.core import FrozenDict, freeze, unfreeze
from jax import random
from collections.abc import Callable

class GWTModel(nn.Module):
    num_processes: int
    workspace_size: int

    def setup(self):
        self.weights = self.param('weights', nn.initializers.uniform(), (self.num_processes,))
        self.specialized_processes = [nn.Dense(self.workspace_size, name=f'specialized_process_{i}') for i in range(self.num_processes)]
        self.global_workspace = nn.Dense(self.workspace_size, name='global_workspace')
        self.consciousness_layer = nn.Dense(self.workspace_size, name='consciousness_layer')
        self.bias_mitigation_layer = nn.Dense(self.workspace_size, name='bias_mitigation_layer')

    @nn.compact
    def __call__(self, inputs):
        # Process inputs
        inputs = jnp.atleast_2d(inputs)  # Ensure inputs are at least 2d

        specialized_outputs = [process(inputs) for process in self.specialized_processes]
        integrated_workspace = self.integrate_workspace(specialized_outputs, self.global_workspace, self.weights)
        broadcasted_workspace = self.broadcast_workspace(integrated_workspace, specialized_outputs)

        # Apply GWT formula
        gwt_output = jnp.sum(jnp.sum(jnp.stack(specialized_outputs) * self.weights[:, jnp.newaxis], axis=0))

        return freeze({
            'params': self.variables['params'],
            'broadcasted_workspace': broadcasted_workspace,
            'integrated_workspace': integrated_workspace,
            'gwt_output': jnp.array([gwt_output])  # Ensure gwt_output has shape (1,)
        })

    def integrate_workspace(self, specialized_outputs, global_workspace, weights):
        """
        Integrate information from specialized processes into the global workspace.
        """
        weighted_sum = jnp.sum(jnp.stack(specialized_outputs) * weights[:, jnp.newaxis], axis=0)
        integrated = global_workspace(weighted_sum)
        # Ensure the integrated output has shape (1, workspace_size)
        integrated = integrated.mean(axis=0, keepdims=True)  # Average across processes and keep dims
        return integrated

    def update_weights(self, new_weights):
        """
        Update the weights for each specialized process.
        """
        if new_weights.shape != (self.num_processes,):
            raise ValueError("Number of weights must match number of processes")
        normalized_weights = new_weights / jnp.sum(new_weights)
        new_variables = unfreeze(self.variables)
        new_variables['params']['weights'] = normalized_weights
        return freeze(new_variables)

    def broadcast_workspace(self, integrated_workspace, specialized_outputs):
        """
        Broadcast the contents of the global workspace to all specialized processes.
        """
        broadcasted = [jnp.broadcast_to(integrated_workspace[i], output.shape) for i, output in enumerate(specialized_outputs)]
        return broadcasted

    @property
    def current_weights(self):
        return self.variables['params']['weights']

# Example usage:
if __name__ == "__main__":
    key = random.PRNGKey(0)
    model = GWTModel(num_processes=5, workspace_size=100)
    x = random.normal(key, (1, 100))
    variables = model.init(key, x)
    y = model.apply(variables, x)
    print(y)

# Example usage:
if __name__ == "__main__":
    import numpy as np

    # Initialize the model
    model = GWTModel(num_processes=5, workspace_size=100)

    # Generate some dummy input data
    inputs = jnp.array(np.random.randn(1, 100))

    # Initialize the model parameters
    key = jax.random.PRNGKey(0)
    params = model.init(key, inputs)

    # Run the model
    output = model.apply(params, inputs)
    print("Broadcasted workspace shape:", [bw.shape for bw in output['broadcasted_workspace']])
    print("Integrated workspace shape:", output['integrated_workspace'].shape)
    print("GWT formula output shape:", output['gwt_output'].shape)
