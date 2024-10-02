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

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import List, Dict

class IITModel(nn.Module):
    """
    Integrated Information Theory (IIT) Model

    This class implements the basic concepts of Integrated Information Theory,
    focusing on measuring and analyzing integrated information across cognitive components.
    """
    num_components: int

    def setup(self):
        self.state_space = self.param('state_space', nn.initializers.zeros, (self.num_components, 2))
        self.connectivity_matrix = self.param('connectivity_matrix', nn.initializers.zeros, (self.num_components, self.num_components))

    def __call__(self, inputs):
        return self.calculate_phi()

    def set_state(self, state: List[int]):
        """
        Set the current state of the system.

        Args:
            state (List[int]): Binary state of each component (0 or 1).
        """
        if len(state) != self.num_components:
            raise ValueError("State length must match number of components")
        self.state_space = self.state_space.at[:, 0].set(jnp.array(state))

    def set_connectivity(self, connectivity: List[List[float]]):
        """
        Set the connectivity matrix between components.

        Args:
            connectivity (List[List[float]]): 2D list representing connection strengths.
        """
        if len(connectivity) != self.num_components or len(connectivity[0]) != self.num_components:
            raise ValueError("Connectivity matrix dimensions must match number of components")
        self.connectivity_matrix = jnp.array(connectivity)

    def calculate_phi(self) -> float:
        """
        Calculate the integrated information (Phi) of the system.

        Returns:
            float: The calculated Phi value.
        """
        return self.calculate_integrated_information()

    def calculate_integrated_information(self) -> float:
        """
        Calculate the integrated information (Phi) of the system using the IIT formula:
        Φ = I(S) - sum(I(S_i))

        Returns:
            float: The calculated Phi value.
        """
        # Calculate I(S) - information of the whole system
        I_S = self._calculate_system_information()

        # Calculate sum(I(S_i)) - sum of information of individual components
        I_Si_sum = jnp.sum(jax.vmap(self._calculate_component_information)(jnp.arange(self.num_components)))

        # Calculate Phi
        phi = I_S - I_Si_sum

        return jnp.maximum(0, phi)  # Phi should not be negative

    def _calculate_system_information(self) -> float:
        # Simplified calculation of system information
        return jnp.log2(jnp.linalg.det(self.connectivity_matrix + jnp.eye(self.num_components)))

    def _calculate_component_information(self, i: int) -> float:
        # Simplified calculation of individual component information
        return jnp.log2(1 + jnp.abs(self.connectivity_matrix[i, i]))

    def analyze_information_flow(self) -> Dict[str, float]:
        """
        Analyze the information flow between components.

        Returns:
            Dict[str, float]: A dictionary of metrics describing information flow.
        """
        # Simplified analysis
        # In a full implementation, this would involve detailed analysis of
        # information transfer and causal relationships
        total_flow = jnp.sum(self.connectivity_matrix)
        max_flow = jnp.max(self.connectivity_matrix)
        min_flow = jnp.min(self.connectivity_matrix[self.connectivity_matrix > 0])

        return {
            "total_flow": float(total_flow),
            "max_flow": float(max_flow),
            "min_flow": float(min_flow)
        }

    def simulate_step(self):
        """
        Simulate one step of the system's evolution.
        """
        # Simple state update based on connectivity
        new_state = jnp.dot(self.connectivity_matrix, self.state_space[:, 0])
        self.state_space = self.state_space.at[:, 1].set((new_state > 0.5).astype(int))
        self.state_space = jnp.roll(self.state_space, -1, axis=1)

    def get_current_state(self) -> List[int]:
        """
        Get the current state of the system.

        Returns:
            List[int]: The current binary state of each component.
        """
        return self.state_space[:, 0].tolist()

# Example usage:
if __name__ == "__main__":
    # Initialize the model
    model = IITModel(num_components=4)

    # Initialize parameters
    key = jax.random.PRNGKey(0)
    params = model.init(key, None)

    # Set initial state and connectivity
    model.apply(params, method=model.set_state, state=[1, 0, 1, 0])
    model.apply(params, method=model.set_connectivity, connectivity=[
        [0, 0.5, 0.2, 0],
        [0.1, 0, 0.3, 0.4],
        [0.2, 0.3, 0, 0.1],
        [0, 0.1, 0.2, 0]
    ])

    # Calculate Phi
    phi = model.apply(params, None)
    print(f"Calculated Phi: {phi}")

    # Analyze information flow
    info_flow = model.apply(params, method=model.analyze_information_flow)
    print(f"Information Flow: {info_flow}")

    # Simulate a step
    model.apply(params, method=model.simulate_step)
    current_state = model.apply(params, method=model.get_current_state)
    print(f"Current State: {current_state}")

# Example usage
if __name__ == "__main__":
    iit_model = IITModel(num_components=4)
    iit_model.set_state([1, 0, 1, 0])
    iit_model.set_connectivity([
        [0, 0.5, 0.2, 0],
        [0.1, 0, 0.3, 0.4],
        [0.2, 0.3, 0, 0.1],
        [0, 0.1, 0.2, 0]
    ])

    print(f"Initial State: {iit_model.get_current_state()}")
    print(f"Phi: {iit_model.calculate_phi()}")
    print(f"Information Flow: {iit_model.analyze_information_flow()}")

    iit_model.simulate_step()
    print(f"State after simulation: {iit_model.get_current_state()}")
