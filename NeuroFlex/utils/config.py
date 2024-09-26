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

import os

class Config:
    DEFAULT_CONFIG = {
        'CORE_MODEL_FEATURES': [64, 32, 10],
        'USE_CNN': True,
        'USE_RNN': True,
        'USE_GAN': False,
        'FAIRNESS_CONSTRAINT': 0.1,
        'USE_QUANTUM': False,
        'USE_ALPHAFOLD': False,
        'USE_BCI': False,
        'USE_CONSCIOUSNESS_SIM': False,
        'USE_EDGE_OPTIMIZATION': False,
        'BACKEND': 'pytorch',
        'INPUT_DIM': 784,
        'HIDDEN_LAYERS': [64, 32],
        'OUTPUT_DIM': 10,
        'INPUT_SHAPE': (28, 28, 1),
        'QUANTUM_N_QUBITS': 4,
        'QUANTUM_N_LAYERS': 2,
        'BCI_SAMPLING_RATE': 250,
        'BCI_NUM_CHANNELS': 32,
        'CONSCIOUSNESS_SIM_FEATURES': [64, 32],
    }

    @classmethod
    def get_config(cls):
        config = cls.DEFAULT_CONFIG.copy()
        for key, value in cls.DEFAULT_CONFIG.items():
            env_value = os.environ.get(f'NEUROFLEX_{key}')
            if env_value is not None:
                if isinstance(value, bool):
                    config[key] = env_value.lower() in ('true', '1', 'yes')
                elif isinstance(value, int):
                    config[key] = int(env_value)
                elif isinstance(value, float):
                    config[key] = float(env_value)
                elif isinstance(value, list):
                    config[key] = [int(x) for x in env_value.split(',')]
                else:
                    config[key] = env_value
        return config
