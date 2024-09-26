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


# config.py

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    # General settings
    PROJECT_NAME = "NeuroFlex"
    DEBUG = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")
    
    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    
    # Neural Network settings
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
    LEARNING_RATE = float(os.getenv("LEARNING_RATE", "0.001"))
    EPOCHS = int(os.getenv("EPOCHS", "100"))
    
    # Reinforcement Learning settings
    GAMMA = float(os.getenv("GAMMA", "0.99"))
    EPSILON_START = float(os.getenv("EPSILON_START", "1.0"))
    EPSILON_END = float(os.getenv("EPSILON_END", "0.01"))
    EPSILON_DECAY = float(os.getenv("EPSILON_DECAY", "0.995"))
    
    # Quantum Neural Network settings
    N_QUBITS = int(os.getenv("N_QUBITS", "4"))
    N_LAYERS = int(os.getenv("N_LAYERS", "2"))
    
    # BCI Integration settings
    SAMPLING_RATE = int(os.getenv("SAMPLING_RATE", "250"))
    CHANNEL_NAMES = os.getenv("CHANNEL_NAMES", "Fp1,Fp2,C3,C4").split(",")
    
    # AI Ethics settings
    ETHICAL_THRESHOLD = float(os.getenv("ETHICAL_THRESHOLD", "0.8"))
    
    # Generative Model settings
    LATENT_DIM = int(os.getenv("LATENT_DIM", "100"))
    
    # Scientific Domain settings
    MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "1000"))
    CONVERGENCE_THRESHOLD = float(os.getenv("CONVERGENCE_THRESHOLD", "1e-6"))

    @classmethod
    def get_config(cls):
        return {key: value for key, value in cls.__dict__.items() if not key.startswith("__")}

# Example usage
if __name__ == "__main__":
    config = Config.get_config()
    print("NeuroFlex Configuration:")
    for key, value in config.items():
        print(f"{key}: {value}")

# TODO: Add more specific configurations for each module
# TODO: Implement configuration validation
# TODO: Add support for different environments (development, production, testing)
# TODO: Implement secure handling of sensitive information
