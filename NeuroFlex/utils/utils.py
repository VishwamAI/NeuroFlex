import numpy as np
import os
import json
from typing import List, Dict, Any

def load_data(file_path: str) -> np.ndarray:
    """
    Load data from a file (supports .npy and .csv formats).

    Args:
    file_path (str): Path to the data file

    Returns:
    np.ndarray: Loaded data
    """
    if file_path.endswith('.npy'):
        return np.load(file_path)
    elif file_path.endswith('.csv'):
        return np.genfromtxt(file_path, delimiter=',')
    else:
        raise ValueError("Unsupported file format. Use .npy or .csv")

def save_data(data: np.ndarray, file_path: str) -> None:
    """
    Save data to a file (supports .npy and .csv formats).

    Args:
    data (np.ndarray): Data to be saved
    file_path (str): Path to save the data
    """
    if file_path.endswith('.npy'):
        np.save(file_path, data)
    elif file_path.endswith('.csv'):
        np.savetxt(file_path, data, delimiter=',')
    else:
        raise ValueError("Unsupported file format. Use .npy or .csv")

def normalize_data(data: np.ndarray) -> np.ndarray:
    """
    Normalize data to have zero mean and unit variance.

    Args:
    data (np.ndarray): Input data

    Returns:
    np.ndarray: Normalized data
    """
    return (data - np.mean(data)) / np.std(data)

def create_directory(directory: str) -> None:
    """
    Create a directory if it doesn't exist.

    Args:
    directory (str): Path of the directory to be created
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_json(file_path: str) -> Dict[str, Any]:
    """
    Load JSON data from a file.

    Args:
    file_path (str): Path to the JSON file

    Returns:
    Dict[str, Any]: Loaded JSON data
    """
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json(data: Dict[str, Any], file_path: str) -> None:
    """
    Save data to a JSON file.

    Args:
    data (Dict[str, Any]): Data to be saved
    file_path (str): Path to save the JSON file
    """
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def flatten_list(nested_list: List[Any]) -> List[Any]:
    """
    Flatten a nested list.

    Args:
    nested_list (List[Any]): A list that may contain nested lists

    Returns:
    List[Any]: Flattened list
    """
    flattened = []
    for item in nested_list:
        if isinstance(item, list):
            flattened.extend(flatten_list(item))
        else:
            flattened.append(item)
    return flattened

def tokenize_text(text: str) -> List[str]:
    """
    Tokenize a given text into words.

    Args:
    text (str): Input text to be tokenized

    Returns:
    List[str]: List of tokens
    """
    # This is a simple tokenization. For more advanced tokenization,
    # consider using libraries like NLTK or spaCy
    return text.lower().split()

def get_activation_function(activation_name: str):
    """
    Get the activation function based on the given name.

    Args:
    activation_name (str): Name of the activation function

    Returns:
    Callable: JAX/Flax compatible activation function
    """
    import jax.nn as jnn
    activations = {
        'relu': jnn.relu,
        'sigmoid': jnn.sigmoid,
        'tanh': jnn.tanh,
        'leaky_relu': jnn.leaky_relu,
    }
    return activations.get(activation_name.lower(), jnn.relu)

# TODO: Add more utility functions as needed for the NeuroFlex project
# TODO: Consider adding functions for data preprocessing specific to neural network inputs
# TODO: Implement error handling and logging in these utility functions
