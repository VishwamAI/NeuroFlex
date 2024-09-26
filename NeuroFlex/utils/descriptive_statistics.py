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

import numpy as np
from typing import Dict, Union, Optional

def calculate_descriptive_statistics(data: np.ndarray, axis: Optional[int] = None) -> Dict[str, Union[float, np.ndarray]]:
    """
    Calculate descriptive statistics for the input data.

    Args:
    data (np.ndarray): Input data array
    axis (int, optional): Axis along which to compute the statistics.
                          If None, compute over the entire array. Defaults to None.

    Returns:
    Dict[str, Union[float, np.ndarray]]: Dictionary containing the following statistics:
        - mean: Mean of the data
        - median: Median of the data
        - variance: Variance of the data
        - std_dev: Standard deviation of the data
    """
    # Check if the input is empty
    if data.size == 0:
        raise ValueError("Input data is empty")

    # Calculate statistics
    mean = np.mean(data, axis=axis)
    median = np.median(data, axis=axis)
    variance = np.var(data, axis=axis, ddof=1)  # Use ddof=1 for sample variance
    std_dev = np.std(data, axis=axis, ddof=1)  # Use ddof=1 for sample standard deviation

    return {
        "mean": mean,
        "median": median,
        "variance": variance,
        "std_dev": std_dev
    }

def preprocess_data(data: np.ndarray) -> np.ndarray:
    """
    Preprocess the input data by replacing NaN and Inf values with 0.

    Args:
    data (np.ndarray): Input data array

    Returns:
    np.ndarray: Preprocessed data array with the same shape as input
    """
    # Create a mask for finite values
    finite_mask = np.isfinite(data)

    # Create a copy of the data and replace non-finite values with 0
    preprocessed_data = np.where(finite_mask, data, 0)

    return preprocessed_data

def analyze_bci_data(data: np.ndarray, axis: Optional[int] = None) -> Dict[str, Union[float, np.ndarray]]:
    """
    Analyze BCI data by preprocessing and calculating descriptive statistics.

    Args:
    data (np.ndarray): Input BCI data array
    axis (int, optional): Axis along which to compute the statistics.
                          If None, compute over the entire array. Defaults to None.

    Returns:
    Dict[str, Union[float, np.ndarray]]: Dictionary containing descriptive statistics
    """
    if axis is None:
        # Preprocess the entire data array
        cleaned_data = preprocess_data(data)
    else:
        # Preprocess along the specified axis
        cleaned_data = np.apply_along_axis(preprocess_data, axis, data)

    # Calculate and return descriptive statistics
    return calculate_descriptive_statistics(cleaned_data, axis)

# Example usage:
# bci_data = np.random.rand(1000, 64)  # Simulated BCI data: 1000 time points, 64 channels
# statistics = analyze_bci_data(bci_data, axis=0)  # Analyze along the time axis
# print(statistics)
