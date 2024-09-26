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

import jax.numpy as jnp
import matplotlib.pyplot as plt
from typing import List, Optional
class Visualization:
    def __init__(self):
        self.plt = plt

    def plot_line(self, x: jnp.ndarray, y: jnp.ndarray, title: str, xlabel: str, ylabel: str):
        """
        Create a line plot.

        Args:
            x (jnp.ndarray): X-axis data
            y (jnp.ndarray): Y-axis data
            title (str): Plot title
            xlabel (str): X-axis label
            ylabel (str): Y-axis label
        """
        self.plt.figure(figsize=(10, 6))
        self.plt.plot(x, y)
        self.plt.title(title)
        self.plt.xlabel(xlabel)
        self.plt.ylabel(ylabel)
        self.plt.show()

    def plot_scatter(self, x: jnp.ndarray, y: jnp.ndarray, title: str, xlabel: str, ylabel: str):
        """
        Create a scatter plot.

        Args:
            x (jnp.ndarray): X-axis data
            y (jnp.ndarray): Y-axis data
            title (str): Plot title
            xlabel (str): X-axis label
            ylabel (str): Y-axis label
        """
        self.plt.figure(figsize=(10, 6))
        self.plt.scatter(x, y)
        self.plt.title(title)
        self.plt.xlabel(xlabel)
        self.plt.ylabel(ylabel)
        self.plt.show()

    def plot_histogram(self, data: jnp.ndarray, bins: int, title: str, xlabel: str, ylabel: str):
        """
        Create a histogram.

        Args:
            data (jnp.ndarray): Input data
            bins (int): Number of bins
            title (str): Plot title
            xlabel (str): X-axis label
            ylabel (str): Y-axis label
        """
        self.plt.figure(figsize=(10, 6))
        self.plt.hist(data, bins=bins)
        self.plt.title(title)
        self.plt.xlabel(xlabel)
        self.plt.ylabel(ylabel)
        self.plt.show()

    def plot_heatmap(self, data: jnp.ndarray, title: str, xlabel: str, ylabel: str):
        """
        Create a heatmap.

        Args:
            data (jnp.ndarray): 2D array of data
            title (str): Plot title
            xlabel (str): X-axis label
            ylabel (str): Y-axis label
        """
        self.plt.figure(figsize=(10, 8))
        self.plt.imshow(data, cmap='viridis', aspect='auto')
        self.plt.colorbar()
        self.plt.title(title)
        self.plt.xlabel(xlabel)
        self.plt.ylabel(ylabel)
        self.plt.show()

    def plot_multiple_lines(self, x: jnp.ndarray, y_list: List[jnp.ndarray], labels: List[str],
                            title: str, xlabel: str, ylabel: str):
        """
        Create a plot with multiple lines.

        Args:
            x (jnp.ndarray): X-axis data
            y_list (List[jnp.ndarray]): List of Y-axis data arrays
            labels (List[str]): List of labels for each line
            title (str): Plot title
            xlabel (str): X-axis label
            ylabel (str): Y-axis label
        """
        self.plt.figure(figsize=(12, 6))
        for y, label in zip(y_list, labels):
            self.plt.plot(x, y, label=label)
        self.plt.title(title)
        self.plt.xlabel(xlabel)
        self.plt.ylabel(ylabel)
        self.plt.legend()
        self.plt.show()

    def save_plot(self, filename: str):
        """
        Save the current plot to a file.

        Args:
            filename (str): Name of the file to save the plot
        """
        self.plt.savefig(filename)
        self.plt.close()
