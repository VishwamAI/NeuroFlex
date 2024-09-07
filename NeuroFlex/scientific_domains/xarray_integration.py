import xarray as xr
import numpy as np

class XarrayIntegration:
    def __init__(self):
        self.datasets = {}

    def create_dataset(self, name, data, coords):
        """
        Create an xarray Dataset.

        Args:
            name (str): Name of the dataset.
            data (dict): Dictionary of data variables.
            coords (dict): Dictionary of coordinates.

        Returns:
            xarray.Dataset: The created dataset.
        """
        dataset = xr.Dataset(data_vars=data, coords=coords)
        self.datasets[name] = dataset
        return dataset

    def apply_operation(self, dataset_name, operation):
        """
        Apply an operation to a dataset.

        Args:
            dataset_name (str): Name of the dataset to operate on.
            operation (str): Name of the operation to apply (e.g., 'mean', 'max', 'min').

        Returns:
            The result of the operation.
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found.")

        dataset = self.datasets[dataset_name]
        if hasattr(dataset, operation):
            return getattr(dataset, operation)()
        else:
            raise ValueError(f"Operation '{operation}' not supported.")

    def merge_datasets(self, dataset_names):
        """
        Merge multiple datasets.

        Args:
            dataset_names (list): List of dataset names to merge.

        Returns:
            xarray.Dataset: The merged dataset.
        """
        datasets_to_merge = [self.datasets[name] for name in dataset_names if name in self.datasets]
        if not datasets_to_merge:
            raise ValueError("No valid datasets to merge.")
        return xr.merge(datasets_to_merge)

    def save_dataset(self, dataset_name, file_path):
        """
        Save a dataset to a NetCDF file.

        Args:
            dataset_name (str): Name of the dataset to save.
            file_path (str): Path to save the NetCDF file.
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found.")

        self.datasets[dataset_name].to_netcdf(file_path)
        print(f"Dataset '{dataset_name}' saved to {file_path}")

    # Additional methods can be added here as needed
