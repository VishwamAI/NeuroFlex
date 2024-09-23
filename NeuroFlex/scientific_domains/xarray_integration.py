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

    def load_dataset(self, file_path, dataset_name=None):
        """
        Load a dataset from a NetCDF file and register it in the datasets dictionary.

        Args:
            file_path (str): Path to the NetCDF file to load.
            dataset_name (str, optional): Name to assign to the loaded dataset.
                If not provided, the filename (without extension) will be used.

        Returns:
            xarray.Dataset: The loaded dataset.

        Raises:
            IOError: If there's an error loading the file.
        """
        try:
            dataset = xr.open_dataset(file_path)
            if dataset_name is None:
                dataset_name = file_path.split('/')[-1].split('.')[0]
            self.datasets[dataset_name] = dataset
            print(f"Dataset loaded from {file_path} and registered as '{dataset_name}'")
            return dataset
        except Exception as e:
            raise IOError(f"Error loading dataset from {file_path}: {str(e)}")

    # Additional methods can be added here as needed
