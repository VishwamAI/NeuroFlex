import xarray as xr
import numpy as np
from typing import Dict, Any, List
import ml_dtypes

class XarrayIntegration:
    def __init__(self):
        self.datasets: Dict[str, xr.Dataset] = {}

    def create_dataset(self, name: str, data: Dict[str, np.ndarray], coords: Dict[str, np.ndarray]) -> None:
        """
        Create an xarray Dataset and store it in the datasets dictionary.

        Args:
            name (str): Name of the dataset
            data (Dict[str, np.ndarray]): Dictionary of variable names and their corresponding data arrays
            coords (Dict[str, np.ndarray]): Dictionary of coordinate names and their corresponding arrays
        """
        self.datasets[name] = xr.Dataset(data_vars=data, coords=coords)

    def get_dataset(self, name: str) -> xr.Dataset:
        """
        Retrieve a dataset by name.

        Args:
            name (str): Name of the dataset

        Returns:
            xr.Dataset: The requested dataset
        """
        return self.datasets.get(name)

    def apply_operation(self, dataset_name: str, operation: str, dim: str = None) -> xr.DataArray:
        """
        Apply a common operation (mean, sum, max, min) to a dataset.

        Args:
            dataset_name (str): Name of the dataset
            operation (str): Operation to apply ('mean', 'sum', 'max', 'min')
            dim (str, optional): Dimension along which to apply the operation. If None, applies to all dimensions.

        Returns:
            xr.DataArray: Result of the operation
        """
        dataset = self.get_dataset(dataset_name)
        if dataset is None:
            raise ValueError(f"Dataset '{dataset_name}' not found")

        if operation == 'mean':
            return dataset.mean(dim=dim)
        elif operation == 'sum':
            return dataset.sum(dim=dim)
        elif operation == 'max':
            return dataset.max(dim=dim)
        elif operation == 'min':
            return dataset.min(dim=dim)
        else:
            raise ValueError(f"Unsupported operation: {operation}")

    def merge_datasets(self, dataset_names: List[str]) -> xr.Dataset:
        """
        Merge multiple datasets.

        Args:
            dataset_names (List[str]): List of dataset names to merge

        Returns:
            xr.Dataset: Merged dataset
        """
        datasets_to_merge = [self.get_dataset(name) for name in dataset_names]
        return xr.merge(datasets_to_merge)

    def save_dataset(self, dataset_name: str, file_path: str, format: str = 'netcdf') -> None:
        """
        Save a dataset to a file.

        Args:
            dataset_name (str): Name of the dataset to save
            file_path (str): Path to save the file
            format (str, optional): File format ('netcdf' or 'zarr'). Defaults to 'netcdf'.
        """
        dataset = self.get_dataset(dataset_name)
        if dataset is None:
            raise ValueError(f"Dataset '{dataset_name}' not found")

        if format == 'netcdf':
            dataset.to_netcdf(file_path)
        elif format == 'zarr':
            dataset.to_zarr(file_path)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def load_dataset(self, name: str, file_path: str, format: str = 'netcdf') -> None:
        """
        Load a dataset from a file.

        Args:
            name (str): Name to assign to the loaded dataset
            file_path (str): Path to the file to load
            format (str, optional): File format ('netcdf' or 'zarr'). Defaults to 'netcdf'.
        """
        if format == 'netcdf':
            self.datasets[name] = xr.open_dataset(file_path)
        elif format == 'zarr':
            self.datasets[name] = xr.open_zarr(file_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
