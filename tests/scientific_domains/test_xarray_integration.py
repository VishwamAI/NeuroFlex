import unittest
import xarray as xr
import numpy as np
from unittest.mock import patch, MagicMock
from NeuroFlex.scientific_domains.xarray_integration import XarrayIntegration


class TestXarrayIntegration(unittest.TestCase):
    def setUp(self):
        self.xarray_integration = XarrayIntegration()

    def test_create_dataset(self):
        name = "test_dataset"
        data = {"temperature": (["x", "y"], [[20, 25], [30, 35]])}
        coords = {"x": [0, 1], "y": [0, 1]}

        dataset = self.xarray_integration.create_dataset(name, data, coords)

        self.assertIsInstance(dataset, xr.Dataset)
        self.assertIn(name, self.xarray_integration.datasets)
        np.testing.assert_array_equal(
            dataset.temperature.values, np.array([[20, 25], [30, 35]])
        )

    def test_apply_operation(self):
        name = "test_dataset"
        data = {"temperature": (["x", "y"], [[20, 25], [30, 35]])}
        coords = {"x": [0, 1], "y": [0, 1]}
        self.xarray_integration.create_dataset(name, data, coords)

        result = self.xarray_integration.apply_operation(name, "mean")
        self.assertEqual(result, 27.5)

        with self.assertRaises(ValueError):
            self.xarray_integration.apply_operation("non_existent_dataset", "mean")

        with self.assertRaises(ValueError):
            self.xarray_integration.apply_operation(name, "unsupported_operation")

    def test_merge_datasets(self):
        data1 = {"temperature": (["x", "y"], [[20, 25], [30, 35]])}
        coords1 = {"x": [0, 1], "y": [0, 1]}
        self.xarray_integration.create_dataset("dataset1", data1, coords1)

        data2 = {"humidity": (["x", "y"], [[50, 55], [60, 65]])}
        coords2 = {"x": [0, 1], "y": [0, 1]}
        self.xarray_integration.create_dataset("dataset2", data2, coords2)

        merged_dataset = self.xarray_integration.merge_datasets(
            ["dataset1", "dataset2"]
        )

        self.assertIsInstance(merged_dataset, xr.Dataset)
        self.assertIn("temperature", merged_dataset.data_vars)
        self.assertIn("humidity", merged_dataset.data_vars)

        with self.assertRaises(ValueError):
            self.xarray_integration.merge_datasets(["non_existent_dataset"])

    @patch("xarray.Dataset.to_netcdf")
    def test_save_dataset(self, mock_to_netcdf):
        name = "test_dataset"
        data = {"temperature": (["x", "y"], [[20, 25], [30, 35]])}
        coords = {"x": [0, 1], "y": [0, 1]}
        self.xarray_integration.create_dataset(name, data, coords)

        file_path = "/path/to/save/dataset.nc"
        self.xarray_integration.save_dataset(name, file_path)

        mock_to_netcdf.assert_called_once_with(file_path)

        with self.assertRaises(ValueError):
            self.xarray_integration.save_dataset("non_existent_dataset", file_path)


if __name__ == "__main__":
    unittest.main()
