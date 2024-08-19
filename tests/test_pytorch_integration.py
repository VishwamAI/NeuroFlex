import unittest
import torch
import numpy as np
from src.NeuroFlex.modules.pytorch import PyTorchModel, train_pytorch_model
from src.NeuroFlex.array_libraries import ArrayLibraries

class TestPyTorchIntegration(unittest.TestCase):
    def setUp(self):
        self.features = [10, 20, 15, 5]  # Example feature sizes
        self.model = PyTorchModel(self.features)
        self.X = np.random.rand(100, 10).astype(np.float32)  # 100 samples, 10 features
        self.y = np.random.randint(0, 5, 100)  # 100 labels, 5 classes

    def test_pytorch_model_structure(self):
        self.assertEqual(len(self.model.layers), len(self.features) * 2 - 3)
        self.assertIsInstance(self.model.layers[0], torch.nn.Linear)
        self.assertIsInstance(self.model.layers[-1], torch.nn.Linear)

    def test_train_pytorch_model(self):
        trained_model = train_pytorch_model(self.model, self.X, self.y, epochs=10)
        self.assertIsInstance(trained_model, PyTorchModel)

    def test_model_forward_pass(self):
        input_tensor = torch.FloatTensor(self.X)
        output = self.model(input_tensor)
        self.assertEqual(output.shape, (100, 5))

    def test_array_conversion(self):
        np_array = np.random.rand(5, 5)
        torch_tensor = ArrayLibraries.convert_numpy_to_pytorch(np_array)
        self.assertIsInstance(torch_tensor, torch.Tensor)
        np_array_converted = ArrayLibraries.convert_pytorch_to_numpy(torch_tensor)
        np.testing.assert_array_almost_equal(np_array, np_array_converted)

if __name__ == '__main__':
    unittest.main()
