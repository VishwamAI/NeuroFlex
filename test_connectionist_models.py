import unittest
import numpy as np
from connectionist_models_module import ConnectionistModelsModule, configure_connectionist_models

class TestConnectionistModels(unittest.TestCase):
    def setUp(self):
        self.config = configure_connectionist_models()
        self.cm_module = ConnectionistModelsModule(self.config)

    def test_initialization(self):
        self.assertIsInstance(self.cm_module, ConnectionistModelsModule)
        self.assertEqual(len(self.cm_module.layers), len(self.config['layer_sizes']) - 1)

    def test_activation_function(self):
        x = np.array([-1, 0, 1])
        sigmoid = self.cm_module._get_activation_function()
        np.testing.assert_array_almost_equal(sigmoid(x), [0.26894142, 0.5, 0.73105858])

    def test_process(self):
        input_data = np.random.rand(64)
        output = self.cm_module.process(input_data)
        self.assertEqual(output.shape, (16,))
        self.assertTrue(np.all((output >= 0) & (output <= 1)))

    def test_train(self):
        input_data = np.random.rand(64)
        target_output = np.random.rand(16)
        initial_output = self.cm_module.process(input_data)
        self.cm_module.train(input_data, target_output, epochs=100)
        final_output = self.cm_module.process(input_data)
        self.assertFalse(np.array_equal(initial_output, final_output))

    def test_integrate_with_standalone_model(self):
        input_data = [1.0] * 64
        result = self.cm_module.integrate_with_standalone_model(input_data)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 16)

if __name__ == '__main__':
    unittest.main()
