import unittest
import torch
import torch.nn as nn

from NeuroFlex.core_neural_networks import CNN

class TestCNN(unittest.TestCase):
    def setUp(self):
        self.input_channels = 1
        self.num_classes = 10

    def test_cnn_initialization(self):
        cnn = CNN(input_channels=self.input_channels, num_classes=self.num_classes)
        self.assertIsInstance(cnn, CNN)
        self.assertIsInstance(cnn.conv1, nn.Conv2d)
        self.assertIsInstance(cnn.conv2, nn.Conv2d)
        self.assertIsInstance(cnn.pool, nn.MaxPool2d)
        self.assertIsInstance(cnn.fc1, nn.Linear)
        self.assertIsInstance(cnn.fc2, nn.Linear)
        self.assertIsInstance(cnn.relu, nn.ReLU)

    def test_cnn_forward_pass(self):
        cnn = CNN(input_channels=self.input_channels, num_classes=self.num_classes)
        input_shape = (1, self.input_channels, 28, 28)
        x = torch.ones(input_shape)
        output = cnn(x)

        self.assertEqual(output.shape, (1, self.num_classes))

    def test_cnn_output_shape(self):
        cnn = CNN(input_channels=self.input_channels, num_classes=self.num_classes)
        input_shape = (32, self.input_channels, 28, 28)
        x = torch.ones(input_shape)
        output = cnn(x)

        self.assertEqual(output.shape, (32, self.num_classes))

    def test_error_handling(self):
        with self.assertRaises(ValueError):
            CNN(input_channels=0, num_classes=10)

        with self.assertRaises(ValueError):
            CNN(input_channels=3, num_classes=0)

        model = CNN(input_channels=1, num_classes=10)
        with self.assertRaises(RuntimeError):
            model(torch.randn(1, 3, 28, 28))  # Incorrect number of input channels

    def test_model_training(self):
        model = CNN(input_channels=1, num_classes=10)
        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()

        # Simulate a small training loop
        for _ in range(5):
            optimizer.zero_grad()
            output = model(torch.randn(32, 1, 28, 28))
            loss = criterion(output, torch.randint(0, 10, (32,)))
            loss.backward()
            optimizer.step()

        # Check if parameters have been updated
        for param in model.parameters():
            self.assertFalse(torch.allclose(param, torch.zeros_like(param)))

if __name__ == '__main__':
    unittest.main()
