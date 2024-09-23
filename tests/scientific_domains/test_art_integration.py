import unittest
import numpy as np
import tensorflow as tf

tf.compat.v1.disable_eager_execution()
import torch
import torch.nn as nn
from NeuroFlex.scientific_domains.art_integration import (
    ARTIntegration,
)


class TestARTIntegration(unittest.TestCase):
    def setUp(self):
        self.input_shape = (28, 28, 1)
        self.num_classes = 10
        self.keras_model = self.create_keras_model()
        self.pytorch_model = self.create_pytorch_model()
        self.art_integration_keras = ARTIntegration(self.keras_model, framework="keras")
        self.art_integration_pytorch = ARTIntegration(
            self.pytorch_model, framework="pytorch"
        )

    def create_keras_model(self):
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Flatten(input_shape=self.input_shape),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(self.num_classes, activation="softmax"),
            ]
        )
        model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )
        return model

    def create_pytorch_model(self):
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_classes),
            nn.Softmax(dim=1),
        )
        return model

    def test_set_model(self):
        # Test for Keras
        self.assertIsNotNone(self.art_integration_keras.art_classifier)
        self.assertEqual(self.art_integration_keras.framework, "keras")

        # Test for PyTorch
        self.assertIsNotNone(self.art_integration_pytorch.art_classifier)
        self.assertEqual(self.art_integration_pytorch.framework, "pytorch")

        # Test for unsupported framework
        with self.assertRaises(ValueError):
            ARTIntegration(self.keras_model, framework="unsupported")

    def test_generate_adversarial_examples(self):
        x = np.random.rand(10, *self.input_shape).astype(np.float32)

        # Test FGSM attack
        try:
            x_keras = tf.convert_to_tensor(x, dtype=tf.float32)
            adv_x_fgsm_keras = self.art_integration_keras.generate_adversarial_examples(
                x_keras, method="fgsm"
            )
            self.assertEqual(adv_x_fgsm_keras.shape, x.shape)
            self.assertIsInstance(adv_x_fgsm_keras, np.ndarray)
            self.assertEqual(adv_x_fgsm_keras.dtype, np.float32)
        except Exception as e:
            self.fail(f"FGSM attack for Keras failed: {str(e)}")

        try:
            x_pytorch = torch.from_numpy(x)
            adv_x_fgsm_pytorch = (
                self.art_integration_pytorch.generate_adversarial_examples(
                    x_pytorch, method="fgsm"
                )
            )
            self.assertEqual(adv_x_fgsm_pytorch.shape, x.shape)
            self.assertIsInstance(adv_x_fgsm_pytorch, np.ndarray)
        except Exception as e:
            self.fail(f"FGSM attack for PyTorch failed: {str(e)}")

        # Test PGD attack
        try:
            adv_x_pgd_keras = self.art_integration_keras.generate_adversarial_examples(
                x_keras, method="pgd"
            )
            self.assertEqual(adv_x_pgd_keras.shape, x.shape)
            self.assertIsInstance(adv_x_pgd_keras, np.ndarray)
            self.assertEqual(adv_x_pgd_keras.dtype, np.float32)
        except Exception as e:
            self.fail(f"PGD attack for Keras failed: {str(e)}")

        try:
            adv_x_pgd_pytorch = (
                self.art_integration_pytorch.generate_adversarial_examples(
                    x_pytorch, method="pgd"
                )
            )
            self.assertEqual(adv_x_pgd_pytorch.shape, x.shape)
            self.assertIsInstance(adv_x_pgd_pytorch, np.ndarray)
        except Exception as e:
            self.fail(f"PGD attack for PyTorch failed: {str(e)}")

        # Test unsupported attack method
        with self.assertRaises(ValueError):
            self.art_integration_keras.generate_adversarial_examples(
                x_keras, method="unsupported"
            )

    def test_apply_defense(self):
        x = np.random.rand(10, *self.input_shape).astype(np.float32)
        clip_values = (0, 1)  # Assuming input values are in the range [0, 1]

        # Test feature squeezing
        try:
            params = {"clip_values": clip_values, "bit_depth": 8}
            defended_x_fs = self.art_integration_keras.apply_defense(
                x, method="feature_squeezing", params=params
            )
            self.assertEqual(defended_x_fs.shape, x.shape)
            self.assertEqual(defended_x_fs.dtype, x.dtype)
        except ImportError:
            print(
                "Warning: Feature squeezing defense could not be imported. Skipping this test."
            )
        except Exception as e:
            self.fail(f"Feature squeezing defense failed: {str(e)}")

        # Test spatial smoothing
        try:
            params = {"clip_values": clip_values, "window_size": 3}
            defended_x_ss = self.art_integration_keras.apply_defense(
                x, method="spatial_smoothing", params=params
            )
            self.assertEqual(defended_x_ss.shape, x.shape)
            self.assertEqual(defended_x_ss.dtype, x.dtype)
        except ImportError:
            print(
                "Warning: Spatial smoothing defense could not be imported. Skipping this test."
            )
        except Exception as e:
            self.fail(f"Spatial smoothing defense failed: {str(e)}")

        # Test unsupported defense method
        with self.assertRaises(ValueError):
            self.art_integration_keras.apply_defense(x, method="unsupported")

        # Test PyTorch defense
        x_torch = torch.from_numpy(x)
        try:
            params = {"clip_values": clip_values, "bit_depth": 8}
            defended_x_torch = self.art_integration_pytorch.apply_defense(
                x_torch, method="feature_squeezing", params=params
            )
            self.assertEqual(defended_x_torch.shape, x.shape)
            self.assertEqual(defended_x_torch.dtype, x.dtype)
        except ImportError:
            print(
                "Warning: PyTorch feature squeezing defense could not be imported. Skipping this test."
            )
        except Exception as e:
            self.fail(f"PyTorch feature squeezing defense failed: {str(e)}")

    def test_adversarial_training(self):
        x = np.random.rand(100, *self.input_shape).astype(np.float32)
        y = np.eye(self.num_classes, dtype=np.float32)[
            np.random.randint(0, self.num_classes, 100)
        ]

        # Test adversarial training for Keras
        try:
            x_keras = tf.convert_to_tensor(x)
            y_keras = tf.convert_to_tensor(y)
            self.art_integration_keras.adversarial_training(
                x_keras, y_keras, nb_epochs=1
            )
        except Exception as e:
            self.fail(f"Adversarial training for Keras failed: {str(e)}")

        # Test adversarial training for PyTorch
        try:
            x_torch = torch.from_numpy(x)
            y_torch = torch.from_numpy(y)
            self.art_integration_pytorch.adversarial_training(
                x_torch, y_torch, nb_epochs=1
            )
        except Exception as e:
            self.fail(f"Adversarial training for PyTorch failed: {str(e)}")

    def test_evaluate_robustness(self):
        x = np.random.rand(100, *self.input_shape).astype(np.float32)
        y = np.eye(self.num_classes, dtype=np.float32)[
            np.random.randint(0, self.num_classes, 100)
        ]

        # Test robustness evaluation for Keras
        try:
            x_keras = tf.convert_to_tensor(x)
            y_keras = tf.convert_to_tensor(y)
            results_keras = self.art_integration_keras.evaluate_robustness(
                x_keras, y_keras
            )
            self.assertIn("fgsm", results_keras)
            self.assertIn("pgd", results_keras)
            self.assertIsInstance(results_keras["fgsm"], float)
            self.assertIsInstance(results_keras["pgd"], float)
        except Exception as e:
            self.fail(f"Robustness evaluation for Keras failed: {str(e)}")

        # Test robustness evaluation for PyTorch
        try:
            x_torch = torch.from_numpy(x)
            y_torch = torch.from_numpy(y)
            results_pytorch = self.art_integration_pytorch.evaluate_robustness(
                x_torch, y_torch
            )
            self.assertIn("fgsm", results_pytorch)
            self.assertIn("pgd", results_pytorch)
            self.assertIsInstance(results_pytorch["fgsm"], float)
            self.assertIsInstance(results_pytorch["pgd"], float)
        except Exception as e:
            self.fail(f"Robustness evaluation for PyTorch failed: {str(e)}")

        # Check if results are not None
        self.assertIsNotNone(results_keras)
        self.assertIsNotNone(results_pytorch)


if __name__ == "__main__":
    unittest.main()
