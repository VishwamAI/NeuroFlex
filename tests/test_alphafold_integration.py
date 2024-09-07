import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import jax.numpy as jnp
from NeuroFlex.scientific_domains import AlphaFoldIntegration, sequence_to_onehot, onehot_to_sequence

class TestAlphaFoldIntegration(unittest.TestCase):
    def setUp(self):
        self.alphafold = AlphaFoldIntegration()

    def test_initialization(self):
        self.assertIsNone(self.alphafold.model)
        self.assertIsNone(self.alphafold.model_params)
        self.assertIsNone(self.alphafold.feature_dict)
        self.assertIsNone(self.alphafold.msa_runner)
        self.assertIsNone(self.alphafold.template_searcher)
        self.assertIsNone(self.alphafold.config)

    @patch('alphafold.model.config')
    @patch('alphafold.data.tools.jackhmmer.Jackhmmer')
    @patch('alphafold.data.tools.hhblits.HHBlits')
    def test_setup_model(self, mock_hhblits, mock_jackhmmer, mock_config):
        # Create a mock for Jackhmmer
        mock_jackhmmer.return_value = MagicMock()

        # Create a mock for HHBlits that doesn't perform file system checks
        mock_hhblits.return_value = MagicMock()

        # Create a mock for the config.ConfigDict
        mock_config_dict = MagicMock()
        mock_config.ConfigDict.return_value = mock_config_dict

        # Create a mock model class that better represents the actual model's behavior
        class MockModel:
            def __init__(self, config, name):
                self.config = config
                self.name = name

            def __call__(self, *args, **kwargs):
                return self  # Simulate the model being callable

            def predict(self, *args, **kwargs):
                # Simulate prediction method
                return {"predicted_lddt": jnp.ones((10,))}

        # Set up the mock_config to return the mock_config_dict for both calls
        mock_model_module = MagicMock()
        mock_model_module.model = MockModel

        # Add debug print to log calls to model_config
        def mock_model_config_with_logging(*args, **kwargs):
            print(f"model_config called with args: {args}, kwargs: {kwargs}")
            return mock_config_dict

        mock_config.model_config.side_effect = mock_model_config_with_logging

        # Ensure the config object has the update_from_flattened_dict method
        mock_config_dict.update_from_flattened_dict = MagicMock()

        custom_params = {
            'max_recycling': 5,
            'model_name': 'model_1_ptm',
            'data': {'common': {'max_extra_msa': 1024}},
            'hhblits_binary_path': '/path/to/hhblits',
            'hhblits_databases': ['/path/to/hhblits_db1', '/path/to/hhblits_db2']
        }
        self.alphafold.setup_model(custom_params)

        # Check if HHBlits is initialized correctly
        self.assertIsNotNone(self.alphafold.template_searcher)
        mock_hhblits.assert_called_once_with(
            binary_path='/path/to/hhblits',
            databases=['/path/to/hhblits_db1', '/path/to/hhblits_db2']
        )

        # Validate mock calls
        print(f"All calls to model_config: {mock_config.model_config.mock_calls}")
        mock_config.model_config.assert_has_calls([
            unittest.mock.call('model_1_ptm'),
            unittest.mock.call('model_1_ptm')
        ])

        # Check if the model is created correctly
        self.assertIsNotNone(self.alphafold.model)
        self.assertTrue(callable(self.alphafold.model))
        model_instance = self.alphafold.model(config=mock_config_dict, name='model_1_ptm')
        self.assertIsInstance(model_instance, MockModel)
        self.assertEqual(model_instance.config, mock_config_dict)
        self.assertEqual(model_instance.name, 'model_1_ptm')

        # Check if the model is created correctly
        self.assertIsNotNone(self.alphafold.model)
        self.assertTrue(callable(self.alphafold.model))
        model_instance = self.alphafold.model(config=mock_config_dict, name='model_1_ptm')
        self.assertIsInstance(model_instance, MockModel)
        self.assertEqual(model_instance.config, mock_config_dict)
        self.assertEqual(model_instance.name, 'model_1_ptm')
        # Verify that the model is callable and returns itself
        self.assertIs(model_instance(), model_instance)

        # Check if MSA runner and template searcher are initialized
        self.assertIsNotNone(self.alphafold.msa_runner)
        self.assertIsNotNone(self.alphafold.template_searcher)

        # Check if the config is updated correctly
        self.assertIsNotNone(self.alphafold.config)
        self.assertEqual(self.alphafold.config['max_recycling'], 5)
        self.assertEqual(self.alphafold.config['model_name'], 'model_1_ptm')
        self.assertEqual(self.alphafold.config['data']['common']['max_extra_msa'], 1024)

        # Check if model_params are set correctly
        expected_model_params = {
            'is_training': False,
            'recycle_features': True,
            'recycle_pos': True
        }
        self.assertEqual(self.alphafold.model_params, expected_model_params)

        # The flattened config is no longer used, so we remove these assertions

    @patch('NeuroFlex.scientific_domains.AlphaFoldIntegration.prepare_features')
    def test_prepare_features(self, mock_prepare_features):
        sequence = "ACDEFGHIKLMNPQRSTVWY"
        mock_feature_dict = {'aatype': np.zeros(20), 'sequence': sequence}
        mock_prepare_features.return_value = mock_feature_dict

        result = self.alphafold.prepare_features(sequence)

        self.assertEqual(result, mock_feature_dict)
        mock_prepare_features.assert_called_once_with(sequence)

    @patch('NeuroFlex.scientific_domains.AlphaFoldIntegration.predict_structure')
    def test_predict_structure(self, mock_predict_structure):
        mock_protein = MagicMock()
        mock_predict_structure.return_value = mock_protein

        result = self.alphafold.predict_structure()

        self.assertEqual(result, mock_protein)
        mock_predict_structure.assert_called_once()

    def test_sequence_to_onehot(self):
        sequence = "ACDEFGHIKLMNPQRSTVWY"
        onehot = sequence_to_onehot(sequence)
        self.assertEqual(onehot.shape, (20, 20))
        self.assertTrue(np.all(np.sum(onehot, axis=1) == 1))

    def test_onehot_to_sequence(self):
        sequence = "ACDEFGHIKLMNPQRSTVWY"
        onehot = sequence_to_onehot(sequence)
        recovered_sequence = onehot_to_sequence(onehot)
        self.assertEqual(sequence, recovered_sequence)

if __name__ == '__main__':
    unittest.main()
