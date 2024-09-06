import unittest
from unittest.mock import patch, MagicMock, ANY
import numpy as np
import jax
import jax.numpy as jnp
from alphafold.common import protein
from alphafold.model import modules, config
from alphafold.data import pipeline
from Bio.SCOP import Scop
from NeuroFlex.scientific_domains.alphafold_integration import AlphaFoldIntegration, sequence_to_onehot, onehot_to_sequence

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

    @patch('alphafold.model.modules_multimer.AlphaFold')
    @patch('alphafold.data.tools.jackhmmer.Jackhmmer')
    @patch('alphafold.data.tools.hhblits.HHBlits')
    @patch('alphafold.model.config')
    @patch('ml_collections.ConfigDict')
    def test_setup_model(self, mock_config_dict, mock_config, mock_hhblits, mock_jackhmmer, mock_alphafold):
        # Create mocks for Jackhmmer and HHBlits
        mock_jackhmmer.return_value = MagicMock()
        mock_hhblits.return_value = MagicMock()

        # Create a mock for the ml_collections.ConfigDict
        mock_config_dict.return_value = MagicMock()

        # Create a mock model class that better represents the actual model's behavior
        class MockModel:
            def __init__(self, config):
                self.config = config

            def apply(self, params, **kwargs):
                return {"predicted_lddt": jnp.ones((10,))}

        # Set up the mock AlphaFold to return the MockModel
        mock_alphafold.return_value = MockModel(mock_config_dict.return_value)

        # Mock the config.model_config function
        mock_config.model_config.return_value = mock_config_dict.return_value

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
        mock_config.model_config.assert_called_once_with('model_1_ptm')

        # Check if the model is created correctly
        self.assertIsNotNone(self.alphafold.model)
        self.assertTrue(callable(self.alphafold.model))

        # Check if MSA runner and template searcher are initialized
        self.assertIsNotNone(self.alphafold.msa_runner)
        self.assertIsNotNone(self.alphafold.template_searcher)

        # Check if the config is updated correctly
        self.assertIsNotNone(self.alphafold.config)
        self.assertEqual(self.alphafold.config.max_recycling, 5)
        self.assertEqual(self.alphafold.config.model_name, 'model_1_ptm')
        self.assertEqual(self.alphafold.config.data.common.max_extra_msa, 1024)

        # Check if model_params are set correctly
        self.assertIsNotNone(self.alphafold.model_params)

        # Test the model with dummy input
        dummy_input = {
            'msa_feat': jnp.zeros((1, 1, 256, 49), dtype=jnp.int32),  # Changed from 'msa' to 'msa_feat' and adjusted shape
            'msa_mask': jnp.ones((1, 1, 256), dtype=jnp.float32),
            'seq_mask': jnp.ones((1, 256), dtype=jnp.float32),
            'aatype': jnp.zeros((1, 256), dtype=jnp.int32),
            'residue_index': jnp.arange(256)[None],
            'template_aatype': jnp.zeros((1, 1, 256), dtype=jnp.int32),
            'template_all_atom_masks': jnp.zeros((1, 1, 256, 37), dtype=jnp.float32),
            'template_all_atom_positions': jnp.zeros((1, 1, 256, 37, 3), dtype=jnp.float32),
            'is_distillation': jnp.array(0, dtype=jnp.int32),
        }

        result = self.alphafold.model(self.alphafold.model_params, jax.random.PRNGKey(0), self.alphafold.config, **dummy_input)
        self.assertIn('predicted_lddt', result)
        self.assertEqual(result['predicted_lddt'].shape, (10,))

    @patch('NeuroFlex.scientific_domains.alphafold_integration.AlphaFoldIntegration._run_msa')
    @patch('NeuroFlex.scientific_domains.alphafold_integration.AlphaFoldIntegration._search_templates')
    @patch('NeuroFlex.scientific_domains.alphafold_integration.pipeline')
    def test_prepare_features(self, mock_pipeline, mock_search_templates, mock_run_msa):
        sequence = "ACDEFGHIKLMNPQRSTVWY"
        mock_sequence_features = {'sequence': sequence}
        mock_msa = [("query", sequence)]
        mock_msa_features = {'msa': mock_msa}
        mock_template_features = {'template': 'mock_template'}

        mock_pipeline.make_sequence_features.return_value = mock_sequence_features
        mock_run_msa.return_value = mock_msa
        mock_pipeline.make_msa_features.return_value = mock_msa_features
        mock_search_templates.return_value = mock_template_features

        self.alphafold.prepare_features(sequence)

        self.assertEqual(self.alphafold.feature_dict, {**mock_sequence_features, **mock_msa_features, **mock_template_features})
        mock_pipeline.make_sequence_features.assert_called_once_with(sequence)
        mock_run_msa.assert_called_once_with(sequence)
        mock_pipeline.make_msa_features.assert_called_once_with(msas=[mock_msa])
        mock_search_templates.assert_called_once_with(sequence)

    @patch('NeuroFlex.scientific_domains.alphafold_integration.AlphaFoldIntegration.predict_structure')
    def test_predict_structure(self, mock_predict_structure):
        mock_protein = MagicMock(spec=protein.Protein)
        mock_predict_structure.return_value = mock_protein

        self.alphafold.model = MagicMock()
        self.alphafold.model_params = MagicMock()
        self.alphafold.feature_dict = MagicMock()
        self.alphafold.config = MagicMock()

        result = self.alphafold.predict_structure()

        self.assertEqual(result, mock_protein)
        mock_predict_structure.assert_called_once()

        # We don't need to assert the model call here since we're mocking predict_structure
        # Instead, we can check if the mocked attributes were accessed
        self.assertTrue(self.alphafold.model.called)
        self.assertTrue(self.alphafold.model_params.called)
        self.assertTrue(self.alphafold.feature_dict.called)
        self.assertTrue(self.alphafold.config.called)

    def test_sequence_to_onehot(self):
        sequence = "ACDEFGHIKLMNPQRSTVWY"
        onehot = sequence_to_onehot(sequence)
        self.assertEqual(onehot.shape, (20, 20))
        self.assertTrue(np.all(np.sum(onehot, axis=1) == 1))
        self.assertEqual(onehot.dtype, np.float32)  # Check for correct data type

    def test_onehot_to_sequence(self):
        sequence = "ACDEFGHIKLMNPQRSTVWY"
        onehot = sequence_to_onehot(sequence)
        recovered_sequence = onehot_to_sequence(onehot)
        self.assertEqual(sequence, recovered_sequence)

if __name__ == '__main__':
    unittest.main()
