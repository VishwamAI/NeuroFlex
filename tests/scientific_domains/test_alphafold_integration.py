import unittest
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import sys
import ml_collections
from unittest.mock import patch, MagicMock
import openmm
import openmm.app as app
import openmm.unit as unit
sys.path.append('/home/ubuntu/NeuroFlex/neuroflex-env-3.8/lib/python3.8/site-packages')
from NeuroFlex.scientific_domains.alphafold_integration import (
    AlphaFoldIntegration, protein, check_version,
    ALPHAFOLD_COMPATIBLE, JAX_COMPATIBLE, HAIKU_COMPATIBLE, OPENMM_COMPATIBLE
)

class TestAlphaFoldIntegration(unittest.TestCase):
    def setUp(self):
        self.alphafold_integration = AlphaFoldIntegration()
        print("DEBUG: AlphaFoldIntegration instance created")
        print(f"DEBUG: AlphaFoldIntegration methods: {dir(self.alphafold_integration)}")
        print(f"DEBUG: run_alphaproteo exists: {'run_alphaproteo' in dir(self.alphafold_integration)}")
        print(f"DEBUG: run_alphamissense exists: {'run_alphamissense' in dir(self.alphafold_integration)}")
        self.alphafold_integration._validate_sequence = MagicMock()
        self.alphafold_integration._calculate_molecular_weight = MagicMock(return_value=10000.0)
        self.alphafold_integration._calculate_isoelectric_point = MagicMock(return_value=7.0)
        self.alphafold_integration._calculate_hydrophobicity = MagicMock(return_value=0.5)
        self.alphafold_integration._validate_alphamissense_input = MagicMock()
        self.alphafold_integration._calculate_severity_score = MagicMock(return_value=1)
        self.alphafold_integration._calculate_confidence_score = MagicMock(return_value=0.8)
        self.alphafold_integration._determine_severity = MagicMock(return_value="moderate")
        self.alphafold_integration._determine_functional_impact = MagicMock(return_value=(0.7, "possibly damaging"))
        print("DEBUG: All mock methods set up")

    def test_version_compatibility(self):
        with patch('NeuroFlex.scientific_domains.alphafold_integration.importlib.metadata.version') as mock_version:
            mock_version.side_effect = ['2.3.2', '0.4.31', '0.0.12', '8.1.1']
            self.assertTrue(check_version("alphafold", "2.3.2"))
            self.assertTrue(check_version("jax", "0.4.31"))
            self.assertTrue(check_version("haiku", "0.0.12"))
            self.assertTrue(check_version("openmm", "8.1.1"))

    @staticmethod
    def create_mock_config():
        class SimpleGlobalConfig:
            def __init__(self):
                self.subbatch_size = 4
                self.use_remat = False
                self.zero_init = True
                self.eval_dropout = False
                self._deterministic = False
                self.use_custom_jit = True  # Additional attribute for AlphaFold compatibility

            @property
            def deterministic(self):
                return self._deterministic

            @deterministic.setter
            def deterministic(self, value):
                self._deterministic = bool(value)

        class SimpleConfig(ml_collections.ConfigDict):
            def __init__(self):
                super().__init__()
                self.model = object()  # Simple object instead of MagicMock
                self.data = object()   # Simple object instead of MagicMock
                self._global_config = SimpleGlobalConfig()
                self.model_name = 'model_1'
                self.num_recycle = 3
                self.max_extra_msa = 1024  # Additional attribute for AlphaFold compatibility
                self.msa_cluster_size = 512  # Additional attribute for AlphaFold compatibility

            @property
            def global_config(self):
                return self._global_config

            @property
            def deterministic(self):
                return self.global_config.deterministic

        config = SimpleConfig()
        # Ensure deterministic is set to False for testing purposes
        config.global_config.deterministic = False
        print(f"DEBUG: deterministic value: {config.global_config.deterministic}")
        assert isinstance(config.global_config.deterministic, bool), "deterministic is not a boolean"
        return config

    @patch('NeuroFlex.scientific_domains.alphafold_integration.modules.AlphaFold')
    @patch('NeuroFlex.scientific_domains.alphafold_integration.jackhmmer.Jackhmmer')
    @patch('NeuroFlex.scientific_domains.alphafold_integration.hhblits.HHBlits')
    @patch('NeuroFlex.scientific_domains.alphafold_integration.hk.transform')
    def test_setup_model(self, mock_transform, mock_hhblits, mock_jackhmmer, mock_alphafold):
        # Set up mock objects
        mock_model = MagicMock()
        mock_alphafold.return_value = mock_model
        mock_transform.return_value.init.return_value = {'params': MagicMock()}
        mock_transform.return_value.apply.return_value = mock_model

        # Define model parameters
        model_params = {
            'jackhmmer_binary_path': '/mock/path/to/jackhmmer',
            'jackhmmer_database_path': '/mock/path/to/jackhmmer_db',
            'hhblits_binary_path': '/mock/path/to/hhblits',
            'hhblits_database_path': '/mock/path/to/hhblits_db',
            'model_name': 'model_1',
            'num_ensemble': 1,
            'num_recycle': 3,
            'random_seed': 42
        }

        # Test successful model setup
        self.alphafold_integration.setup_model(model_params)

        # Verify model components initialization
        self.assertIsNotNone(self.alphafold_integration.model)
        self.assertIsNotNone(self.alphafold_integration.model_params)
        self.assertIsNotNone(self.alphafold_integration.config)

        # Verify critical configuration values
        self.assertTrue(hasattr(self.alphafold_integration.config, 'model'))
        self.assertTrue(hasattr(self.alphafold_integration.config, 'data'))
        self.assertTrue(hasattr(self.alphafold_integration.config, 'global_config'))
        self.assertTrue(hasattr(self.alphafold_integration.config.global_config, 'deterministic'))

        # Verify global_config attributes
        self.assertEqual(self.alphafold_integration.config.global_config.subbatch_size, 4)
        self.assertFalse(self.alphafold_integration.config.global_config.use_remat)
        self.assertTrue(self.alphafold_integration.config.global_config.zero_init)
        self.assertFalse(self.alphafold_integration.config.global_config.eval_dropout)
        self.assertTrue(self.alphafold_integration.config.global_config.use_custom_jit)

        # Verify deterministic attribute
        self.assertIsInstance(self.alphafold_integration.config.global_config.deterministic, bool)
        self.assertFalse(self.alphafold_integration.config.global_config.deterministic)

        # Verify model configuration
        self.assertEqual(self.alphafold_integration.config.model_name, 'model_1')
        self.assertEqual(self.alphafold_integration.config.num_recycle, 3)

        # Verify AlphaFold model initialization
        mock_alphafold.assert_called_once()
        args, kwargs = mock_alphafold.call_args
        self.assertEqual(len(args), 2)
        self.assertIsInstance(args[0], object)  # model config
        self.assertIsInstance(args[1], object)  # data config

        # Verify MSA runner and template searcher initialization
        mock_jackhmmer.assert_called_once_with(binary_path='/mock/path/to/jackhmmer', database_path='/mock/path/to/jackhmmer_db')
        mock_hhblits.assert_called_once_with(binary_path='/mock/path/to/hhblits', databases=['/mock/path/to/hhblits_db'])

        # Verify model parameters
        for key, value in model_params.items():
            self.assertEqual(getattr(self.alphafold_integration.config, key, None), value)

        # Verify additional AlphaFold-specific configurations
        self.assertTrue(hasattr(self.alphafold_integration.config, 'data'))
        self.assertTrue(hasattr(self.alphafold_integration.config, 'model'))
        self.assertEqual(self.alphafold_integration.config.model.num_ensemble, 1)
        self.assertEqual(self.alphafold_integration.config.model.num_recycle, 3)

        # Verify the model is ready for predictions
        self.assertTrue(self.alphafold_integration.is_model_ready())

        # Test model with dummy input
        dummy_seq_length = 50
        dummy_input = {
            'aatype': jnp.zeros((dummy_seq_length,), dtype=jnp.int32),
            'residue_index': jnp.arange(dummy_seq_length),
            'seq_mask': jnp.ones((dummy_seq_length,), dtype=jnp.float32),
            'msa_feat': jnp.zeros((1, dummy_seq_length, 49), dtype=jnp.int32),
            'msa_mask': jnp.ones((1, dummy_seq_length), dtype=jnp.float32),
            'num_alignments': jnp.array([1], dtype=jnp.int32),
            'template_aatype': jnp.zeros((1, dummy_seq_length), dtype=jnp.int32),
            'template_all_atom_masks': jnp.zeros((1, dummy_seq_length, 37), dtype=jnp.float32),
            'template_all_atom_positions': jnp.zeros((1, dummy_seq_length, 37, 3), dtype=jnp.float32),
            'template_mask': jnp.zeros((1,), dtype=jnp.float32),
            'template_pseudo_beta': jnp.zeros((1, dummy_seq_length, 3), dtype=jnp.float32),
            'template_pseudo_beta_mask': jnp.zeros((1, dummy_seq_length), dtype=jnp.float32),
        }
        try:
            _ = self.alphafold_integration.model(self.alphafold_integration.model_params, jax.random.PRNGKey(0), self.alphafold_integration.config, **dummy_input)
        except Exception as e:
            self.fail(f"Model failed with dummy input: {str(e)}")

        # Verify that the model was called with the correct arguments
        mock_model.assert_called_once()
        model_call_args = mock_model.call_args
        self.assertIn('is_training', model_call_args.kwargs)
        self.assertFalse(model_call_args.kwargs['is_training'])
        self.assertIn('compute_loss', model_call_args.kwargs)
        self.assertFalse(model_call_args.kwargs['compute_loss'])

        # Test handling of 'aatype' derived from sequence
        sequence_input = dummy_input.copy()
        del sequence_input['aatype']
        sequence_input['sequence'] = "A" * dummy_seq_length
        try:
            _ = self.alphafold_integration.model(self.alphafold_integration.model_params, jax.random.PRNGKey(0), self.alphafold_integration.config, **sequence_input)
        except Exception as e:
            self.fail(f"Model failed with sequence input: {str(e)}")

        # Verify that the sequence was correctly converted to aatype
        self.assertIn('aatype', model_call_args.kwargs)
        np.testing.assert_array_equal(model_call_args.kwargs['aatype'], np.zeros(dummy_seq_length, dtype=np.int32))

        # Test error handling for missing 'aatype' and 'sequence'
        invalid_input = dummy_input.copy()
        del invalid_input['aatype']
        with self.assertRaises(ValueError) as cm:
            self.alphafold_integration.model(self.alphafold_integration.model_params, jax.random.PRNGKey(0), self.alphafold_integration.config, **invalid_input)
        self.assertIn("Either 'aatype' or 'sequence' must be provided in the input", str(cm.exception))

        # Test error handling for invalid 'aatype' shape
        invalid_input = dummy_input.copy()
        invalid_input['aatype'] = jnp.zeros((dummy_seq_length, 23), dtype=jnp.float32)  # Invalid shape
        with self.assertRaises(ValueError) as cm:
            self.alphafold_integration.model(self.alphafold_integration.model_params, jax.random.PRNGKey(0), self.alphafold_integration.config, **invalid_input)
        self.assertIn("'aatype' has incorrect shape", str(cm.exception))

        # Test error handling for invalid residue_index
        invalid_input = dummy_input.copy()
        invalid_input['residue_index'] = jnp.arange(dummy_seq_length - 1)  # Wrong length
        with self.assertRaises(ValueError) as cm:
            self.alphafold_integration.model(self.alphafold_integration.model_params, jax.random.PRNGKey(0), self.alphafold_integration.config, **invalid_input)
        self.assertIn("'residue_index' length mismatch", str(cm.exception))

        # Test fallback method for model creation
        mock_alphafold.side_effect = [AttributeError, MagicMock()]  # First call raises AttributeError, second call succeeds
        self.alphafold_integration.setup_model(model_params)
        self.assertEqual(mock_alphafold.call_count, 2)  # Ensure the fallback method was called

        # Test error handling for invalid model parameters
        with self.assertRaises(ValueError):
            self.alphafold_integration.setup_model({'invalid_param': 'value'})

        # Test handling of missing database paths
        incomplete_params = model_params.copy()
        del incomplete_params['jackhmmer_database_path']
        del incomplete_params['hhblits_database_path']
        with self.assertLogs(level='WARNING') as cm:
            self.alphafold_integration.setup_model(incomplete_params)
        self.assertIn("Jackhmmer database path not provided", cm.output[0])
        self.assertIn("HHBlits database path not provided", cm.output[1])

        # Test error handling for invalid sequence input
        invalid_sequence_input = sequence_input.copy()
        invalid_sequence_input['sequence'] = "X" * dummy_seq_length  # Invalid amino acid
        with self.assertRaises(ValueError) as cm:
            self.alphafold_integration.model(self.alphafold_integration.model_params, jax.random.PRNGKey(0), self.alphafold_integration.config, **invalid_sequence_input)
        self.assertIn("Invalid amino acid in sequence", str(cm.exception))

        # Test error handling for mismatched sequence and residue_index lengths
        mismatched_input = sequence_input.copy()
        mismatched_input['sequence'] = "A" * (dummy_seq_length - 1)  # One residue short
        with self.assertRaises(ValueError) as cm:
            self.alphafold_integration.model(self.alphafold_integration.model_params, jax.random.PRNGKey(0), self.alphafold_integration.config, **mismatched_input)
        self.assertIn("Sequence length does not match residue_index length", str(cm.exception))

    def test_prepare_features(self):
        # Mock the necessary methods
        self.alphafold_integration._run_msa = MagicMock(return_value=[('query', 'SEQUENCE')])
        self.alphafold_integration._search_templates = MagicMock(return_value={'template_features': 'dummy'})

        # Test with valid sequence
        valid_sequence = "MKFLKFSLLTAVLLSVVFAFSSCGD"
        self.alphafold_integration.prepare_features(valid_sequence)
        self.assertIsNotNone(self.alphafold_integration.feature_dict)
        self.assertIn('template_features', self.alphafold_integration.feature_dict)

        # Test with invalid sequence
        with self.assertRaises(ValueError):
            self.alphafold_integration.prepare_features("INVALID123")

        # Test with empty sequence
        with self.assertRaises(ValueError):
            self.alphafold_integration.prepare_features("")

        # Test error handling in _run_msa
        self.alphafold_integration._run_msa.side_effect = Exception("MSA error")
        with self.assertRaises(Exception):
            self.alphafold_integration.prepare_features(valid_sequence)

        # Test error handling in _search_templates
        self.alphafold_integration._run_msa = MagicMock(return_value=[('query', 'SEQUENCE')])
        self.alphafold_integration._search_templates.side_effect = Exception("Template search error")
        with self.assertRaises(Exception):
            self.alphafold_integration.prepare_features(valid_sequence)

    def test_is_model_ready(self):
        self.assertFalse(self.alphafold_integration.is_model_ready())
        self.alphafold_integration.model = MagicMock()
        self.alphafold_integration.model_params = MagicMock()
        self.assertTrue(self.alphafold_integration.is_model_ready())

    @patch('NeuroFlex.scientific_domains.alphafold_integration.pipeline')
    @patch('NeuroFlex.scientific_domains.alphafold_integration.SeqIO')
    def test_prepare_features(self, mock_seqio, mock_pipeline):
        valid_sequence = "MKFLKFSLLTAVLLSVVFAFSSCGD"
        mock_pipeline.make_sequence_features.return_value = {'seq_features': 'dummy_seq'}
        mock_pipeline.make_msa_features.return_value = {'msa_features': 'dummy_msa'}
        self.alphafold_integration._run_msa = MagicMock(return_value=[('query', valid_sequence)])
        self.alphafold_integration._search_templates = MagicMock(return_value={'template_features': 'dummy_template'})

        self.alphafold_integration.prepare_features(valid_sequence)

        self.assertIsNotNone(self.alphafold_integration.feature_dict)
        self.assertIn('seq_features', self.alphafold_integration.feature_dict)
        self.assertIn('msa_features', self.alphafold_integration.feature_dict)
        self.assertIn('template_features', self.alphafold_integration.feature_dict)

        mock_pipeline.make_sequence_features.assert_called_once_with(
            sequence=valid_sequence,
            description="query",
            num_res=len(valid_sequence)
        )
        mock_pipeline.make_msa_features.assert_called_once_with(msas=[[('query', valid_sequence)]])
        self.alphafold_integration._run_msa.assert_called_once_with(valid_sequence)
        self.alphafold_integration._search_templates.assert_called_once_with(valid_sequence)

        # Check if the feature_dict is correctly populated
        self.assertEqual(self.alphafold_integration.feature_dict['seq_features'], 'dummy_seq')
        self.assertEqual(self.alphafold_integration.feature_dict['msa_features'], 'dummy_msa')
        self.assertEqual(self.alphafold_integration.feature_dict['template_features'], 'dummy_template')

    @patch('NeuroFlex.scientific_domains.alphafold_integration.protein')
    @patch('NeuroFlex.scientific_domains.alphafold_integration.openmm')
    @patch('NeuroFlex.scientific_domains.alphafold_integration.jax.random.PRNGKey')
    def test_predict_structure(self, mock_prng_key, mock_openmm, mock_protein):
        # Set up mocks
        self.alphafold_integration.model = MagicMock()
        self.alphafold_integration.model_params = MagicMock()
        self.alphafold_integration.feature_dict = MagicMock()
        self.alphafold_integration.config = MagicMock()
        mock_prediction = {
            'predicted_lddt': {'logits': jnp.zeros((1, 50, 50))},
            'structure_module': {
                'final_atom_positions': jnp.zeros((1, 50, 37, 3)),
                'final_atom_mask': jnp.ones((1, 50, 37)),
            }
        }
        self.alphafold_integration.model.return_value = mock_prediction
        mock_protein_instance = MagicMock()
        mock_protein_instance.atom_positions = jnp.zeros((50, 37, 3))
        mock_protein_instance.atom_mask = jnp.ones((50, 37))
        mock_protein.from_prediction.return_value = mock_protein_instance

        mock_simulation = MagicMock()
        mock_openmm.LangevinMiddleIntegrator.return_value = MagicMock()
        mock_openmm.app.Simulation.return_value = mock_simulation
        mock_prng_key.return_value = jax.random.PRNGKey(0)

        # Mock setup_openmm_simulation
        self.alphafold_integration.setup_openmm_simulation = MagicMock()
        self.alphafold_integration.openmm_simulation = mock_simulation

        # Call the method under test
        try:
            result = self.alphafold_integration.predict_structure()
        except Exception as e:
            self.fail(f"predict_structure raised an unexpected exception: {str(e)}")

        # Assertions
        self.assertIsNotNone(result)
        self.alphafold_integration.model.assert_called_once()
        mock_protein.from_prediction.assert_called_once_with(mock_prediction)
        self.assertIsInstance(result, MagicMock)  # Ensure result is the mocked protein instance
        self.assertTrue(hasattr(result, 'atom_positions'))
        self.assertTrue(hasattr(result, 'atom_mask'))
        self.assertTrue(hasattr(result, 'residue_index'))
        mock_simulation.minimizeEnergy.assert_called_once()
        mock_simulation.step.assert_called_once_with(1000)
        mock_prng_key.assert_called_once_with(0)

        # Check if setup_openmm_simulation was called with the correct argument
        self.alphafold_integration.setup_openmm_simulation.assert_called_once_with(mock_protein_instance)

        # Verify that atom positions were updated after refinement
        self.assertTrue(jnp.any(result.atom_positions != 0))  # Assuming refinement changes positions

        # Test error handling
        self.alphafold_integration.model.side_effect = ValueError("Model error")
        with self.assertRaises(ValueError):
            self.alphafold_integration.predict_structure()

        # Test OpenMM simulation failure
        self.alphafold_integration.model.side_effect = None
        self.alphafold_integration.setup_openmm_simulation.side_effect = ValueError("OpenMM setup error")
        result = self.alphafold_integration.predict_structure()
        self.assertIsNotNone(result)  # Should still return a result even if OpenMM fails

    def test_predict_structure_not_ready(self):
        self.alphafold_integration.model = None
        self.alphafold_integration.feature_dict = None
        with self.assertRaises(ValueError) as context:
            self.alphafold_integration.predict_structure()
        self.assertIn("Model or features not set up", str(context.exception))

    @patch('NeuroFlex.scientific_domains.alphafold_integration.openmm')
    @patch('NeuroFlex.scientific_domains.alphafold_integration.app')
    @patch('NeuroFlex.scientific_domains.alphafold_integration.unit')
    def test_setup_openmm_simulation(self, mock_unit, mock_app, mock_openmm):
        mock_protein = MagicMock()
        mock_protein.residue_index = range(10)
        mock_protein.sequence = "ACDEFGHIKL"  # Using valid amino acid codes
        mock_protein.atom_names = [['N', 'CA', 'C', 'O', 'CB'] for _ in range(10)]
        mock_protein.atom_positions = [[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0], [4.0, 4.0, 4.0], [5.0, 5.0, 5.0]] for _ in range(10)]

        mock_topology = MagicMock()
        mock_app.Topology.return_value = mock_topology
        mock_forcefield = MagicMock()
        mock_app.ForceField.return_value = mock_forcefield
        mock_system = MagicMock()
        mock_forcefield.createSystem.return_value = mock_system

        mock_unit.angstrom = MagicMock()
        mock_unit.nanometer = MagicMock()
        mock_unit.kelvin = MagicMock()
        mock_unit.picosecond = MagicMock()
        mock_unit.picoseconds = MagicMock()

        mock_chain = MagicMock()
        mock_topology.addChain.return_value = mock_chain

        mock_simulation = MagicMock()
        mock_app.Simulation.return_value = mock_simulation

        # Test successful setup
        self.alphafold_integration.setup_openmm_simulation(mock_protein)

        # Verify method calls
        mock_app.Topology.assert_called_once()
        mock_app.ForceField.assert_called_once_with('amber14-all.xml', 'amber14/tip3pfb.xml')
        mock_forcefield.createSystem.assert_called_once()
        mock_openmm.LangevinMiddleIntegrator.assert_called_once_with(
            300 * mock_unit.kelvin,
            1 / mock_unit.picosecond,
            0.002 * mock_unit.picoseconds
        )
        mock_app.Simulation.assert_called_once()

        self.assertIsNotNone(self.alphafold_integration.openmm_simulation)

        # Verify topology setup
        mock_topology.addChain.assert_called_once()
        self.assertEqual(mock_topology.addResidue.call_count, 10)
        self.assertEqual(mock_topology.addAtom.call_count, 50)  # 5 atoms per residue, 10 residues

        # Verify atom addition for each residue
        for i, residue in enumerate(mock_protein.sequence):
            mock_topology.addResidue.assert_any_call(residue, mock_chain)
            for atom_name in mock_protein.atom_names[i]:
                mock_topology.addAtom.assert_any_call(atom_name, mock_app.Element.getBySymbol.return_value, mock_topology.addResidue.return_value)
                mock_app.Element.getBySymbol.assert_any_call(atom_name[0])

        # Verify position setting
        mock_simulation.context.setPositions.assert_called_once()
        positions_arg = mock_simulation.context.setPositions.call_args[0][0]
        self.assertEqual(len(positions_arg), 50)  # Total number of atoms

        # Verify unit conversion
        mock_unit.Quantity.assert_called()
        mock_unit.Quantity.return_value.in_units_of.assert_called_with(mock_unit.nanometer)

        # Test error handling for invalid protein input
        with self.assertRaises(ValueError):
            self.alphafold_integration.setup_openmm_simulation(None)

        # Test handling of non-standard residues
        mock_protein.sequence = "ACDEFGHIKLX"  # 'X' is a non-standard residue
        mock_app.PDBFile.standardResidues = set("ACDEFGHIKL")
        with self.assertLogs(level='WARNING') as cm:
            self.alphafold_integration.setup_openmm_simulation(mock_protein)
        self.assertIn("Non-standard residue X", cm.output[0])

        # Test error handling for mismatched atom names and positions
        mock_protein.sequence = "ACDEFGHIKL"
        mock_protein.atom_names = [['N', 'CA', 'C', 'O'] for _ in range(10)]  # Missing 'CB'
        mock_protein.atom_positions = [[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0], [4.0, 4.0, 4.0]] for _ in range(10)]
        with self.assertLogs(level='WARNING') as cm:
            self.alphafold_integration.setup_openmm_simulation(mock_protein)
        self.assertIn("Mismatch in atom names and positions", cm.output[0])

        # Test handling of invalid atom positions
        mock_protein.atom_names = [['N', 'CA', 'C', 'O', 'CB'] for _ in range(10)]
        mock_protein.atom_positions = [[[None, None, None], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0], [4.0, 4.0, 4.0], [5.0, 5.0, 5.0]] for _ in range(10)]
        with self.assertLogs(level='WARNING') as cm:
            self.alphafold_integration.setup_openmm_simulation(mock_protein)
        self.assertIn("Skipping invalid atom", cm.output[0])

        # Test error handling when no valid positions are found
        mock_protein.atom_positions = [[[None, None, None]] * 5 for _ in range(10)]
        with self.assertRaises(ValueError) as cm:
            self.alphafold_integration.setup_openmm_simulation(mock_protein)
        self.assertIn("No valid atom positions found", str(cm.exception))

        # Test handling of residues with no valid atoms
        mock_protein.atom_names = [['N', 'CA', 'C', 'O', 'CB'] if i != 5 else [] for i in range(10)]
        mock_protein.atom_positions = [[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0], [4.0, 4.0, 4.0], [5.0, 5.0, 5.0]] if i != 5 else [] for i in range(10)]
        with self.assertLogs(level='WARNING') as cm:
            self.alphafold_integration.setup_openmm_simulation(mock_protein)
        self.assertIn("No valid atoms found for residue", cm.output[0])

        # Verify that the simulation is still created even with some invalid data
        self.assertIsNotNone(self.alphafold_integration.openmm_simulation)

        # Test handling of all invalid atom positions
        mock_protein.atom_positions = [[[float('nan'), float('nan'), float('nan')] for _ in range(5)] for _ in range(10)]
        with self.assertRaises(ValueError) as cm:
            self.alphafold_integration.setup_openmm_simulation(mock_protein)
        self.assertIn("No valid atom positions found", str(cm.exception))

        # Test handling of mixed valid and invalid atom positions
        mock_protein.atom_positions = [
            [[1.0, 1.0, 1.0], [float('nan'), float('nan'), float('nan')], [3.0, 3.0, 3.0], [4.0, 4.0, 4.0], [5.0, 5.0, 5.0]]
            for _ in range(10)
        ]
        with self.assertLogs(level='WARNING') as cm:
            self.alphafold_integration.setup_openmm_simulation(mock_protein)
        self.assertIn("Skipping invalid atom", cm.output[0])
        self.assertEqual(mock_topology.addAtom.call_count, 40)  # 4 valid atoms per residue, 10 residues

        # Verify that the simulation handles empty atom names or positions gracefully
        mock_protein.atom_names = [[] for _ in range(10)]
        mock_protein.atom_positions = [[] for _ in range(10)]
        with self.assertRaises(ValueError) as cm:
            self.alphafold_integration.setup_openmm_simulation(mock_protein)
        self.assertIn("No valid atom positions found", str(cm.exception))

    def test_get_plddt_scores(self):
        self.alphafold_integration.model = MagicMock()
        self.alphafold_integration.model_params = MagicMock()
        self.alphafold_integration.feature_dict = MagicMock()
        self.alphafold_integration.config = MagicMock()
        mock_prediction = {'plddt': jnp.array([0.1, 0.2, 0.3])}
        self.alphafold_integration.model.return_value = mock_prediction

        scores = self.alphafold_integration.get_plddt_scores()
        self.assertTrue(jnp.allclose(scores, jnp.array([0.1, 0.2, 0.3])))
        self.alphafold_integration.model.assert_called_once()

    def test_get_plddt_scores_not_ready(self):
        self.alphafold_integration.model = None
        self.alphafold_integration.feature_dict = None
        with self.assertRaises(ValueError) as context:
            self.alphafold_integration.get_plddt_scores()
        self.assertIn("Model or features not set up", str(context.exception))

    def test_get_predicted_aligned_error(self):
        self.alphafold_integration.model = MagicMock()
        self.alphafold_integration.model_params = MagicMock()
        self.alphafold_integration.feature_dict = MagicMock()
        self.alphafold_integration.config = MagicMock()
        mock_prediction = {'predicted_aligned_error': jnp.array([[0.1, 0.2], [0.3, 0.4]])}
        self.alphafold_integration.model.return_value = mock_prediction

        error = self.alphafold_integration.get_predicted_aligned_error()
        self.assertTrue(jnp.allclose(error, jnp.array([[0.1, 0.2], [0.3, 0.4]])))
        self.alphafold_integration.model.assert_called_once()

    def test_get_predicted_aligned_error_not_ready(self):
        self.alphafold_integration.model = None
        self.alphafold_integration.feature_dict = None
        with self.assertRaises(ValueError) as context:
            self.alphafold_integration.get_predicted_aligned_error()
        self.assertIn("Model or features not set up", str(context.exception))

    @patch('NeuroFlex.scientific_domains.alphafold_integration.AlphaFoldIntegration._validate_sequence')
    @patch('NeuroFlex.scientific_domains.alphafold_integration.AlphaFoldIntegration._calculate_molecular_weight')
    @patch('NeuroFlex.scientific_domains.alphafold_integration.AlphaFoldIntegration._calculate_isoelectric_point')
    @patch('NeuroFlex.scientific_domains.alphafold_integration.AlphaFoldIntegration._calculate_hydrophobicity')
    def test_run_alphaproteo(self, mock_hydrophobicity, mock_isoelectric, mock_molecular_weight, mock_validate):
        valid_sequence = "MKFLKFSLLTAVLLSVVFAFSSCGD"
        mock_molecular_weight.return_value = 100.0
        mock_isoelectric.return_value = 7.0
        mock_hydrophobicity.return_value = 0.5

        result = self.alphafold_integration.run_alphaproteo(valid_sequence)

        self.assertIsInstance(result, dict)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["sequence_length"], len(valid_sequence))
        self.assertIn("predicted_properties", result)
        self.assertIn("molecular_weight", result["predicted_properties"])
        self.assertIn("isoelectric_point", result["predicted_properties"])
        self.assertIn("hydrophobicity", result["predicted_properties"])

        mock_validate.assert_called_once_with(valid_sequence)
        mock_molecular_weight.assert_called_once_with(valid_sequence)
        mock_isoelectric.assert_called_once_with(valid_sequence)
        mock_hydrophobicity.assert_called_once_with(valid_sequence)

        # Test with invalid inputs
        invalid_inputs = [
            "INVALID123",  # Non-standard amino acids
            "",  # Empty string
            "A" * 2001,  # Sequence too long
            123,  # Non-string input
            None,  # None input
        ]
        mock_validate.side_effect = ValueError("Invalid sequence")
        for invalid_input in invalid_inputs:
            with self.assertRaises(ValueError):
                self.alphafold_integration.run_alphaproteo(invalid_input)

        # Reset mock_validate for subsequent tests
        mock_validate.side_effect = None

        # Test with edge cases
        edge_cases = [
            ("M", 1),  # Single amino acid
            ("A" * 2000, 2000),  # Maximum allowed length
            ("ACDEFGHIKLMNPQRSTVWY", 20),  # All standard amino acids
        ]
        for sequence, expected_length in edge_cases:
            result = self.alphafold_integration.run_alphaproteo(sequence)
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["sequence_length"], expected_length)

        # Test error handling
        mock_molecular_weight.side_effect = Exception("Unexpected error")
        with self.assertRaises(RuntimeError):
            self.alphafold_integration.run_alphaproteo(valid_sequence)

        # Reset mock for final test
        mock_molecular_weight.side_effect = None
        mock_molecular_weight.return_value = 2395.74  # Actual molecular weight for the sequence

        # Test result values
        result = self.alphafold_integration.run_alphaproteo("ACDEFGHIKLMNPQRSTVWY")
        self.assertAlmostEqual(result["predicted_properties"]["molecular_weight"], 2395.74, places=2)
        self.assertGreater(result["predicted_properties"]["isoelectric_point"], 0)
        self.assertIsInstance(result["predicted_properties"]["hydrophobicity"], float)

    @patch('NeuroFlex.scientific_domains.alphafold_integration.AlphaFoldIntegration._calculate_severity_score')
    @patch('NeuroFlex.scientific_domains.alphafold_integration.AlphaFoldIntegration._calculate_confidence_score')
    @patch('NeuroFlex.scientific_domains.alphafold_integration.AlphaFoldIntegration._determine_severity')
    @patch('NeuroFlex.scientific_domains.alphafold_integration.AlphaFoldIntegration._determine_functional_impact')
    def test_run_alphamissense(self, mock_functional_impact, mock_severity, mock_confidence, mock_severity_score):
        valid_sequence = "MKFLKFSLLTAVLLSVVFAFSSCGD"
        valid_mutation = "F5L"

        # Mock return values
        mock_severity_score.return_value = 1
        mock_confidence.return_value = 0.8
        mock_severity.return_value = "moderate"
        mock_functional_impact.return_value = (0.7, "possibly damaging")

        result = self.alphafold_integration.run_alphamissense(valid_sequence, valid_mutation)

        self.assertIsInstance(result, dict)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["mutation"], valid_mutation)
        self.assertIn("predicted_effect", result)
        self.assertEqual(result["predicted_effect"]["severity"], "moderate")
        self.assertEqual(result["predicted_effect"]["confidence"], 0.8)
        self.assertEqual(result["predicted_effect"]["functional_impact"], "possibly damaging")
        self.assertEqual(result["predicted_effect"]["severity_score"], 1)
        self.assertEqual(result["predicted_effect"]["impact_score"], 0.7)

        # Test with different valid mutations
        mutations = ["M1A", "K2R", "L3I", "S10T", "D25E"]
        for mutation in mutations:
            result = self.alphafold_integration.run_alphamissense(valid_sequence, mutation)
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["mutation"], mutation)

        # Test with invalid inputs
        invalid_inputs = [
            ("INVALID123", "F5L", "Invalid amino acid sequence"),
            (valid_sequence, "INVALID", "Invalid mutation format"),
            (valid_sequence, "X5Y", "Invalid amino acid in mutation"),
            (valid_sequence, "F50L", "Invalid mutation position"),
            (valid_sequence, "G5L", "Original amino acid mismatch"),
            ("", "F5L", "Invalid sequence input"),
            (valid_sequence, "", "Invalid mutation format"),
        ]

        for seq, mut, expected_error in invalid_inputs:
            with self.assertRaises(ValueError) as context:
                self.alphafold_integration.run_alphamissense(seq, mut)
            self.assertIn(expected_error, str(context.exception))

        # Test with edge cases
        edge_cases = [
            ("A", "A1G"),  # Shortest possible sequence
            ("A" * 2000, "A1G"),  # Longest allowed sequence
            ("A" * 2001, "A1G"),  # Sequence too long
        ]

        for seq, mut in edge_cases[:2]:
            result = self.alphafold_integration.run_alphamissense(seq, mut)
            self.assertEqual(result["status"], "success")

        with self.assertRaises(ValueError) as context:
            self.alphafold_integration.run_alphamissense(edge_cases[2][0], edge_cases[2][1])
        self.assertIn("Sequence too long", str(context.exception))

        # Test error handling
        mock_severity_score.side_effect = Exception("Unexpected error")
        with self.assertRaises(RuntimeError):
            self.alphafold_integration.run_alphamissense(valid_sequence, valid_mutation)

    def test_input_validation(self):
        with self.assertRaises(ValueError):
            self.alphafold_integration.prepare_features("")
        with self.assertRaises(ValueError):
            self.alphafold_integration.prepare_features("INVALID_SEQUENCE")

    def test_model_compatibility(self):
        print(f"DEBUG: HAIKU_COMPATIBLE value: {HAIKU_COMPATIBLE}")  # Debug print
        self.assertTrue(ALPHAFOLD_COMPATIBLE)
        self.assertTrue(JAX_COMPATIBLE)
        self.assertTrue(HAIKU_COMPATIBLE)
        self.assertTrue(OPENMM_COMPATIBLE)

if __name__ == '__main__':
    unittest.main()
