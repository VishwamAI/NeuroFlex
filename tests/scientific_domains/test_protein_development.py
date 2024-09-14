import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from NeuroFlex.scientific_domains.protein_development import ProteinDevelopment

class TestProteinDevelopment(unittest.TestCase):
    def setUp(self):
        self.protein_dev = ProteinDevelopment()

    @patch('NeuroFlex.scientific_domains.protein_development.config.model_config')
    @patch('NeuroFlex.scientific_domains.protein_development.data.get_model_haiku_params')
    @patch('NeuroFlex.scientific_domains.protein_development.model.AlphaFold')
    def test_setup_alphafold(self, mock_alphafold, mock_get_params, mock_model_config):
        mock_model_config.return_value = {'mock': 'config'}
        mock_get_params.return_value = {'mock': 'params'}
        mock_alphafold_instance = mock_alphafold.return_value

        self.protein_dev.setup_alphafold()

        mock_model_config.assert_called_once_with('model_1')
        mock_get_params.assert_called_once_with('model_1', '/path/to/alphafold/data')
        mock_alphafold.assert_called_once_with({'mock': 'config'})
        mock_alphafold_instance.init_params.assert_called_once_with({'mock': 'params'})
        self.assertIsNotNone(self.protein_dev.alphafold_model)

    @patch('NeuroFlex.scientific_domains.protein_development.pipeline.make_sequence_features')
    @patch('NeuroFlex.scientific_domains.protein_development.protein.from_prediction')
    def test_predict_structure(self, mock_from_prediction, mock_make_sequence_features):
        sequence = "MKFLKFSLLTAVLLSVVFAFSSCGD"
        mock_features = {
            'aatype': np.zeros((len(sequence), 21)),
            'between_segment_residues': np.zeros((len(sequence),)),
            'domain_name': np.array(['test_domain'], dtype=np.object_),
            'residue_index': np.arange(len(sequence)),
            'seq_length': np.array([len(sequence)] * len(sequence)),
            'sequence': np.array([sequence.encode()], dtype=np.object_)
        }
        mock_make_sequence_features.return_value = mock_features

        mock_prediction = {
            'plddt': np.random.rand(len(sequence)),
            'predicted_tm_score': 0.85,
            'final_atom_positions': np.random.rand(len(sequence), 37, 3),
            'final_atom_mask': np.ones((len(sequence), 37))
        }
        self.protein_dev.alphafold_model = MagicMock()
        self.protein_dev.alphafold_model.predict.return_value = mock_prediction

        mock_structure = MagicMock()
        mock_from_prediction.return_value = mock_structure

        result = self.protein_dev.predict_structure(sequence)

        mock_make_sequence_features.assert_called_once_with(sequence, description="", num_res=len(sequence))
        self.protein_dev.alphafold_model.predict.assert_called_once_with(mock_features)
        mock_from_prediction.assert_called_once_with(mock_prediction, mock_features)

        self.assertIsInstance(result, dict)
        self.assertIn('structure', result)
        self.assertIn('plddt', result)
        self.assertIn('predicted_tm_score', result)
        self.assertEqual(result['structure'], mock_structure)
        np.testing.assert_array_equal(result['plddt'], mock_prediction['plddt'])
        self.assertEqual(result['predicted_tm_score'], mock_prediction['predicted_tm_score'])

    def test_predict_structure_no_model(self):
        with self.assertRaises(ValueError):
            self.protein_dev.predict_structure("MKFLKFSLLTAVLLSVVFAFSSCGD")

    @patch('NeuroFlex.scientific_domains.protein_development.app.Topology')
    @patch('NeuroFlex.scientific_domains.protein_development.app.ForceField')
    @patch('NeuroFlex.scientific_domains.protein_development.openmm.LangevinMiddleIntegrator')
    @patch('NeuroFlex.scientific_domains.protein_development.app.Simulation')
    def test_setup_openmm_simulation(self, mock_simulation, mock_integrator, mock_forcefield, mock_topology):
        mock_protein_structure = MagicMock()
        mock_protein_structure.residue_index = range(3)
        mock_protein_structure.sequence = "ABC"
        mock_protein_structure.atom_mask = [[True, False], [True, True], [False, True]]
        mock_protein_structure.atom_positions = [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]], [[5, 5, 5], [6, 6, 6]]]

        self.protein_dev.setup_openmm_simulation(mock_protein_structure)

        mock_topology.assert_called_once()
        mock_forcefield.assert_called_once_with('amber14-all.xml', 'amber14/tip3pfb.xml')
        mock_integrator.assert_called_once()
        mock_simulation.assert_called_once()
        self.assertIsNotNone(self.protein_dev.openmm_simulation)

    def test_run_molecular_dynamics(self):
        self.protein_dev.openmm_simulation = MagicMock()
        self.protein_dev.run_molecular_dynamics(1000)
        self.protein_dev.openmm_simulation.minimizeEnergy.assert_called_once()
        self.protein_dev.openmm_simulation.step.assert_called_once_with(1000)

    def test_run_molecular_dynamics_no_simulation(self):
        with self.assertRaises(ValueError):
            self.protein_dev.run_molecular_dynamics(1000)

    def test_get_current_positions(self):
        mock_state = MagicMock()
        mock_state.getPositions.return_value = np.array([[1, 2, 3], [4, 5, 6]])
        self.protein_dev.openmm_simulation = MagicMock()
        self.protein_dev.openmm_simulation.context.getState.return_value = mock_state

        positions = self.protein_dev.get_current_positions()
        np.testing.assert_array_equal(positions, np.array([[1, 2, 3], [4, 5, 6]]))

    def test_get_current_positions_no_simulation(self):
        with self.assertRaises(ValueError):
            self.protein_dev.get_current_positions()

if __name__ == '__main__':
    unittest.main()
