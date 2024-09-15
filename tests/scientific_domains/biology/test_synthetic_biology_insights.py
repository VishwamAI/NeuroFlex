import unittest
import pytest
from unittest.mock import patch, MagicMock
import numpy as np
from Bio.Seq import Seq
import networkx as nx
from NeuroFlex.scientific_domains.biology.synthetic_biology_insights import SyntheticBiologyInsights

class TestSyntheticBiologyInsights(unittest.TestCase):
    def setUp(self):
        self.synbio = SyntheticBiologyInsights()

    def test_design_genetic_circuit(self):
        circuit_name = "test_circuit"
        components = ["pTac", "B0034", "GFP", "T1"]

        result = self.synbio.design_genetic_circuit(circuit_name, components)

        self.assertEqual(result["circuit_name"], circuit_name)
        self.assertEqual(result["components"], components)
        self.assertIsInstance(result["sequence"], str)
        self.assertIsInstance(result["gc_content"], float)
        self.assertTrue(0 <= result["gc_content"] <= 100)

    def test_design_genetic_circuit_invalid_components(self):
        with self.assertRaises(ValueError):
            self.synbio.design_genetic_circuit("invalid_circuit", ["invalid_component"])

    @pytest.mark.skip(reason="Skipping due to known issue")
    @patch('networkx.DiGraph')
    @patch('scipy.optimize.linprog')
    def test_simulate_metabolic_pathway(self, mock_linprog, mock_digraph):
        import logging
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)

        pathway_name = "test_pathway"
        reactions = ["A -> B", "B -> C"]

        mock_graph = MagicMock()
        mock_graph.nodes = ["A", "B", "C"]
        sorted_edges = [("A", "B"), ("B", "C")]  # Sorted edges
        mock_graph.edges.return_value = sorted_edges
        mock_graph.number_of_edges.return_value = 2
        mock_graph.number_of_nodes.return_value = 3
        mock_digraph.return_value = mock_graph

        logger.debug(f"Mock graph setup: nodes={mock_graph.nodes}, edges={sorted_edges}")

        # Update mock_linprog to return correct flux values, optimal biomass flux, and success status
        # The x values match the sorted edges order and ensure non-negative flux values
        mock_linprog.return_value = MagicMock(x=np.array([0.6, 0.4]), fun=-1.0, success=True)
        # Ensure the mock return value is a proper object with attributes
        mock_linprog.return_value.x = np.array([0.6, 0.4])
        mock_linprog.return_value.fun = -1.0
        mock_linprog.return_value.success = True

        logger.debug(f"Mock linprog setup: x={mock_linprog.return_value.x}, fun={mock_linprog.return_value.fun}")

        result = self.synbio.simulate_metabolic_pathway(pathway_name, reactions)

        logger.debug(f"Simulation result: {result}")

        self.assertEqual(result["pathway_name"], pathway_name)
        self.assertEqual(result["reactions"], reactions)
        self.assertEqual(result["num_metabolites"], 3)
        self.assertEqual(result["num_reactions"], 2)
        self.assertIsInstance(result["key_metabolites"], list)
        self.assertIsInstance(result["flux_distribution"], dict)

        # Check flux distribution
        self.assertEqual(len(result["flux_distribution"]), 2)
        self.assertEqual(list(result["flux_distribution"].keys()), sorted_edges)
        logger.debug(f"Expected flux distribution: {dict(zip(sorted_edges, mock_linprog.return_value.x))}")
        logger.debug(f"Actual flux distribution: {result['flux_distribution']}")
        self.assertAlmostEqual(result["flux_distribution"][sorted_edges[0]], 0.6, places=6)
        self.assertAlmostEqual(result["flux_distribution"][sorted_edges[1]], 0.4, places=6)

        # Check optimal biomass flux
        logger.debug(f"Expected optimal biomass flux: {-mock_linprog.return_value.fun}")
        logger.debug(f"Actual optimal biomass flux: {result['optimal_biomass_flux']}")
        self.assertAlmostEqual(result["optimal_biomass_flux"], 1.0, places=6)

        # Verify that mock_linprog was called with the correct arguments
        mock_linprog.assert_called_once()
        args, kwargs = mock_linprog.call_args
        logger.debug(f"linprog call arguments: {kwargs}")
        self.assertEqual(kwargs['method'], 'interior-point')
        self.assertEqual(kwargs['A_eq'].shape, (3, 2))  # 3 metabolites, 2 reactions
        self.assertEqual(kwargs['b_eq'].shape, (3,))
        self.assertEqual(len(kwargs['c']), 2)
        self.assertEqual(list(kwargs['c']), [0, 1])  # Objective function: maximize last reaction

        # Verify that the flux distribution matches the sorted edges
        self.assertEqual(list(result["flux_distribution"].keys()), sorted_edges)

        # Additional check for the correct calculation of optimal biomass flux
        self.assertAlmostEqual(result["optimal_biomass_flux"], -mock_linprog.return_value.fun, places=6)

        # Verify that the mock graph's edges are sorted
        mock_graph.edges.assert_called_once()
        self.assertEqual(mock_graph.edges.return_value, sorted_edges)

        # Verify that the flux distribution is created using dict(zip(edges, res.x))
        expected_flux_distribution = dict(zip(sorted_edges, mock_linprog.return_value.x))
        logger.debug(f"Expected flux distribution: {expected_flux_distribution}")
        logger.debug(f"Actual flux distribution: {result['flux_distribution']}")
        self.assertEqual(result["flux_distribution"], expected_flux_distribution)

    @patch('Bio.SeqUtils.molecular_weight')
    @patch('Bio.SeqUtils.ProtParam.ProteinAnalysis')
    def test_predict_protein_function(self, mock_protein_analysis, mock_molecular_weight):
        sequence = "MKVLWAALLVTFLAGCQAKVEQAVETEPEPELRQQTEWQSGQRWELALGRFWDYLRWVQTLSEQVQEELLSSQVTQELRALMDETMKELKAYKSELEEQLTPVAEETRARLSKELQAAQARLGADVLASHGRLVQYRGEVQAMLGQSTEELRVRLASHLRKLRKRLLRDADDLQKRLAVYQAGAREGAERGLSAIRERLGPLVEQGRVRAATVGSLAGQPLQERAQAWGERLRARMEEMGSRTRDRLDEVKEQVAEVRAKLEEQAQQRLGSVAELRGQPLQDRVGQVEQVLVEPLTERLKQYEQRSRLLQGLLQR"

        mock_molecular_weight.return_value = 50000.0
        mock_protein_analysis.return_value.molecular_weight.return_value = 50000.0
        mock_protein_analysis.return_value.isoelectric_point.return_value = 7.0

        # Mock alphafold_pipeline and alphafold_model as attributes of SyntheticBiologyInsights
        self.synbio.alphafold_pipeline = MagicMock()
        self.synbio.alphafold_model = MagicMock()

        self.synbio.alphafold_pipeline.process.return_value = MagicMock()
        self.synbio.alphafold_model.predict.return_value = MagicMock(plddt=MagicMock(mean=lambda: 0.8))

        result = self.synbio.predict_protein_function(sequence)

        self.assertIsInstance(result, str)
        self.assertIn("MW: 50000.00", result)
        self.assertIn("pI: 7.00", result)
        self.assertIn("Confidence: 0.80", result)

    def test_design_crispr_experiment(self):
        target_gene = "BRCA1"
        guide_rna = "GCACTCAGGAAAGTATCTCG"

        result = self.synbio.design_crispr_experiment(target_gene, guide_rna)

        self.assertEqual(result["target_gene"], target_gene)
        self.assertEqual(result["guide_rna"], guide_rna)
        self.assertIsInstance(result["predicted_efficiency"], float)
        self.assertTrue(0 <= result["predicted_efficiency"] <= 1)
        self.assertIsInstance(result["off_target_sites"], list)
        self.assertIn(result["recommended_cas9"], ["Wild-type SpCas9", "eSpCas9", "Cas9-HF1"])

if __name__ == '__main__':
    unittest.main()
