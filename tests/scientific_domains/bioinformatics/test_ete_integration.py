import unittest
import pytest
from unittest.mock import patch, MagicMock
from NeuroFlex.scientific_domains.bioinformatics.ete_integration import ETEIntegration
from ete3 import Tree

class TestETEIntegration(unittest.TestCase):
    def setUp(self):
        self.ete_integration = ETEIntegration()

    def test_create_tree_valid_newick(self):
        newick_string = "((A,B),(C,D));"
        with patch('NeuroFlex.scientific_domains.bioinformatics.ete_integration.Tree') as mock_tree:
            mock_tree_instance = MagicMock()
            mock_tree.return_value = mock_tree_instance
            result = self.ete_integration.create_tree(newick_string)
            mock_tree.assert_called_once_with(newick_string)
            self.assertIsNotNone(result)
            self.assertEqual(result, mock_tree_instance)

    def test_create_tree_invalid_newick(self):
        invalid_newick = "((A,B),(C,D);"  # Missing closing parenthesis
        with patch('NeuroFlex.scientific_domains.bioinformatics.ete_integration.Tree') as mock_tree:
            mock_tree.side_effect = ValueError("Invalid newick format")
            with self.assertRaises(ValueError):
                self.ete_integration.create_tree(invalid_newick)

    def test_render_tree(self):
        mock_tree = MagicMock(spec=Tree)
        mock_tree.render.return_value = "rendered_tree.png"
        with patch('NeuroFlex.scientific_domains.bioinformatics.ete_integration.TreeStyle') as mock_tree_style:
            mock_tree_style_instance = MagicMock()
            mock_tree_style.return_value = mock_tree_style_instance
            result = self.ete_integration.render_tree(mock_tree)
            mock_tree.render.assert_called_once_with("phylo.png", tree_style=mock_tree_style_instance)
            self.assertEqual(result, "rendered_tree.png")
            self.assertIsNotNone(result)

    def test_render_tree_invalid_input(self):
        with self.assertRaises(TypeError):
            self.ete_integration.render_tree("not a tree object")

    def test_analyze_tree(self):
        mock_tree = MagicMock(spec=Tree)
        mock_tree.get_leaf_names.return_value = ['A', 'B', 'C', 'D']
        mock_tree.get_distance.return_value = 10.0
        mock_tree.get_tree_root.return_value = MagicMock()
        mock_tree.get_farthest_leaf.return_value = (MagicMock(), 10.0)

        result = self.ete_integration.analyze_tree(mock_tree)

        self.assertIsInstance(result, dict)
        self.assertIn('num_leaves', result)
        self.assertIn('total_branch_length', result)
        self.assertEqual(result['num_leaves'], 4)
        self.assertEqual(result['total_branch_length'], 10.0)

    def test_analyze_tree_empty(self):
        mock_tree = MagicMock(spec=Tree)
        mock_tree.get_leaf_names.return_value = []
        mock_tree.get_distance.return_value = 0.0
        mock_tree.get_tree_root.return_value = MagicMock()
        mock_tree.get_farthest_leaf.return_value = (MagicMock(), 0.0)

        result = self.ete_integration.analyze_tree(mock_tree)

        self.assertEqual(result['num_leaves'], 0)
        self.assertEqual(result['total_branch_length'], 0.0)
        self.assertIn('root', result)
        self.assertIn('farthest_leaf', result)
        self.assertEqual(result['farthest_leaf'][1], 0.0)

if __name__ == '__main__':
    unittest.main()
