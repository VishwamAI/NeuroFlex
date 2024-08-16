from ete3 import Tree, TreeStyle
from typing import List, Optional

class ETEIntegration:
    def __init__(self):
        pass

    def create_tree(self, newick_string: str) -> Tree:
        """
        Create a phylogenetic tree from a Newick string.

        Args:
            newick_string (str): Newick format string representing the tree.

        Returns:
            Tree: ETE Tree object.
        """
        return Tree(newick_string)

    def visualize_tree(self, tree: Tree, output_file: str, show_branch_length: bool = True) -> None:
        """
        Visualize the phylogenetic tree and save it to a file.

        Args:
            tree (Tree): ETE Tree object to visualize.
            output_file (str): Path to save the visualization.
            show_branch_length (bool): Whether to display branch lengths (default: True).
        """
        ts = TreeStyle()
        ts.show_branch_length = show_branch_length
        ts.show_branch_support = True
        tree.render(output_file, tree_style=ts)

    def load_tree_from_file(self, file_path: str, format: int = 0) -> Tree:
        """
        Load a tree from a file.

        Args:
            file_path (str): Path to the tree file.
            format (int): File format (0 for auto-detection, 1 for Newick, 2 for NHX, etc.).

        Returns:
            Tree: ETE Tree object.
        """
        return Tree(file_path, format=format)

    def get_tree_statistics(self, tree: Tree) -> dict:
        """
        Get basic statistics about the tree.

        Args:
            tree (Tree): ETE Tree object.

        Returns:
            dict: Dictionary containing tree statistics.
        """
        return {
            "num_leaves": len(tree.get_leaves()),
            "num_nodes": len(tree.get_descendants()) + 1,
            "tree_length": tree.get_farthest_leaf()[1],
        }

    def compare_trees(self, tree1: Tree, tree2: Tree) -> float:
        """
        Compare two trees using Robinson-Foulds distance.

        Args:
            tree1 (Tree): First ETE Tree object.
            tree2 (Tree): Second ETE Tree object.

        Returns:
            float: Robinson-Foulds distance between the two trees.
        """
        return tree1.compare(tree2, unrooted=True)["rf"]
