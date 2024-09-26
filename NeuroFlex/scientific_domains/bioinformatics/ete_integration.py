# MIT License
# 
# Copyright (c) 2024 VishwamAI
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# ete_integration.py
from ete3 import Tree
from ete3.treeview import TreeStyle
import logging

class ETEIntegration:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def create_tree(self, newick_string):
        try:
            return Tree(newick_string)
        except Exception as e:
            raise ValueError(f"Invalid Newick string: {str(e)}")

    def render_tree(self, tree):
        if not isinstance(tree, Tree):
            raise TypeError("Input must be an ete3 Tree object")
        ts = TreeStyle()
        ts.show_leaf_name = True
        return tree.render("phylo.png", tree_style=ts)

    def analyze_tree(self, tree):
        if not isinstance(tree, Tree):
            raise TypeError("Input must be an ete3 Tree object")
        root = tree.get_tree_root()
        farthest_leaf = tree.get_farthest_leaf()
        analysis = {
            'num_leaves': len(tree.get_leaf_names()),
            'total_branch_length': tree.get_distance(root, farthest_leaf[0]),
            'root': root,
            'farthest_leaf': farthest_leaf
        }
        return analysis

    def visualize_tree(self, tree, output_file):
        if not isinstance(tree, Tree):
            raise TypeError("Input must be an ete3 Tree object")
        try:
            ts = TreeStyle()
            ts.show_leaf_name = True
            ts.show_branch_length = True
            ts.show_branch_support = True
            tree.render(output_file, tree_style=ts)
            self.logger.info(f"Tree visualization saved to {output_file}")
        except Exception as e:
            self.logger.error(f"Error visualizing tree: {str(e)}")
            raise

    def get_tree_statistics(self, tree):
        if not isinstance(tree, Tree):
            raise TypeError("Input must be an ete3 Tree object")
        try:
            stats = {
                'num_leaves': len(tree.get_leaves()),
                'num_internal_nodes': len(tree.get_descendants()) - len(tree.get_leaves()),
                'tree_depth': tree.get_farthest_node()[1],
                'total_branch_length': tree.get_distance(tree.get_tree_root(), tree.get_farthest_leaf()[0]),
                'root_children': len(tree.get_children()),
                'is_binary': all(len(node.children) in (0, 2) for node in tree.traverse()),
                # 'is_ultrametric' check removed as it's not available for TreeNode objects
            }
            self.logger.info("Tree statistics calculated successfully")
            return stats
        except Exception as e:
            self.logger.error(f"Error calculating tree statistics: {str(e)}")
            raise
