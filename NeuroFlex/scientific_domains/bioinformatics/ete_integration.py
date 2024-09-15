# ete_integration.py
from ete3 import Tree
from ete3.treeview import TreeStyle

class ETEIntegration:
    def __init__(self):
        pass

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
        farthest_leaf, distance = tree.get_farthest_leaf()

        analysis = {
            'num_leaves': len(tree.get_leaf_names()),
            'total_branch_length': distance,
            'root': root,
            'farthest_leaf': (farthest_leaf, distance)
        }

        return analysis
