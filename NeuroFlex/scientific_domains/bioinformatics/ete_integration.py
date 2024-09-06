# ete_integration.py

from ete3 import Tree

class ETEIntegration:
    def __init__(self):
        pass

    def create_tree(self, newick_string):
        """
        Create a phylogenetic tree from a Newick string.
        
        :param newick_string: A string in Newick format.
        :return: A Tree object.
        """
        try:
            tree = Tree(newick_string)
            return tree
        except Exception as e:
            print(f"Error creating tree: {e}")
            return None

    def render_tree(self, tree, output_file="tree.png"):
        """
        Render a phylogenetic tree and save it as an image.
        
        :param tree: A Tree object.
        :param output_file: The file path to save the rendered tree image.
        """
        try:
            tree.render(output_file, w=183, units="mm")
            print(f"Tree rendered and saved to {output_file}")
        except Exception as e:
            print(f"Error rendering tree: {e}")

    def analyze_tree(self, tree):
        """
        Analyze the phylogenetic tree (basic statistics).
        
        :param tree: A Tree object.
        """
        try:
            print(f"Tree has {len(tree)} leaves.")
            print(f"Tree has {len(tree.get_descendants())} nodes.")
            
            for leaf in tree:
                print(f"Leaf: {leaf.name}, Distance to root: {leaf.get_distance(tree)}")
                
        except Exception as e:
            print(f"Error analyzing tree: {e}")