# alphafold_integration.py

import sys
import os

# Add the AlphaFold directory to the Python path
alphafold_path = os.environ.get('ALPHAFOLD_PATH')
if not alphafold_path:
    print("Error: ALPHAFOLD_PATH environment variable is not set.")
    print("Please set the ALPHAFOLD_PATH environment variable to the AlphaFold directory.")
    sys.exit(1)

sys.path.append(alphafold_path)

try:
    from alphafold.common import confidence, residue_constants
    from alphafold.model import features, modules, modules_multimer
    from alphafold.data import msa_identifiers, parsers, templates
    from alphafold.data.tools import hhblits, hhsearch, hmmsearch, jackhmmer
except ImportError as e:
    print(f"Error importing AlphaFold modules: {e}")
    print("Please ensure that the AlphaFold path is correctly set and all dependencies are installed.")
    sys.exit(1)

class AlphaFoldIntegration:
    def __init__(self):
        self.confidence_module = confidence
        self.features_module = features
        self.modules_module = modules
        self.modules_multimer_module = modules_multimer
        self.residue_constants_module = residue_constants
        self.msa_identifiers_module = msa_identifiers
        self.parsers_module = parsers
        self.templates_module = templates
        self.hhblits_module = hhblits
        self.hhsearch_module = hhsearch
        self.hmmsearch_module = hmmsearch
        self.jackhmmer_module = jackhmmer

    def get_plddt_bands(self):
        """
        Returns the pLDDT confidence bands used in AlphaFold.
        """
        return self.confidence_module.PLDDT_BANDS

    def calculate_plddt(self, logits):
        """
        Computes per-residue pLDDT from logits.

        Args:
            logits: The logits output from the AlphaFold model.

        Returns:
            Per-residue pLDDT scores.
        """
        return self.confidence_module.compute_plddt(logits)

    def get_feature_names(self):
        """
        Returns a list of feature names used in AlphaFold.
        """
        return self.features_module.get_feature_names()

    def make_sequence_features(self, sequence, description, num_res):
        """
        Creates sequence features for AlphaFold.

        Args:
            sequence: The protein sequence.
            description: A description of the sequence.
            num_res: The number of residues in the sequence.

        Returns:
            A dictionary of sequence features.
        """
        return self.features_module.make_sequence_features(
            sequence=sequence,
            description=description,
            num_res=num_res
        )

    def make_msa_features(self, msas):
        """
        Creates MSA features for AlphaFold.

        Args:
            msas: A list of MSAs, each MSA is a list of sequences.

        Returns:
            A dictionary of MSA features.
        """
        return self.features_module.make_msa_features(msas=msas)

    def create_embeddings_and_evoformer(self, config):
        """
        Creates an instance of EmbeddingsAndEvoformer module.

        Args:
            config: A configuration dictionary for the module.

        Returns:
            An instance of EmbeddingsAndEvoformer.
        """
        return self.modules_module.EmbeddingsAndEvoformer(**config)

    def create_embeddings_and_evoformer_multimer(self, config):
        """
        Creates an instance of EmbeddingsAndEvoformer module for multimers.

        Args:
            config: A configuration dictionary for the module.

        Returns:
            An instance of EmbeddingsAndEvoformer for multimers.
        """
        return self.modules_multimer_module.EmbeddingsAndEvoformer(**config)

    def create_structure_module_multimer(self, config):
        """
        Creates an instance of StructureModule for multimers.

        Args:
            config: A configuration dictionary for the module.

        Returns:
            An instance of StructureModule for multimers.
        """
        return self.modules_multimer_module.StructureModule(**config)

    def create_heads_multimer(self, config):
        """
        Creates instances of prediction heads for multimers.

        Args:
            config: A configuration dictionary for the heads.

        Returns:
            A dictionary of prediction head instances for multimers.
        """
        return {
            'predicted_lddt': self.modules_multimer_module.PredictedLDDTHead(**config),
            'predicted_aligned_error': self.modules_multimer_module.PredictedAlignedErrorHead(**config),
            'predicted_tm_score': self.modules_multimer_module.PredictedTMScoreHead(**config),
        }

    def get_residue_constants(self):
        """
        Returns the residue constants used in AlphaFold.

        Returns:
            A dictionary containing various residue-related constants.
        """
        return {
            'restype_order': self.residue_constants.restype_order,
            'restype_num': self.residue_constants.restype_num,
            'restypes': self.residue_constants.restypes,
            'hhblits_aa_to_id': self.residue_constants.hhblits_aa_to_id,
            'atom_types': self.residue_constants.atom_types,
            'atom_order': self.residue_constants.atom_order,
            'restype_name_to_atom14_names': self.residue_constants.restype_name_to_atom14_names,
            'restype_name_to_atom37_names': self.residue_constants.restype_name_to_atom37_names,
        }

    def sequence_to_onehot(self, sequence):
        """
        Converts an amino acid sequence to one-hot encoding.

        Args:
            sequence: A string of amino acid letters.

        Returns:
            A numpy array of shape (len(sequence), 20) representing the one-hot encoding.
        """
        return self.residue_constants.sequence_to_onehot(sequence)

    def get_chi_angles(self, restype):
        """
        Returns the chi angles for a given residue type.

        Args:
            restype: A string representing the residue type (e.g., 'ALA', 'GLY', etc.).

        Returns:
            A list of chi angles for the given residue type.
        """
        return self.residue_constants.chi_angles_atoms.get(restype, [])

    def parse_msa_identifier(self, msa_identifier):
        """
        Parses an MSA identifier string into its components.

        Args:
            msa_identifier: A string representing the MSA identifier.

        Returns:
            A dictionary containing the parsed components of the MSA identifier.
        """
        return self.msa_identifiers_module.parse_msa_identifier(msa_identifier)

    def get_msa_identifier_type(self, msa_identifier):
        """
        Determines the type of the MSA identifier.

        Args:
            msa_identifier: A string representing the MSA identifier.

        Returns:
            A string indicating the type of the MSA identifier.
        """
        return self.msa_identifiers_module.get_msa_identifier_type(msa_identifier)

    def create_msa_identifier(self, database, sequence_id, chain_id=None):
        """
        Creates an MSA identifier string from its components.

        Args:
            database: The name of the database.
            sequence_id: The sequence identifier.
            chain_id: The chain identifier (optional).

        Returns:
            A string representing the MSA identifier.
        """
        return self.msa_identifiers_module.create_msa_identifier(database, sequence_id, chain_id)

    def parse_pdb(self, pdb_string):
        """
        Parses a PDB file string.

        Args:
            pdb_string: A string containing the contents of a PDB file.

        Returns:
            A dictionary containing parsed PDB data.
        """
        return self.parsers_module.parse_pdb(pdb_string)

    def parse_a3m(self, a3m_string):
        """
        Parses an A3M file string.

        Args:
            a3m_string: A string containing the contents of an A3M file.

        Returns:
            A tuple containing the parsed A3M data.
        """
        return self.parsers_module.parse_a3m(a3m_string)

    def parse_hhr(self, hhr_string):
        """
        Parses an HHSearch result file string.

        Args:
            hhr_string: A string containing the contents of an HHSearch result file.

        Returns:
            A list of dictionaries containing parsed HHSearch results.
        """
        return self.parsers_module.parse_hhr(hhr_string)

    def create_template_features(self, query_sequence, hits):
        """
        Creates template features for AlphaFold.

        Args:
            query_sequence: The query protein sequence.
            hits: A list of template hits.

        Returns:
            A dictionary of template features.
        """
        return self.templates_module.create_template_features(
            query_sequence=query_sequence,
            hits=hits
        )

    def realign_templates(self, query_sequence, template_features):
        """
        Realigns templates to the query sequence.

        Args:
            query_sequence: The query protein sequence.
            template_features: A dictionary of template features.

        Returns:
            A dictionary of realigned template features.
        """
        return self.templates_module.realign_templates(
            query_sequence=query_sequence,
            template_features=template_features
        )

    def get_template_hits(self, query_sequence, mmcif_dir):
        """
        Retrieves template hits for a given query sequence.

        Args:
            query_sequence: The query protein sequence.
            mmcif_dir: Directory containing mmCIF files.

        Returns:
            A list of template hits.
        """
        return self.templates_module.get_template_hits(
            query_sequence=query_sequence,
            mmcif_dir=mmcif_dir
        )

    def run_hhblits(self, input_fasta_path, database_paths, num_iterations=3):
        """
        Runs HHBlits search using the provided input sequence and databases.

        Args:
            input_fasta_path: Path to the input FASTA file.
            database_paths: List of paths to the HHBlits databases.
            num_iterations: Number of HHBlits iterations to perform (default: 3).

        Returns:
            A tuple containing the output A3M string and the HHBlits output string.
        """
        return self.hhblits_module.run_hhblits(
            input_fasta_path=input_fasta_path,
            database_paths=database_paths,
            num_iterations=num_iterations
        )

    def run_hhsearch(self, input_a3m_path, database_path):
        """
        Runs HHSearch using the provided input A3M file and database.

        Args:
            input_a3m_path: Path to the input A3M file.
            database_path: Path to the HHSearch database.

        Returns:
            A tuple containing the output HHR string and the HHSearch output string.
        """
        return self.hhsearch_module.run_hhsearch(
            input_a3m_path=input_a3m_path,
            database_path=database_path
        )

    def run_hmmsearch(self, input_fasta_path, database_path):
        """
        Runs HMMSearch using the provided input FASTA file and database.

        Args:
            input_fasta_path: Path to the input FASTA file.
            database_path: Path to the HMMSearch database.

        Returns:
            A tuple containing the output HMM string and the HMMSearch output string.
        """
        return self.hmmsearch_module.run_hmmsearch(
            input_fasta_path=input_fasta_path,
            database_path=database_path
        )

    def run_jackhmmer(self, input_fasta_path, database_path, num_iterations=1):
        """
        Runs Jackhmmer search using the provided input FASTA file and database.

        Args:
            input_fasta_path: Path to the input FASTA file.
            database_path: Path to the Jackhmmer database.
            num_iterations: Number of Jackhmmer iterations to perform (default: 1).

        Returns:
            A tuple containing the output A3M string and the Jackhmmer output string.
        """
        return self.jackhmmer_module.run_jackhmmer(
            input_fasta_path=input_fasta_path,
            database_path=database_path,
            num_iterations=num_iterations
        )

if __name__ == "__main__":
    # Example usage
    af_integration = AlphaFoldIntegration()
    print("pLDDT bands:", af_integration.get_plddt_bands())

    # You would typically get logits from the AlphaFold model output
    # This is just a placeholder for demonstration
    import numpy as np
    dummy_logits = np.random.randn(100, 50)  # 100 residues, 50 bins
    plddt_scores = af_integration.calculate_plddt(dummy_logits)
    print("Example pLDDT scores (first 5 residues):", plddt_scores[:5])

    # Example usage of new feature-related methods
    print("Feature names:", af_integration.get_feature_names())

    # Example sequence features
    sequence = "MKFLKFSLLTAVLLSVVFAFSSCGDDDDTGYLPPSQAIQDLLKRMKV"
    description = "Example protein"
    num_res = len(sequence)
    seq_features = af_integration.make_sequence_features(sequence, description, num_res)
    print("Sequence features keys:", seq_features.keys())

    # Example MSA features
    msas = [["MKFLKFSLLTAVLLSVVFAFSSCGDDDDTGYLPPSQAIQDLLKRMKV",
             "MKFLKFSLLTAVLLSVVFAFSSCGDDDDTGYLPPSQAIQDLLKRMKV"]]
    msa_features = af_integration.make_msa_features(msas)
    print("MSA features keys:", msa_features.keys())

    # Example usage of EmbeddingsAndEvoformer
    config = {
        "num_features": 384,
        "num_msa": 512,
        "num_extra_msa": 1024,
        "num_evoformer_blocks": 48,
        "evoformer_config": {
            "msa_row_attention_with_pair_bias": {"dropout_rate": 0.15},
            "msa_column_attention": {"dropout_rate": 0.0},
            "msa_transition": {"dropout_rate": 0.0},
            "outer_product_mean": {"chunk_size": 128},
            "triangle_attention_starting_node": {"dropout_rate": 0.25},
            "triangle_attention_ending_node": {"dropout_rate": 0.25},
            "triangle_multiplication_outgoing": {"dropout_rate": 0.25},
            "triangle_multiplication_incoming": {"dropout_rate": 0.25},
            "pair_transition": {"dropout_rate": 0.0},
        },
    }
    embeddings_and_evoformer = af_integration.create_embeddings_and_evoformer(config)
    print("EmbeddingsAndEvoformer created successfully")

    # Example usage of MSA identifier methods
    msa_sequence = "MKFLKFSLLTAVLLSVVFAFSSCGDDDDTGYLPPSQAIQDLLKRMKV"
    msa_description = "sp|P12345|EXAMPLE_HUMAN Example protein"
    parsed_msa = af_integration.parse_msa_identifier(msa_description)
    print("Parsed MSA identifier:", parsed_msa)

    msa_type = af_integration.get_msa_identifier_type(msa_description)
    print("MSA identifier type:", msa_type)

    # Example usage of new parsing methods
    pdb_string = """ATOM      1  N   ASP A   1      22.981  23.225  50.360  1.00 18.79           N
ATOM      2  CA  ASP A   1      21.967  22.778  49.506  1.00 17.87           C
ATOM      3  C   ASP A   1      21.980  23.431  48.129  1.00 15.88           C
END
"""
    parsed_pdb = af_integration.parse_pdb(pdb_string)
    print("Parsed PDB data keys:", parsed_pdb.keys())

    a3m_string = """>seq1
MKFLKFSLLTAVLLSVVFAFSSCGDDDDTGYLPPSQAIQDLLKRMKV
>seq2
MK--KFSLLTAVLLSVVFAFSSCGDDDDTGYLPPSQAIQDLLKRMKV
"""
    parsed_a3m = af_integration.parse_a3m(a3m_string)
    print("Number of sequences in parsed A3M:", len(parsed_a3m[0]))

    hhr_string = """
HHsearch 1.5
Query         query
Match_columns 50
No_of_seqs    1 out of 1
Neff          1.0
Searched_HMMs 85718

No 1
>1aab_A
Probab=99.96 E-value=5.5e-35 Score=146.42 Aligned_cols=50 Identities=100% Similarity=1.542 Sum_probs=49.4

Q query            1 MKFLKFSLLTAVLLSVVFAFSSCGDDDDTGYLPPSQAIQDLLKRMKV   50 (50)
Q Consensus        1 mkflkfslltavllsvvfafsscgddddtgylppsqaiqdllkrmkv   50 (50)
                     |||||||||||||||||||||||||||||||||||||||||||||||||
T Consensus        1 mkflkfslltavllsvvfafsscgddddtgylppsqaiqdllkrmkv   50 (50)
T 1aab_A           1 MKFLKFSLLTAVLLSVVFAFSSCGDDDDTGYLPPSQAIQDLLKRMKV   50 (50)
T ss_dssp             CCHHHHHHHHHHHHHHCCCCCCCCCCCHHHHHHHHHHHHHHHHHHHH
T ss_pred             CCHHHHHHHHHHHHHHCCCCCCCCCCCHHHHHHHHHHHHHHHHHHHH
Confidence            9999999999999999999999999999999999999999999999
"""
    parsed_hhr = af_integration.parse_hhr(hhr_string)
    print("Number of HHSearch results:", len(parsed_hhr))
    if parsed_hhr:
        print("First HHSearch result keys:", parsed_hhr[0].keys())

    # Example usage of template-related methods
    template_hits = [
        {"name": "1xyz", "aligned_sequence": "MKF-KFSLLTAVLLSVVFAFSSCGDDDDTGYLPPSQAIQDLLKRMKV"}
    ]
    template_features = af_integration.create_template_features(
        query_sequence=sequence,
        hits=template_hits
    )
    print("Template features keys:", template_features.keys())

    # Example of getting template hits
    mmcif_dir = "/path/to/mmcif/files"  # Replace with actual path
    template_hits = af_integration.get_template_hits(sequence, mmcif_dir)
    print("Number of template hits:", len(template_hits))
    if template_hits:
        print("First template hit:", template_hits[0])

    # Example of creating template features
    template_features = af_integration.create_template_features(sequence, template_hits)
    print("Template features keys:", template_features.keys())

    # Example usage of run_jackhmmer
    input_fasta_path = "/path/to/input.fasta"  # Replace with actual path
    database_path = "/path/to/jackhmmer/database"  # Replace with actual path
    jackhmmer_binary_path = "/path/to/jackhmmer"  # Replace with actual path

    try:
        jackhmmer_result = af_integration.run_jackhmmer(
            input_fasta_path=input_fasta_path,
            database_path=database_path,
            jackhmmer_binary_path=jackhmmer_binary_path
        )
        print("Jackhmmer search completed successfully")
        print("Number of hits:", len(jackhmmer_result[0]))
        print("Jackhmmer output:", jackhmmer_result[1][:100] + "...")  # Print first 100 characters
    except Exception as e:
        print(f"Error running Jackhmmer: {str(e)}")
