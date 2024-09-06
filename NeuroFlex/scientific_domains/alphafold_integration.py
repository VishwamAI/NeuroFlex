import jax
import jax.numpy as jnp
import numpy as np
import haiku as hk
from typing import List, Dict, Any, Tuple
# TODO: Resolve AlphaFold import issues and reintegrate functionality
from alphafold.model import modules_multimer, config
from alphafold.model.config import CONFIG, CONFIG_MULTIMER, CONFIG_DIFFS
from alphafold.common import protein
from alphafold.data import pipeline, templates
from alphafold.data.tools import hhblits, jackhmmer
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import logging
import copy
import ml_collections
from unittest.mock import MagicMock

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global variable to track which version of SCOPData is being used
USING_BIO_SCOP_DATA = False

# Conditionally import SCOPData
try:
    from Bio.Data import SCOPData
    USING_BIO_SCOP_DATA = True
    logging.info("Using SCOPData from Bio.Data")
except ImportError:
    # Fallback SCOPData
    class SCOPData:
        protein_letters_3to1 = {
            'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
            'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
            'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
            'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
        }
    USING_BIO_SCOP_DATA = False
    logging.warning("Failed to import SCOPData from Bio.Data. Using fallback SCOPData in alphafold_integration.py")

# Export AlphaFoldIntegration, SCOPData, and USING_BIO_SCOP_DATA for use in other modules
__all__ = ['AlphaFoldIntegration', 'SCOPData', 'USING_BIO_SCOP_DATA']

class AlphaFoldIntegration:
    def __init__(self):
        self.model = None
        self.model_params = None
        self.feature_dict = None
        self.msa_runner = None
        self.template_searcher = None
        self.config = None  # Will be initialized in setup_model
        logging.info("AlphaFoldIntegration initialized")

    def setup_model(self, model_params: Dict[str, Any] = None):
        """
        Set up the AlphaFold model with given parameters.

        Args:
            model_params (Dict[str, Any], optional): Parameters for the AlphaFold model.
                If None, default parameters will be used.
        """
        logging.info("Setting up AlphaFold model")

        try:
            if model_params is None:
                model_params = {'max_recycling': 3, 'model_name': 'model_1'}

            # Initialize the config
            model_name = model_params.get('model_name', 'model_1')
            if 'multimer' in model_name:
                base_config = copy.deepcopy(CONFIG_MULTIMER)
            else:
                base_config = copy.deepcopy(CONFIG)

            # Update the base config with model-specific differences
            if model_name in CONFIG_DIFFS:
                base_config.update_from_flattened_dict(CONFIG_DIFFS[model_name])

            # Ensure global_config is present and correctly initialized
            if 'global_config' not in base_config:
                base_config.global_config = ml_collections.ConfigDict({
                    'deterministic': False,
                    'subbatch_size': 4,
                    'use_remat': False,
                    'zero_init': True,
                    'eval_dropout': False,
                })

            # Update config with any additional parameters
            base_config.update(model_params)

            self.config = base_config
            logging.debug(f"Config initialized: {self.config}")

            def create_model(config):
                logging.debug("Creating AlphaFold model")
                try:
                    model = modules_multimer.AlphaFold(config)
                    def apply(params, rng, inputs):
                        logging.debug("Applying model with inputs")
                        try:
                            # Process inputs
                            processed_inputs = {
                                'aatype': inputs.get('aatype'),
                                'residue_index': inputs.get('residue_index'),
                                'seq_mask': inputs.get('seq_mask'),
                                'msa_feat': inputs.get('msa_feat'),  # Only use msa_feat, remove fallback to msa
                                'msa_mask': inputs.get('msa_mask'),
                                'num_alignments': jnp.sum(inputs.get('msa_mask', jnp.array([0])), axis=0),
                                'template_aatype': inputs.get('template_aatype'),
                                'template_all_atom_masks': inputs.get('template_all_atom_masks'),
                                'template_all_atom_positions': inputs.get('template_all_atom_positions'),
                                'template_mask': jnp.any(inputs.get('template_all_atom_masks', jnp.array([0])), axis=-1),
                                'template_pseudo_beta_mask': jnp.any(inputs.get('template_all_atom_masks', jnp.array([0]))[:, :, :, 1], axis=-1),
                                'template_pseudo_beta': inputs.get('template_all_atom_positions', jnp.zeros((1, 1, 1, 37, 3)))[:, :, :, 1, :],
                                'is_distillation': inputs.get('is_distillation', jnp.array(0))
                            }

                            # Ensure all inputs are JAX arrays
                            for key, value in processed_inputs.items():
                                if value is not None and not isinstance(value, jnp.ndarray):
                                    processed_inputs[key] = jnp.array(value)

                            # Check input shapes
                            expected_shapes = {
                                'aatype': (None, None),
                                'residue_index': (None, None),
                                'seq_mask': (None, None),
                                'msa_feat': (None, None, None, None),  # Updated to match the new shape
                                'msa_mask': (None, None, None),
                                'num_alignments': (None,),
                                'template_aatype': (None, None, None),
                                'template_all_atom_masks': (None, None, None, None),
                                'template_all_atom_positions': (None, None, None, None, 3),
                                'template_mask': (None, None, None),
                                'template_pseudo_beta_mask': (None, None, None),
                                'template_pseudo_beta': (None, None, None, 3),
                                'is_distillation': ()
                            }
                            for input_name, expected_shape in expected_shapes.items():
                                if processed_inputs[input_name] is not None:
                                    if processed_inputs[input_name].ndim != len(expected_shape):
                                        raise ValueError(f"Input '{input_name}' has incorrect number of dimensions. Expected {len(expected_shape)}, got {processed_inputs[input_name].ndim}")

                            logging.debug(f"Processed inputs: {processed_inputs.keys()}")
                            logging.debug(f"msa_feat shape: {processed_inputs['msa_feat'].shape}")
                            return model.apply({'params': params}, **processed_inputs, rngs={'dropout': rng})
                        except Exception as e:
                            logging.error(f"Error applying model: {str(e)}", exc_info=True)
                            raise
                    return model, apply
                except Exception as e:
                    logging.error(f"Error creating AlphaFold model: {str(e)}", exc_info=True)
                    raise

            model_creator = hk.transform(create_model)

            rng = jax.random.PRNGKey(0)
            dummy_seq_length = 256
            dummy_input = {
                'msa': jnp.zeros((1, 1, dummy_seq_length), dtype=jnp.int32),
                'msa_mask': jnp.ones((1, 1, dummy_seq_length), dtype=jnp.float32),
                'seq_mask': jnp.ones((1, dummy_seq_length), dtype=jnp.float32),
                'residue_index': jnp.arange(dummy_seq_length)[None],
                'template_aatype': jnp.zeros((1, 1, dummy_seq_length), dtype=jnp.int32),
                'template_all_atom_masks': jnp.zeros((1, 1, dummy_seq_length, 37), dtype=jnp.float32),
                'template_all_atom_positions': jnp.zeros((1, 1, dummy_seq_length, 37, 3), dtype=jnp.float32),
                'template_sum_probs': jnp.zeros((1, 1), dtype=jnp.float32),
                'is_distillation': jnp.array(0, dtype=jnp.int32),
            }
            logging.debug("Initializing model parameters")
            try:
                self.model_params = model_creator.init(rng, self.config)
                self.model = model_creator.apply
            except Exception as e:
                logging.error(f"Error initializing model parameters: {str(e)}")
                raise ValueError(f"Failed to initialize AlphaFold model parameters: {str(e)}")

            # Test the model with dummy input
            logging.debug("Testing model with dummy input")
            try:
                _ = self.model(self.model_params, rng, self.config, **dummy_input)
                logging.info("AlphaFold model initialized and tested successfully")
            except Exception as e:
                logging.error(f"Error during model test: {str(e)}")
                logging.debug(f"Dummy input keys: {dummy_input.keys()}")
                raise ValueError(f"Failed to test AlphaFold model: {str(e)}")

            self.msa_runner = jackhmmer.Jackhmmer(binary_path=model_params.get('jackhmmer_binary_path', '/usr/bin/jackhmmer'))
            self.template_searcher = hhblits.HHBlits(binary_path=model_params.get('hhblits_binary_path', '/usr/bin/hhblits'))
            logging.info("MSA runner and template searcher initialized")

        except Exception as e:
            logging.error(f"Error in AlphaFold setup: {str(e)}", exc_info=True)
            raise ValueError(f"Failed to set up AlphaFold model: {str(e)}")

    def is_model_ready(self) -> bool:
        """Check if the AlphaFold model is ready for predictions."""
        return self.model is not None and self.model_params is not None

    def prepare_features(self, sequence: str):
        """
        Prepare feature dictionary for AlphaFold prediction.

        Args:
            sequence (str): Amino acid sequence.

        Returns:
            Dict: Feature dictionary for AlphaFold.
        """
        sequence_features = pipeline.make_sequence_features(sequence)
        msa = self._run_msa(sequence)
        msa_features = pipeline.make_msa_features(msas=[msa])
        template_features = self._search_templates(sequence)

        self.feature_dict = {**sequence_features, **msa_features, **template_features}

    def _run_msa(self, sequence: str) -> List[Tuple[str, str]]:
        """Run MSA and return results."""
        with open("temp.fasta", "w") as f:
            SeqIO.write(SeqRecord(Seq(sequence), id="query"), f, "fasta")
        result = self.msa_runner.query("temp.fasta")
        return [("query", sequence)] + [(hit.name, hit.sequence) for hit in result.hits]

    def _search_templates(self, sequence: str) -> Dict[str, Any]:
        """Search for templates and prepare features."""
        with open("temp.fasta", "w") as f:
            SeqIO.write(SeqRecord(Seq(sequence), id="query"), f, "fasta")
        hits = self.template_searcher.query("temp.fasta")
        templates_result = templates.TemplateHitFeaturizer(
            mmcif_dir="/path/to/mmcif/files",
            max_template_date="2021-11-01",
            max_hits=20,
            kalign_binary_path="/path/to/kalign"
        ).get_templates(query_sequence=sequence, hits=hits)
        return templates_result.features

    def predict_structure(self) -> protein.Protein:
        """
        Predict protein structure using AlphaFold.

        Returns:
            protein.Protein: Predicted protein structure.
        """
        if self.model is None or self.feature_dict is None:
            raise ValueError("Model or features not set up. Call setup_model() and prepare_features() first.")

        prediction_result = self.model({'params': self.model_params}, jax.random.PRNGKey(0), self.config, **self.feature_dict)
        return protein.from_prediction(prediction_result)

    def get_plddt_scores(self) -> jnp.ndarray:
        """
        Get pLDDT (predicted LDDT) scores for the predicted structure.

        Returns:
            jnp.ndarray: Array of pLDDT scores.
        """
        if self.model is None or self.feature_dict is None:
            raise ValueError("Model or features not set up. Call setup_model() and prepare_features() first.")

        prediction_result = self.model({'params': self.model_params}, jax.random.PRNGKey(0), self.config, **self.feature_dict)
        return prediction_result['plddt']

    def get_predicted_aligned_error(self) -> jnp.ndarray:
        """
        Get predicted aligned error for the structure.

        Returns:
            jnp.ndarray: 2D array of predicted aligned errors.
        """
        if self.model is None or self.feature_dict is None:
            raise ValueError("Model or features not set up. Call setup_model() and prepare_features() first.")

        prediction_result = self.model({'params': self.model_params}, jax.random.PRNGKey(0), self.config, **self.feature_dict)
        return prediction_result['predicted_aligned_error']

def sequence_to_onehot(sequence: str) -> np.ndarray:
    """
    Convert an amino acid sequence to a one-hot encoded matrix.

    Args:
        sequence (str): The amino acid sequence.

    Returns:
        np.ndarray: A 2D numpy array of shape (len(sequence), 20) representing the one-hot encoding.
    """
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    mapping = dict(zip(amino_acids, range(20)))

    onehot = np.zeros((len(sequence), 20), dtype=np.float32)
    for i, aa in enumerate(sequence):
        if aa in mapping:
            onehot[i, mapping[aa]] = 1.0

    return onehot

def onehot_to_sequence(onehot: np.ndarray) -> str:
    """
    Convert a one-hot encoded matrix back to an amino acid sequence.

    Args:
        onehot (np.ndarray): A 2D numpy array of shape (sequence_length, 20) representing the one-hot encoding.

    Returns:
        str: The amino acid sequence.
    """
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    return ''.join(amino_acids[np.argmax(encoding)] for encoding in onehot)

# Update __all__ to include the new functions
__all__ = ['AlphaFoldIntegration', 'SCOPData', 'USING_BIO_SCOP_DATA', 'sequence_to_onehot', 'onehot_to_sequence']
