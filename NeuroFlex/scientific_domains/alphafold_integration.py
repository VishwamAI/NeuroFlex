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

# alphafold_integration.py

import jax
import jax.numpy as jnp
import haiku as hk
from typing import List, Dict, Any, Tuple, Union
from alphafold.model import config, modules, features, modules_multimer
from alphafold.model.config import CONFIG, CONFIG_MULTIMER, CONFIG_DIFFS
from alphafold.common import protein, confidence, residue_constants
from alphafold.data import pipeline, templates, msa_identifiers, parsers
from alphafold.data.tools import hhblits, jackhmmer, hhsearch, hmmsearch
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import logging
import copy
import ml_collections
from unittest.mock import MagicMock
import importlib
import importlib.metadata
import openmm
import openmm.app as app
import openmm.unit as unit
import os
import sys
import re
import random
import numpy as np
import tempfile
import pickle
import traceback

# Add the AlphaFold directory to the Python path
alphafold_path = os.environ.get('ALPHAFOLD_PATH')
if not alphafold_path:
    print("Error: ALPHAFOLD_PATH environment variable is not set.")
    print("Please set the ALPHAFOLD_PATH environment variable to the AlphaFold directory.")
    sys.exit(1)

sys.path.append(alphafold_path)

# Define the path to the AlphaFold parameters
ALPHAFOLD_PARAMS_DIR = "/home/ubuntu/NeuroFlex/alphafold_data/params/"

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
        self.model = None
        self.model_params = None
        self.config = None
        self.feature_dict = None
        self.msa_runner = None
        self.template_searcher = None
        self.alphafold_params = None

    def is_model_ready(self):
        """Check if the AlphaFold model is ready for predictions."""
        logging.info("Checking if AlphaFold model is ready")
        components = {
            'model': self.model,
            'model_params': self.model_params,
            'config': self.config,
            'feature_dict': self.feature_dict
        }
        for name, component in components.items():
            if component is None:
                logging.error(f"{name} is not initialized")
            else:
                logging.debug(f"{name} is initialized")
        is_ready = all(component is not None for component in components.values())
        logging.info(f"AlphaFold model ready: {is_ready}")
        return is_ready

    def prepare_features(self, sequence: str):
        """
        Prepare feature dictionary for AlphaFold prediction.

        Args:
            sequence (str): Amino acid sequence.

        Raises:
            ValueError: If the sequence is invalid.
        """
        if not sequence or not all(aa in 'ACDEFGHIKLMNPQRSTVWY' for aa in sequence.upper()):
            raise ValueError("Invalid amino acid sequence provided.")

        sequence_features = self.features_module.make_sequence_features(
            sequence=sequence,
            description="query",
            num_res=len(sequence)
        )
        msa = self._run_msa(sequence)
        msa_features = self.features_module.make_msa_features(msas=[msa])
        template_features = self._search_templates(sequence)

        self.feature_dict = {**sequence_features, **msa_features, **template_features}

    def _search_templates(self, sequence: str) -> Dict[str, Any]:
        """Search for templates and prepare features."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            SeqIO.write(SeqRecord(Seq(sequence), id="query"), temp_file, "fasta")
            temp_file_path = temp_file.name

        try:
            hits = self.template_searcher.query(temp_file_path)
            templates_result = self.templates_module.TemplateHitFeaturizer(
                mmcif_dir=os.path.join(alphafold_path, "alphafold", "data", "pdb_mmcif"),
                max_template_date="2021-11-01",
                max_hits=20,
                kalign_binary_path="kalign"
            ).get_templates(query_sequence=sequence, hits=hits)
            return templates_result.features
        finally:
            os.remove(temp_file_path)

    def _run_msa(self, sequence: str) -> List[Tuple[str, str]]:
        """Run Multiple Sequence Alignment (MSA) for the given protein sequence."""
        if self.msa_runner is None:
            database_path = os.environ.get('JACKHMMER_DATABASE_PATH', '/path/to/jackhmmer/database')
            binary_path = os.environ.get('JACKHMMER_BINARY_PATH', '/usr/bin/jackhmmer')
            try:
                self.msa_runner = self.jackhmmer_module.Jackhmmer(
                    binary_path=binary_path,
                    database_path=database_path
                )
            except Exception as e:
                logging.error(f"Failed to initialize Jackhmmer: {str(e)}")
                return [("query", sequence)]

        try:
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
                SeqIO.write(SeqRecord(Seq(sequence), id="query"), temp_file, "fasta")
                temp_file_path = temp_file.name

            result = self.msa_runner.query(temp_file_path)
            return [("query", sequence)] + [(hit.name, hit.sequence) for hit in result.hits]
        except Exception as e:
            logging.error(f"Error running MSA: {str(e)}")
            return [("query", sequence)]
        finally:
            if 'temp_file_path' in locals():
                try:
                    os.remove(temp_file_path)
                except Exception as e:
                    logging.warning(f"Failed to remove temporary file: {str(e)}")

    def setup_model(self, model_params: Dict[str, Any] = None, use_cpu: bool = False, use_quantum: bool = False):
        """Set up the AlphaFold model with given parameters."""
        try:
            logging.info("Starting AlphaFold model setup")
            if model_params is None:
                model_params = {'max_recycling': 3, 'model_name': 'model_1'}
            logging.info(f"Setting up AlphaFold model with parameters: {model_params}")
            logging.info(f"Using CPU: {use_cpu}")
            logging.info(f"Using Quantum: {use_quantum}")

            # Initialize the config
            model_name = model_params.get('model_name', 'model_1')
            base_config = CONFIG_MULTIMER if 'multimer' in model_name else CONFIG
            logging.info(f"Using base config for model: {model_name}")

            # Create a ConfigDict from the base config and update with model-specific differences
            config = ml_collections.ConfigDict(base_config)
            if model_name in CONFIG_DIFFS:
                config.update_from_flattened_dict(CONFIG_DIFFS[model_name])
                logging.info(f"Updated config with model-specific differences for {model_name}")

            # Ensure global_config is present and correctly initialized
            if 'global_config' not in config:
                config.global_config = ml_collections.ConfigDict({
                    'deterministic': False,
                    'subbatch_size': 4,
                    'use_remat': False,
                    'zero_init': True,
                    'eval_dropout': False,
                })

            # CPU-specific configurations
            if use_cpu:
                config.global_config.update({
                    'use_cpu': True,
                    'precision': jax.lax.Precision.DEFAULT,
                    'subbatch_size': 1,  # Reduce subbatch size for CPU
                })
                logging.info("Applied CPU-specific configurations")

            # Quantum-specific configurations
            if use_quantum:
                config.global_config.update({
                    'use_quantum': True,
                    'quantum_circuit_depth': 2,
                    'quantum_measurement_shots': 1000,
                })
                logging.info("Applied Quantum-specific configurations")

            config.update(model_params)
            self.config = config
            logging.info("Configuration setup completed")
            logging.debug(f"Final config: {self.config}")

            # Load AlphaFold parameters
            params_file = os.path.join(ALPHAFOLD_PARAMS_DIR, f"params_{model_name}.npz")
            if not os.path.exists(params_file):
                logging.error(f"AlphaFold parameters file not found: {params_file}")
                raise FileNotFoundError(f"AlphaFold parameters file not found: {params_file}")

            def convert_to_jax_array(item):
                if isinstance(item, np.ndarray):
                    return jnp.array(item)
                elif isinstance(item, dict):
                    return {k: convert_to_jax_array(v) for k, v in item.items()}
                elif isinstance(item, list):
                    return [convert_to_jax_array(v) for v in item]
                else:
                    return item

            logging.info(f"Loading AlphaFold parameters from {params_file}")
            with np.load(params_file) as loaded_params:
                self.alphafold_params = convert_to_jax_array(dict(loaded_params))
            logging.info(f"Loaded AlphaFold parameters from {params_file}")
            logging.debug(f"AlphaFold parameters types: {jax.tree_map(lambda x: type(x).__name__, self.alphafold_params)}")
            logging.debug(f"AlphaFold parameters shapes: {jax.tree_map(lambda x: x.shape if hasattr(x, 'shape') else None, self.alphafold_params)}")

            # Initialize the model using hk.transform
            def _init_alphafold(batch, config):
                model = self.modules_module.AlphaFold(config)
                return model(batch)

            model_creator = hk.transform(_init_alphafold)
            rng = jax.random.PRNGKey(0)
            dummy_batch = {
                'aatype': jnp.zeros((1, 50), dtype=jnp.int32),
                'residue_index': jnp.arange(50)[None],
                'seq_length': jnp.array([50], dtype=jnp.int32),
                'is_distillation': jnp.array(0, dtype=jnp.int32),
            }
            logging.info("Initializing model with dummy batch")
            logging.debug(f"Dummy batch types: {jax.tree_map(lambda x: type(x).__name__, dummy_batch)}")
            logging.debug(f"Dummy batch shapes: {jax.tree_map(lambda x: x.shape if hasattr(x, 'shape') else None, dummy_batch)}")

            try:
                logging.info("Initializing model parameters")
                with jax.default_device(jax.devices('cpu')[0] if use_cpu else None):
                    self.model_params = model_creator.init(rng, dummy_batch, config=self.config)
                logging.info("Model parameters initialized successfully")
                logging.debug(f"Initial model_params structure: {jax.tree_map(lambda x: x.shape if hasattr(x, 'shape') else None, self.model_params)}")
            except Exception as init_error:
                logging.error(f"Error during model initialization: {str(init_error)}")
                logging.debug(f"Config used for initialization: {self.config}")
                logging.debug(f"Dummy batch used for initialization: {dummy_batch}")
                raise ValueError(f"Failed to initialize AlphaFold model: {str(init_error)}")

            # Log the structure and types of model_params and alphafold_params
            logging.debug(f"Structure of model_params: {jax.tree_map(lambda x: x.shape if hasattr(x, 'shape') else None, self.model_params)}")
            logging.debug(f"Structure of alphafold_params: {jax.tree_map(lambda x: x.shape if hasattr(x, 'shape') else None, self.alphafold_params)}")
            logging.debug(f"Types in model_params: {jax.tree_map(lambda x: type(x).__name__, self.model_params)}")
            logging.debug(f"Types in alphafold_params: {jax.tree_map(lambda x: type(x).__name__, self.alphafold_params)}")

            # Integrate loaded AlphaFold parameters
            try:
                logging.info("Merging model parameters with AlphaFold parameters")

                # Validate parameter compatibility
                def validate_params(params1, params2, path=""):
                    if isinstance(params1, dict) and isinstance(params2, dict):
                        for k in params1:
                            if k not in params2:
                                logging.warning(f"Key '{path + k}' present in model_params but not in alphafold_params")
                            else:
                                validate_params(params1[k], params2[k], path + k + ".")
                        for k in params2:
                            if k not in params1:
                                logging.warning(f"Key '{path + k}' present in alphafold_params but not in model_params")
                    elif isinstance(params1, jnp.ndarray) and isinstance(params2, jnp.ndarray):
                        if params1.shape != params2.shape:
                            logging.warning(f"Mismatched shapes at {path}: {params1.shape} vs {params2.shape}")
                    elif type(params1) != type(params2):
                        logging.warning(f"Mismatched types at {path}: {type(params1)} vs {type(params2)}")

                logging.info("Validating parameter compatibility")
                validate_params(self.model_params, self.alphafold_params)
                logging.info("Parameter compatibility validation completed")

                # Merge parameters
                def merge_params(params1, params2):
                    if isinstance(params1, dict) and isinstance(params2, dict):
                        return {k: merge_params(params1.get(k), params2.get(k)) for k in set(params1) | set(params2)}
                    elif isinstance(params1, jnp.ndarray) and isinstance(params2, jnp.ndarray):
                        return params2 if params1.shape == params2.shape else params1
                    else:
                        return params2 if params2 is not None else params1

                merged_params = merge_params(self.model_params, self.alphafold_params)

                # Log types and shapes after merging
                logging.debug("Types and shapes after merging:")
                logging.debug(f"merged_params: {jax.tree_map(lambda x: (type(x).__name__, x.shape if hasattr(x, 'shape') else None), merged_params)}")

                self.model_params = merged_params
                logging.info("Parameters merged successfully")

                # Log final types and shapes
                logging.debug("Final types and shapes:")
                logging.debug(f"model_params: {jax.tree_map(lambda x: (type(x).__name__, x.shape if hasattr(x, 'shape') else None), self.model_params)}")

            except Exception as merge_error:
                logging.error(f"Error merging parameters: {str(merge_error)}")
                logging.error(f"model_params structure: {jax.tree_util.tree_structure(self.model_params)}")
                logging.error(f"alphafold_params structure: {jax.tree_util.tree_structure(self.alphafold_params)}")
                logging.debug(f"model_params keys: {jax.tree_util.tree_leaves(self.model_params)}")
                logging.debug(f"alphafold_params keys: {jax.tree_util.tree_leaves(self.alphafold_params)}")
                raise ValueError(f"Failed to integrate AlphaFold parameters: {str(merge_error)}")

            self.model = model_creator.apply
            logging.info("Model initialization completed")

            # Initialize MSA runner and template searcher
            jackhmmer_binary_path = os.environ.get('JACKHMMER_BINARY_PATH', '/usr/bin/jackhmmer')
            hhblits_binary_path = os.environ.get('HHBLITS_BINARY_PATH', '/usr/bin/hhblits')
            jackhmmer_database_path = os.environ.get('JACKHMMER_DATABASE_PATH', '/path/to/jackhmmer/database')
            hhblits_database_path = os.environ.get('HHBLITS_DATABASE_PATH', '/path/to/hhblits/database')

            if jackhmmer_database_path == '/path/to/jackhmmer/database':
                logging.warning(f"JACKHMMER_DATABASE_PATH not set. Using default path: {jackhmmer_database_path}")
            if hhblits_database_path == '/path/to/hhblits/database':
                logging.warning(f"HHBLITS_DATABASE_PATH not set. Using default path: {hhblits_database_path}")

            logging.info("Initializing MSA runner and template searcher")
            self.msa_runner = self.jackhmmer_module.Jackhmmer(binary_path=jackhmmer_binary_path, database_path=jackhmmer_database_path)
            self.template_searcher = self.hhblits_module.HHBlits(binary_path=hhblits_binary_path, databases=[hhblits_database_path])
            logging.info("MSA runner and template searcher initialized")

            # Initialize feature_dict with an empty dictionary
            self.feature_dict = {}
            logging.info("Initialized feature_dict with an empty dictionary")

            # Initialize quantum circuit if quantum mode is enabled
            if use_quantum:
                self.quantum_circuit = self._initialize_quantum_circuit()
                logging.info("Quantum circuit initialized")

            # Verify that the model is set up correctly
            logging.info("Verifying model setup")
            if not self.is_model_ready():
                logging.error("Failed to set up AlphaFold model correctly")
                logging.debug(f"Model state: model={self.model}, model_params={bool(self.model_params)}, config={bool(self.config)}, feature_dict={bool(self.feature_dict)}")
                raise ValueError("Failed to set up AlphaFold model correctly.")

            logging.info("AlphaFold model set up successfully.")
        except Exception as e:
            logging.error(f"Error setting up AlphaFold model: {str(e)}")
            logging.debug(f"Stack trace: {traceback.format_exc()}")
            raise ValueError(f"Failed to set up AlphaFold model: {str(e)}")

    def get_plddt_bands(self):
        """
        Returns the pLDDT confidence bands used in AlphaFold.
        """
        return self.confidence_module.PLDDT_BANDS

    def predict_structure(self):
        """
        Predict protein structure using AlphaFold and refine with OpenMM.

        Returns:
            protein.Protein: Predicted and refined protein structure.

        Raises:
            ValueError: If the model or features are not set up.
        """
        if not self.is_model_ready():
            raise ValueError("Model or features not set up. Call setup_model() and prepare_features() first.")

        # Predict structure using AlphaFold
        prediction_result = self.model({'params': self.model_params}, jax.random.PRNGKey(0), self.config, **self.feature_dict)
        predicted_protein = self.protein_module.from_prediction(prediction_result)

        # Set up and run OpenMM simulation for refinement
        self.setup_openmm_simulation(predicted_protein)
        if self.openmm_simulation:
            # Run a short simulation to refine the structure
            self.openmm_simulation.minimizeEnergy()
            self.openmm_simulation.step(1000)  # Run for 1000 steps

            # Get refined positions
            refined_positions = self.openmm_simulation.context.getState(getPositions=True).getPositions(asNumpy=True)

            # Update the predicted protein with refined positions
            for i, residue in enumerate(predicted_protein.residue_index):
                predicted_protein.atom_positions[residue] = refined_positions[i].value_in_unit(self.unit_module.angstrom)

        return predicted_protein

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

    def get_plddt_scores(self):
        """
        Get pLDDT (predicted LDDT) scores for the predicted structure.

        Returns:
            np.ndarray: Array of pLDDT scores.

        Raises:
            ValueError: If the model or features are not set up.
        """
        if not self.is_model_ready():
            raise ValueError("Model or features not set up. Call setup_model() and prepare_features() first.")

        prediction_result = self.model({'params': self.model_params}, jax.random.PRNGKey(0), self.config, **self.feature_dict)
        logits = prediction_result['predicted_lddt']['logits']
        plddt_scores = self.confidence_module.compute_plddt(logits)
        return np.array(plddt_scores).flatten()

    def get_predicted_aligned_error(self):
        """
        Get predicted aligned error for the structure.

        Returns:
            np.ndarray: 2D array of predicted aligned errors.

        Raises:
            ValueError: If the model or features are not set up, or if the output is invalid.
        """
        if not self.is_model_ready():
            raise ValueError("Model or features not set up. Call setup_model() and prepare_features() first.")

        prediction_result = self.model({'params': self.model_params}, jax.random.PRNGKey(0), self.config, **self.feature_dict)

        if 'predicted_aligned_error' not in prediction_result:
            raise ValueError("Predicted aligned error not found in model output.")

        pae_output = prediction_result['predicted_aligned_error']

        if isinstance(pae_output, dict):
            if 'logits' in pae_output and 'breaks' in pae_output:
                try:
                    pae = self.confidence_module.compute_predicted_aligned_error(pae_output['logits'], pae_output['breaks'])
                except Exception as e:
                    raise ValueError(f"Error computing predicted aligned error: {str(e)}")
            else:
                raise ValueError("Invalid structure of predicted aligned error in model output: missing 'logits' or 'breaks'.")
        elif isinstance(pae_output, (np.ndarray, list)):
            pae = np.array(pae_output)
        else:
            raise ValueError(f"Invalid type for predicted aligned error: {type(pae_output)}. Expected dict, list, or numpy.ndarray.")

        if pae.size == 0:
            raise ValueError("Computed PAE is empty.")

        pae = np.atleast_1d(pae)  # Convert to at least 1D array

        if pae.ndim == 1:
            # Handle 1D array
            size = int(np.sqrt(pae.size))
            if size * size == pae.size:
                pae = pae.reshape(size, size)
            else:
                # If perfect square reshaping is not possible, use the closest square size
                size = int(np.ceil(np.sqrt(pae.size)))
                pae = np.pad(pae, (0, size*size - pae.size), mode='constant', constant_values=np.nan)
                pae = pae.reshape(size, size)
        elif pae.ndim > 2:
            raise ValueError(f"Invalid PAE shape. Expected 1D or 2D array, got shape {pae.shape}")

        if pae.shape[0] != pae.shape[1]:
            raise ValueError(f"Invalid PAE shape. Expected square array, got shape {pae.shape}")

        return pae

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

    def run_alphamissense_analysis(self, sequence: str, variant: str) -> Dict[str, float]:
        """
        Run AlphaMissense analysis on the given sequence and variant.

        Args:
            sequence (str): The amino acid sequence to analyze.
            variant (str): The variant in the format 'OriginalAA{Position}NewAA' (e.g., 'G56A').

        Returns:
            Dict[str, float]: A dictionary containing 'pathogenic_score' and 'benign_score'.

        Raises:
            ValueError: If the input is invalid.
        """
        # Input validation
        if not isinstance(sequence, str):
            raise ValueError("Invalid input type for sequence. Expected str, got {type(sequence).__name__}.")
        if not isinstance(variant, str):
            raise ValueError("Invalid input type for variant. Expected str, got {type(variant).__name__}.")
        if not sequence:
            raise ValueError("Empty sequence provided. Please provide a valid amino acid sequence.")
        if not all(aa in 'ACDEFGHIKLMNPQRSTVWY' for aa in sequence.upper()):
            raise ValueError("Invalid amino acid(s) found in sequence.")

        # Validate variant format
        if not re.match(r'^[A-Z]\d+[A-Z]$', variant):
            raise ValueError("Invalid variant format. Use 'OriginalAA{Position}NewAA' (e.g., 'G56A').")

        original_aa, position, new_aa = variant[0], int(variant[1:-1]), variant[-1]
        if position < 1 or position > len(sequence):
            raise ValueError("Invalid variant position.")
        if sequence[position - 1] != original_aa:
            raise ValueError(f"Original amino acid in variant ({original_aa}) does not match sequence at position {position} ({sequence[position - 1]}).")
        if new_aa not in 'ACDEFGHIKLMNPQRSTVWY':
            raise ValueError(f"Invalid new amino acid in variant: {new_aa}")

        # Placeholder for actual AlphaMissense analysis
        # In a real implementation, this would call the AlphaMissense model
        pathogenic_score = random.uniform(0, 1)
        benign_score = 1 - pathogenic_score

        return {
            'pathogenic_score': pathogenic_score,
            'benign_score': benign_score
        }

    def run_alphaproteo_analysis(self, sequence: str) -> Dict[str, List[Union[str, float]]]:
        """
        Run AlphaProteo analysis on the given sequence.

        Args:
            sequence (str): The amino acid sequence to analyze.

        Returns:
            Dict[str, List[Union[str, float]]]: A dictionary containing 'novel_proteins' and 'binding_affinities'.

        Raises:
            ValueError: If the input is invalid.
        """
        # Input validation
        if not isinstance(sequence, str):
            raise ValueError("Invalid input type. Sequence must be a string.")
        if not sequence:
            raise ValueError("Empty sequence provided. Please provide a valid amino acid sequence.")
        if not all(aa in 'ACDEFGHIKLMNPQRSTVWY' for aa in sequence.upper()):
            raise ValueError("Invalid amino acid(s) found in sequence.")
        if len(sequence) < 10:
            raise ValueError("Sequence is too short. Minimum length is 10 amino acids.")
        if len(sequence) > 2000:
            raise ValueError("Sequence is too long. Maximum length is 2000 amino acids.")

        # Placeholder for actual AlphaProteo analysis
        # In a real implementation, this would call the AlphaProteo model
        novel_proteins = [''.join(random.choices('ACDEFGHIKLMNPQRSTVWY', k=len(sequence))) for _ in range(3)]
        binding_affinities = [random.uniform(0, 1) for _ in range(3)]

        return {
            'novel_proteins': novel_proteins,
            'binding_affinities': binding_affinities
        }

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

    def get_predicted_aligned_error(self):
        """
        Get the predicted aligned error from the AlphaFold model.

        Returns:
            jnp.ndarray: The predicted aligned error matrix.

        Raises:
            ValueError: If the model or features are not set up.
        """
        if not self.is_model_ready():
            raise ValueError("Model or features not set up. Call setup_model() and prepare_features() first.")

        prediction_result = self.model({'params': self.model_params}, jax.random.PRNGKey(0), self.config, **self.feature_dict)
        return prediction_result['predicted_aligned_error']

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
