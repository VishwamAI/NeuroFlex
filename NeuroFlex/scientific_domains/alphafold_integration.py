import jax
import jax.numpy as jnp
import haiku as hk
from typing import List, Dict, Any, Tuple
from alphafold.model import config, modules, data, model
from alphafold.model.config import CONFIG, CONFIG_MULTIMER, CONFIG_DIFFS
from alphafold.common import protein, confidence, residue_constants
from alphafold.data import pipeline, pipeline_multimer, templates
from alphafold.data.tools import hhblits, jackhmmer, hhsearch, hmmsearch
from alphafold.relax import relax
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
import re
import random

# Configure logging only if it hasn't been configured yet
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

from packaging import version

def check_version(package_name: str, expected_version: str) -> bool:
    try:
        installed_version = importlib.metadata.version(package_name)
        logger.info(f"{package_name} version: {installed_version}")
        print(f"DEBUG: {package_name} version string: '{installed_version}'")
        parsed_installed = version.parse(installed_version)
        parsed_expected = version.parse(expected_version)
        print(f"DEBUG: Parsed versions - Installed: {parsed_installed}, Expected: {parsed_expected}")
        if parsed_installed != parsed_expected:
            logger.warning(f"This integration was tested with {package_name} {expected_version}. You are using version {installed_version}. Some features may not work as expected.")
            return False
        logger.info(f"{package_name} version check passed.")
        return True
    except importlib.metadata.PackageNotFoundError:
        logger.error(f"Unable to determine {package_name} version. Make sure it's installed correctly.")
        return False

# Check versions and set flags for fallback strategies
ALPHAFOLD_COMPATIBLE = check_version("alphafold", "2.3.2")
JAX_COMPATIBLE = check_version("jax", "0.4.13")
print("DEBUG: Checking Haiku version")
HAIKU_COMPATIBLE = check_version("dm-haiku", "0.0.9")
print(f"DEBUG: HAIKU_COMPATIBLE = {HAIKU_COMPATIBLE}")
OPENMM_COMPATIBLE = check_version("openmm", "8.1.1")  # Add OpenMM version check

if not all([ALPHAFOLD_COMPATIBLE, JAX_COMPATIBLE, HAIKU_COMPATIBLE, OPENMM_COMPATIBLE]):
    logger.warning("This integration may have compatibility issues. Some features might be missing or work differently.")
    logger.info("Fallback strategies will be used where possible.")

# Set up OpenMM simulation environment
if OPENMM_COMPATIBLE:
    logger.info("Setting up OpenMM simulation environment")
    try:
        platform = openmm.Platform.getPlatformByName('CUDA')
        properties = {'CudaPrecision': 'mixed'}
    except Exception:
        logger.warning("CUDA platform not available for OpenMM. Falling back to CPU.")
        platform = openmm.Platform.getPlatformByName('CPU')
        properties = {}
else:
    logger.warning("OpenMM not compatible. Some molecular simulation features may be limited.")

try:
    from alphafold.model import modules_multimer
except ImportError:
    logging.warning("Failed to import alphafold.model.modules_multimer. Some functionality may be limited.")
    modules_multimer = MagicMock()

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global variable to track which version of PDBData is being used
USING_BIO_PDB_DATA = False

# Conditionally import PDBData
try:
    from Bio.Data import PDBData
    USING_BIO_PDB_DATA = True
    logging.info("Using PDBData from Bio.Data")
except ImportError:
    # Fallback PDBData
    class PDBData:
        protein_letters_3to1 = {
            'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
            'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
            'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
            'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
        }
    USING_BIO_PDB_DATA = False
    logging.warning("Failed to import PDBData from Bio.Data. Using fallback PDBData in alphafold_integration.py")

# Export AlphaFoldIntegration, PDBData, and USING_BIO_PDB_DATA for use in other modules
__all__ = ['AlphaFoldIntegration', 'PDBData', 'USING_BIO_PDB_DATA']

class AlphaFoldIntegration:
    def __init__(self):
        self.model = None
        self.model_params = None
        self.feature_dict = None
        self.msa_runner = None
        self.template_searcher = None
        self.config = None  # Will be initialized in setup_model
        self.openmm_system = None
        self.openmm_simulation = None
        self.openmm_integrator = None
        self.multimer_config = None
        self.relaxation_config = None
        logging.info("AlphaFoldIntegration initialized with OpenMM support")

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
                self.multimer_config = base_config
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
                logging.debug(f"Creating AlphaFold model with config: {config}")
                try:
                    model = modules.AlphaFold(config)
                    def apply(params, rng, config, **inputs):
                        logging.debug(f"Applying model with inputs: {inputs.keys()}")
                        try:
                            # Process inputs
                            processed_inputs = {}
                            required_inputs = {
                                'aatype': (None, None),
                                'residue_index': (None, None),
                                'seq_mask': (None, None),
                                'msa': (None, None, None),
                                'msa_mask': (None, None, None),
                                'num_alignments': (None,),
                                'template_aatype': (None, None, None),
                                'template_all_atom_masks': (None, None, None, None),
                                'template_all_atom_positions': (None, None, None, None, 3)
                            }

                            for input_name, expected_shape in required_inputs.items():
                                if input_name in inputs:
                                    processed_inputs[input_name] = inputs[input_name]
                                    logging.debug(f"Input '{input_name}' provided with shape: {inputs[input_name].shape}")
                                else:
                                    # Create dummy input if not provided
                                    processed_inputs[input_name] = jnp.zeros(expected_shape, dtype=jnp.float32)
                                    logging.warning(f"Input '{input_name}' not provided. Using dummy input with shape: {expected_shape}")

                                if processed_inputs[input_name].ndim != len(expected_shape):
                                    error_msg = f"Input '{input_name}' has incorrect number of dimensions. Expected {len(expected_shape)}, got {processed_inputs[input_name].ndim}"
                                    logging.error(error_msg)
                                    raise ValueError(error_msg)

                            # Special handling for 'msa' input
                            if isinstance(processed_inputs['msa'], str):
                                processed_inputs['msa'] = jnp.array([[ord(c) for c in processed_inputs['msa']]])
                                logging.debug("Converted string 'msa' input to numerical representation")

                            # Ensure 'num_alignments' is set correctly
                            processed_inputs['num_alignments'] = jnp.array([processed_inputs['msa'].shape[0]], dtype=jnp.int32)
                            logging.debug(f"Set 'num_alignments' to {processed_inputs['num_alignments']}")

                            logging.debug(f"Processed inputs: {processed_inputs.keys()}")
                            return model.apply({'params': params}, **processed_inputs, rngs={'dropout': rng})
                        except Exception as e:
                            logging.error(f"Error applying model: {str(e)}", exc_info=True)
                            raise
                    return model, apply
                except Exception as e:
                    logging.error(f"Error creating AlphaFold model: {str(e)}", exc_info=True)
                    raise

            # Transform the model creation function
            model_creator = hk.transform(create_model)

            # Initialize random number generator
            rng = jax.random.PRNGKey(0)

            # Create dummy input for model initialization
            dummy_seq_length = 256
            dummy_input = {
                'msa': jnp.zeros((1, 1, dummy_seq_length), dtype=jnp.int32),
                'msa_mask': jnp.ones((1, 1, dummy_seq_length), dtype=jnp.float32),
                'seq_mask': jnp.ones((1, dummy_seq_length), dtype=jnp.float32),
                'aatype': jnp.zeros((1, dummy_seq_length), dtype=jnp.int32),
                'residue_index': jnp.arange(dummy_seq_length)[None],
                'template_aatype': jnp.zeros((1, 1, dummy_seq_length), dtype=jnp.int32),
                'template_all_atom_masks': jnp.zeros((1, 1, dummy_seq_length, 37), dtype=jnp.float32),
                'template_all_atom_positions': jnp.zeros((1, 1, dummy_seq_length, 37, 3), dtype=jnp.float32),
            }

            logging.debug("Initializing model parameters")
            try:
                # Initialize model parameters
                self.model_params = model_creator.init(rng, self.config)
                self.model = model_creator.apply
                logging.info("AlphaFold model parameters initialized successfully")
            except Exception as e:
                logging.error(f"Error initializing model parameters: {str(e)}", exc_info=True)
                raise ValueError(f"Failed to initialize AlphaFold model parameters: {str(e)}")

            # Test the model with dummy input
            logging.debug("Testing model with dummy input")
            try:
                _ = self.model(self.model_params, rng, self.config, **dummy_input)
                logging.info("AlphaFold model initialized and tested successfully")
            except Exception as e:
                logging.error(f"Error during model test: {str(e)}")
                logging.debug(f"Dummy input keys: {dummy_input.keys()}")
                for key, value in dummy_input.items():
                    logging.debug(f"Dummy input '{key}' shape: {value.shape}")
                raise ValueError(f"Failed to test AlphaFold model: {str(e)}")

            self.msa_runner = jackhmmer.Jackhmmer(
                binary_path=model_params.get('jackhmmer_binary_path', '/usr/bin/jackhmmer'),
                database_path=model_params.get('database_path', '/path/to/default/database')
            )
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

        Raises:
            ValueError: If the sequence is invalid.
        """
        if not sequence or not all(aa in 'ACDEFGHIKLMNPQRSTVWY' for aa in sequence.upper()):
            raise ValueError("Invalid amino acid sequence provided.")

        sequence_features = pipeline.make_sequence_features(
            sequence=sequence,
            description="query",
            num_res=len(sequence)
        )
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

    def setup_openmm_simulation(self, protein: protein.Protein):
        """Set up OpenMM simulation for the predicted protein structure."""
        if not OPENMM_COMPATIBLE:
            logging.warning("OpenMM is not compatible. Skipping molecular dynamics simulation.")
            return

        try:
            # Convert AlphaFold protein to OpenMM topology and positions
            topology = app.Topology()
            chain = topology.addChain()
            positions = []

            for residue in protein.residue_index:
                omm_residue = topology.addResidue(protein.sequence[residue], chain)
                for atom_name, atom_position in zip(protein.atom_mask[residue], protein.atom_positions[residue]):
                    if atom_name:
                        topology.addAtom(atom_name, app.Element.getBySymbol(atom_name[0]), omm_residue)
                        positions.append(atom_position * unit.angstrom)

            # Create OpenMM system
            forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
            system = forcefield.createSystem(
                topology,
                nonbondedMethod=app.PME,
                nonbondedCutoff=1*unit.nanometer,
                constraints=app.HBonds
            )

            # Set up integrator and simulation
            integrator = openmm.LangevinMiddleIntegrator(300*unit.kelvin, 1/unit.picosecond, 0.002*unit.picoseconds)
            platform = openmm.Platform.getPlatformByName('CUDA')
            properties = {'CudaPrecision': 'mixed'}
            self.openmm_simulation = app.Simulation(topology, system, integrator, platform, properties)
            self.openmm_simulation.context.setPositions(positions)

            logging.info("OpenMM simulation set up successfully.")
        except Exception as e:
            logging.error(f"Error setting up OpenMM simulation: {str(e)}")
            self.openmm_simulation = None

    def predict_structure(self) -> protein.Protein:
        """
        Predict protein structure using AlphaFold and refine with OpenMM.

        Returns:
            protein.Protein: Predicted and refined protein structure.
        """
        if self.model is None or self.feature_dict is None:
            raise ValueError("Model or features not set up. Call setup_model() and prepare_features() first.")

        prediction_result = self.model({'params': self.model_params}, jax.random.PRNGKey(0), self.config, **self.feature_dict)
        predicted_protein = protein.from_prediction(prediction_result)

        # Set up and run OpenMM simulation for refinement
        self.setup_openmm_simulation(predicted_protein)
        if self.openmm_simulation:
            # Run a short simulation to refine the structure
            self.openmm_simulation.minimizeEnergy()
            self.openmm_simulation.step(1000)  # Run for 1000 steps

            # Get refined positions
            refined_positions = self.openmm_simulation.context.getState(getPositions=True).getPositions(asNumpy=True)

            # Validate refined positions
            if refined_positions is None or len(refined_positions) == 0:
                logging.warning("Refined positions are empty or None. Skipping structure refinement.")
                return predicted_protein

            if len(refined_positions) != len(predicted_protein.residue_index):
                logging.error(f"Mismatch in number of residues: {len(refined_positions)} refined vs {len(predicted_protein.residue_index)} original")
                raise ValueError("Refined positions do not match the original structure.")

            # Update the predicted protein with refined positions
            for i, residue in enumerate(predicted_protein.residue_index):
                try:
                    refined_pos = refined_positions[i].value_in_unit(unit.angstrom)
                    if refined_pos.shape != (5, 3):  # Assuming 5 atoms per residue and 3D coordinates
                        raise ValueError(f"Unexpected shape for refined position: {refined_pos.shape}")
                    predicted_protein.atom_positions[residue] = refined_pos
                except Exception as e:
                    logging.error(f"Error updating atom positions for residue {residue}: {str(e)}")
                    raise

        return predicted_protein

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

    def predict_multimer_structure(self, sequences: List[str]) -> protein.Protein:
        """
        Predict multimer protein structure using AlphaFold.

        Args:
            sequences (List[str]): List of amino acid sequences for each chain.

        Returns:
            protein.Protein: Predicted multimer protein structure.
        """
        if self.multimer_config is None:
            raise ValueError("Multimer configuration not set. Use a multimer model when calling setup_model().")

        # Prepare features for multimer prediction
        multimer_features = pipeline_multimer.make_multimer_features(sequences)

        # Run prediction
        prediction_result = self.model({'params': self.model_params}, jax.random.PRNGKey(0), self.multimer_config, **multimer_features)
        predicted_protein = protein.from_prediction(prediction_result, multimer=True)

        return predicted_protein

    def relax_structure(self, predicted_protein: protein.Protein) -> protein.Protein:
        """
        Relax the predicted protein structure using OpenMM.

        Args:
            predicted_protein (protein.Protein): Predicted protein structure.

        Returns:
            protein.Protein: Relaxed protein structure.
        """
        if not OPENMM_COMPATIBLE:
            logging.warning("OpenMM is not compatible. Skipping structure relaxation.")
            return predicted_protein

        amber_relaxer = relax.AmberRelaxation(
            max_iterations=0,
            tolerance=2.39,
            stiffness=10.0,
            exclude_residues=[],
            max_outer_iterations=20)

        relaxed_pdb, _, _ = amber_relaxer.process(prot=predicted_protein)
        relaxed_protein = protein.from_pdb_string(relaxed_pdb)

        return relaxed_protein

    def calculate_tm_score(self, predicted_protein: protein.Protein, native_protein: protein.Protein) -> float:
        """
        Calculate TM-score between predicted and native protein structures.

        Args:
            predicted_protein (protein.Protein): Predicted protein structure.
            native_protein (protein.Protein): Native (experimental) protein structure.

        Returns:
            float: TM-score between the two structures.
        """
        return confidence.tm_score(
            predicted_protein.atom_positions,
            native_protein.atom_positions,
            predicted_protein.residue_index,
            native_protein.residue_index)

    def get_residue_constants(self) -> Dict[str, Any]:
        """
        Get residue constants used in AlphaFold.

        Returns:
            Dict[str, Any]: Dictionary of residue constants.
        """
        return {
            'restype_order': residue_constants.restype_order,
            'restype_num': residue_constants.restype_num,
            'restypes': residue_constants.restypes,
            'atom_types': residue_constants.atom_types,
            'atom_order': residue_constants.atom_order,
        }

    def run_alphamissense_analysis(self, sequence: str, variant: str) -> Dict[str, float]:
        """
        Run AlphaMissense analysis on the given sequence and variant.

        Args:
            sequence (str): The protein sequence.
            variant (str): The variant to analyze (format: 'OriginalAA{Position}NewAA', e.g., 'G56A').

        Returns:
            Dict[str, float]: A dictionary containing the pathogenicity scores.
                              Keys: 'pathogenic_score', 'benign_score'

        Raises:
            ValueError: If the input sequence or variant is invalid.
        """
        # Validate sequence
        if not sequence:
            raise ValueError("Empty sequence provided. Please provide a valid amino acid sequence.")
        if not isinstance(sequence, str):
            raise ValueError("Invalid input type. Sequence must be a string.")
        sequence = sequence.upper()
        invalid_chars = set(sequence) - set('ACDEFGHIKLMNPQRSTVWY')
        if invalid_chars:
            raise ValueError(f"Invalid amino acid(s) found in sequence: {', '.join(invalid_chars)}. Only standard amino acids are allowed.")

        # Validate variant
        if not isinstance(variant, str):
            raise ValueError("Invalid input type. Variant must be a string.")
        if not re.match(r'^[A-Z]\d+[A-Z]$', variant):
            raise ValueError("Invalid variant format. Use 'OriginalAA{Position}NewAA' (e.g., 'G56A').")

        # Validate variant position and original amino acid
        original_aa, position_str, new_aa = variant[0], variant[1:-1], variant[-1]
        try:
            position = int(position_str)
        except ValueError:
            raise ValueError(f"Invalid position in variant: '{position_str}'. Must be an integer.")

        if position < 1 or position > len(sequence):
            raise ValueError(f"Invalid variant position: {position}. Must be between 1 and {len(sequence)}.")
        if sequence[position - 1] != original_aa:
            raise ValueError(f"Original amino acid in variant ({original_aa}) does not match sequence at position {position} ({sequence[position - 1]}).")
        if new_aa not in 'ACDEFGHIKLMNPQRSTVWY':
            raise ValueError(f"Invalid new amino acid in variant: '{new_aa}'. Must be a standard amino acid.")

        # TODO: Implement actual AlphaMissense analysis
        # This is a placeholder implementation
        pathogenic_score = random.uniform(0, 1)
        benign_score = 1 - pathogenic_score

        return {
            'pathogenic_score': pathogenic_score,
            'benign_score': benign_score
        }

    def run_alphaproteo_analysis(self, sequence: str) -> Dict[str, Any]:
        """
        Run AlphaProteo analysis on the given sequence to generate novel proteins.

        Args:
            sequence (str): The protein sequence to analyze.

        Returns:
            Dict[str, Any]: A dictionary containing the analysis results.
                            Keys: 'novel_proteins', 'binding_affinities'

        Raises:
            ValueError: If the input sequence is invalid.
        """
        if not sequence:
            raise ValueError("Empty sequence provided. Please provide a valid amino acid sequence.")

        if not isinstance(sequence, str):
            raise ValueError("Invalid input type. Sequence must be a string.")

        sequence = sequence.upper()
        valid_amino_acids = set('ACDEFGHIKLMNPQRSTVWY')
        invalid_chars = set(sequence) - valid_amino_acids
        if invalid_chars:
            raise ValueError(f"Invalid amino acid(s) found in sequence: {', '.join(invalid_chars)}. Only standard amino acids (ACDEFGHIKLMNPQRSTVWY) are allowed.")

        if len(sequence) < 20:
            raise ValueError(f"Sequence is too short. Minimum length is 20 amino acids, but got {len(sequence)}.")

        if len(sequence) > 2000:
            raise ValueError(f"Sequence is too long. Maximum length is 2000 amino acids, but got {len(sequence)}.")

        # TODO: Implement actual AlphaProteo analysis
        # This is a placeholder implementation
        novel_proteins = [
            ''.join(random.choices(tuple(valid_amino_acids), k=len(sequence)))
            for _ in range(3)
        ]
        binding_affinities = [random.uniform(0, 1) for _ in range(3)]

        return {
            'novel_proteins': novel_proteins,
            'binding_affinities': binding_affinities
        }
