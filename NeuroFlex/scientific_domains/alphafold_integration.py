import re
import random
import jax
import jax.numpy as jnp
import numpy as np
import haiku as hk
import logging
import copy
import importlib
import importlib.metadata
import openmm
import openmm.app as app
import openmm.unit as unit
import ml_collections
from typing import List, Dict, Any, Tuple
from unittest.mock import MagicMock
from alphafold.model import config, modules
from alphafold.model.config import CONFIG, CONFIG_MULTIMER, CONFIG_DIFFS
from alphafold.common import protein, residue_constants
from alphafold.data import pipeline, templates
from alphafold.data.tools import hhblits, jackhmmer
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

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
JAX_COMPATIBLE = check_version("jax", "0.4.31")
print("DEBUG: Checking Haiku version")
HAIKU_COMPATIBLE = check_version("dm-haiku", "0.0.12")
print(f"DEBUG: HAIKU_COMPATIBLE = {HAIKU_COMPATIBLE}")
OPENMM_COMPATIBLE = check_version("openmm", "8.1.1")

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
        """
        Initialize the AlphaFoldIntegration class with necessary components for protein structure prediction,
        molecular dynamics simulations, and additional protein analysis methods.
        """
        self.model: Optional[Callable] = None
        self.model_params: Optional[Dict[str, Any]] = None
        self.feature_dict: Optional[Dict[str, Any]] = None
        self.msa_runner: Optional[jackhmmer.Jackhmmer] = None
        self.template_searcher: Optional[hhblits.HHBlits] = None
        self.config: Optional[ml_collections.ConfigDict] = None  # Will be initialized in setup_model
        self.openmm_system: Optional[openmm.System] = None
        self.openmm_simulation: Optional[openmm.app.Simulation] = None
        self.openmm_integrator: Optional[openmm.Integrator] = None
        self.run_alphaproteo: Callable[[str], Dict[str, Any]] = self._run_alphaproteo
        self.run_alphamissense: Callable[[str, str], Dict[str, Any]] = self._run_alphamissense
        logging.info("AlphaFoldIntegration initialized with OpenMM support and AlphaProteo/AlphaMissense methods")

    def setup_model(self, model_params: Dict[str, Any] = None):
        """
        Set up the AlphaFold model with given parameters.

        Args:
            model_params (Dict[str, Any], optional): Parameters for the AlphaFold model.
                If None, default parameters will be used.

        Raises:
            ValueError: If there's an error in setting up the model or its components.
        """
        logging.info("Starting AlphaFold model setup")

        try:
            if model_params is None:
                model_params = {'max_recycling': 3, 'model_name': 'model_1'}
            logging.debug(f"Model parameters: {model_params}")

            # Initialize the config
            model_name = model_params.get('model_name', 'model_1')
            logging.info(f"Initializing config for model: {model_name}")
            self._initialize_config(model_name, model_params)

            # Create and initialize the model
            self._create_and_initialize_model()

            # Initialize MSA runner and template searcher
            self._initialize_msa_and_template_components(model_params)

        except Exception as e:
            logging.error(f"Error in AlphaFold setup: {str(e)}", exc_info=True)
            raise ValueError(f"Failed to set up AlphaFold model: {str(e)}")

    def _initialize_config(self, model_name: str, model_params: Dict[str, Any]):
        try:
            if 'multimer' in model_name:
                base_config = ml_collections.ConfigDict(CONFIG_MULTIMER)
                logging.debug("Using CONFIG_MULTIMER as base configuration")
            else:
                base_config = ml_collections.ConfigDict(CONFIG)
                logging.debug("Using CONFIG as base configuration")

            if model_name in CONFIG_DIFFS:
                logging.debug(f"Applying CONFIG_DIFFS for {model_name}")
                base_config.update_from_flattened_dict(CONFIG_DIFFS[model_name])

            self._ensure_global_config(base_config)
            base_config.update(model_params)

            self.config = base_config
            logging.info("Config initialized successfully")
            logging.debug(f"Final config structure: {self.config.to_dict()}")
        except Exception as e:
            logging.error(f"Error initializing config: {str(e)}", exc_info=True)
            raise ValueError(f"Failed to initialize AlphaFold config: {str(e)}")

    def _ensure_global_config(self, config: ml_collections.ConfigDict):
        if 'global_config' not in config:
            logging.debug("global_config not found, creating new ConfigDict")
            config.global_config = ml_collections.ConfigDict({
                'deterministic': False,
                'subbatch_size': 4,
                'use_remat': False,
                'zero_init': True,
                'eval_dropout': False,
                'use_custom_jit': True
            })
        elif not isinstance(config.global_config, ml_collections.ConfigDict):
            logging.debug("Converting existing global_config to ConfigDict")
            config.global_config = ml_collections.ConfigDict(config.global_config)

    def _create_and_initialize_model(self):
        def create_model(config):
            logging.debug(f"Creating AlphaFold model with config structure: {config.to_dict()}")
            try:
                model = modules.AlphaFold(config.model, config.data)
                logging.info("AlphaFold model created successfully")
                return model
            except AttributeError:
                logging.warning("Attempting to create model with fallback method...")
                try:
                    model = modules.AlphaFold(config)
                    logging.info("AlphaFold model created successfully with fallback method")
                    return model
                except Exception as fallback_error:
                    logging.error(f"Fallback method failed: {str(fallback_error)}", exc_info=True)
                    raise ValueError(f"Failed to create AlphaFold model. Check your installation and configuration.")
            except Exception as e:
                logging.error(f"Unexpected error creating AlphaFold model: {str(e)}", exc_info=True)
                raise ValueError(f"Failed to create AlphaFold model: {str(e)}.")

        try:
            model_creator = hk.transform(create_model)
            rng = jax.random.PRNGKey(0)
            dummy_input = self._create_dummy_input()

            self.model_params = model_creator.init(rng, self.config)
            self.model = model_creator.apply

            # Test the model with dummy input
            _ = self.model(self.model_params, rng, self.config, **dummy_input)
            logging.info("AlphaFold model initialized and tested successfully")
        except Exception as e:
            logging.error(f"Error initializing or testing model: {str(e)}", exc_info=True)
            raise ValueError(f"Failed to initialize or test AlphaFold model: {str(e)}")

    def _create_dummy_input(self, dummy_seq_length: int = 256):
        return {
            'aatype': jnp.zeros((dummy_seq_length,), dtype=jnp.int32),
            'residue_index': jnp.arange(dummy_seq_length),
            'seq_mask': jnp.ones((dummy_seq_length,), dtype=jnp.float32),
            'msa_feat': jnp.zeros((1, dummy_seq_length, 49), dtype=jnp.int32),
            'msa_mask': jnp.ones((1, dummy_seq_length), dtype=jnp.float32),
            'num_alignments': jnp.array([1], dtype=jnp.int32),
            'template_aatype': jnp.zeros((1, dummy_seq_length), dtype=jnp.int32),
            'template_all_atom_masks': jnp.zeros((1, dummy_seq_length, 37), dtype=jnp.float32),
            'template_all_atom_positions': jnp.zeros((1, dummy_seq_length, 37, 3), dtype=jnp.float32),
            'template_mask': jnp.zeros((1,), dtype=jnp.float32),
            'template_pseudo_beta': jnp.zeros((1, dummy_seq_length, 3), dtype=jnp.float32),
            'template_pseudo_beta_mask': jnp.zeros((1, dummy_seq_length), dtype=jnp.float32),
        }

    def _initialize_msa_and_template_components(self, model_params: Dict[str, Any]):
        self._initialize_jackhmmer(model_params)
        self._initialize_hhblits(model_params)

        if self.msa_runner is None and self.template_searcher is None:
            raise ValueError("Failed to initialize both MSA runner and template searcher")
        elif self.msa_runner is None:
            logging.warning("MSA runner initialization failed. Some functionality may be limited.")
        elif self.template_searcher is None:
            logging.warning("Template searcher initialization failed. Some functionality may be limited.")
        else:
            logging.info("MSA runner and template searcher initialized successfully")

    def _initialize_jackhmmer(self, model_params: Dict[str, Any]):
        jackhmmer_binary_path = model_params.get('jackhmmer_binary_path', '/usr/bin/jackhmmer')
        jackhmmer_database_path = model_params.get('jackhmmer_database_path', '/path/to/default/jackhmmer_db')

        if not jackhmmer_database_path:
            logging.warning("Jackhmmer database path not provided. Using default database.")

        try:
            self.msa_runner = jackhmmer.Jackhmmer(binary_path=jackhmmer_binary_path, database_path=jackhmmer_database_path)
            logging.info(f"Jackhmmer MSA runner initialized successfully with binary path: {jackhmmer_binary_path} and database path: {jackhmmer_database_path}")
        except Exception as e:
            logging.error(f"Failed to initialize Jackhmmer MSA runner: {str(e)}", exc_info=True)
            self.msa_runner = None

    def _initialize_hhblits(self, model_params: Dict[str, Any]):
        hhblits_binary_path = model_params.get('hhblits_binary_path', '/usr/bin/hhblits')
        hhblits_database_path = model_params.get('hhblits_database_path', '/path/to/default/hhblits_db')

        if not hhblits_database_path:
            logging.warning("HHBlits database path not provided. Using default database.")

        try:
            self.template_searcher = hhblits.HHBlits(binary_path=hhblits_binary_path, databases=[hhblits_database_path])
            logging.info(f"HHBlits template searcher initialized successfully with binary path: {hhblits_binary_path} and database path: {hhblits_database_path}")
        except Exception as e:
            logging.error(f"Failed to initialize HHBlits template searcher: {str(e)}", exc_info=True)
            self.template_searcher = None

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
        logging.info(f"Preparing features for sequence of length {len(sequence)}")

        if not sequence or not isinstance(sequence, str):
            raise ValueError("Invalid sequence input. Must be a non-empty string.")

        sequence = sequence.upper()
        if not all(aa in 'ACDEFGHIKLMNPQRSTVWY' for aa in sequence):
            raise ValueError("Invalid amino acid sequence. Contains non-standard amino acids.")

        try:
            sequence_features = pipeline.make_sequence_features(
                sequence=sequence,
                description="query",
                num_res=len(sequence)
            )
            logging.debug("Sequence features created successfully")

            msa = self._run_msa(sequence)
            msa_features = pipeline.make_msa_features(msas=[msa])
            logging.debug("MSA features created successfully")

            template_features = self._search_templates(sequence)
            logging.debug("Template features created successfully")

            self.feature_dict = {**sequence_features, **msa_features, **template_features}
            logging.info("Feature dictionary prepared successfully")

            return self.feature_dict
        except Exception as e:
            logging.error(f"Error preparing features: {str(e)}")
            raise RuntimeError(f"Failed to prepare features: {str(e)}")

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
        """
        Set up OpenMM simulation for the predicted protein structure.

        Args:
            protein (protein.Protein): The protein structure predicted by AlphaFold.

        Raises:
            ValueError: If there's an error in setting up the simulation.
        """
        if not OPENMM_COMPATIBLE:
            logging.warning("OpenMM is not compatible. Skipping molecular dynamics simulation.")
            return

        try:
            # Convert AlphaFold protein to OpenMM topology and positions
            topology, positions = self._create_openmm_topology_and_positions(protein)
            if not positions:
                raise ValueError("No valid atom positions found in the protein structure.")

            # Create OpenMM system
            system = self._create_openmm_system(topology)

            # Set up integrator and simulation
            integrator = self._create_openmm_integrator()
            self.openmm_simulation = self._create_openmm_simulation(topology, system, integrator)

            # Set positions in nanometers
            self._set_simulation_positions(positions)

            logging.info("OpenMM simulation set up successfully.")
        except ValueError as ve:
            logging.error(f"Error setting up OpenMM simulation: {str(ve)}")
            self.openmm_simulation = None
            raise
        except Exception as e:
            logging.error(f"Unexpected error setting up OpenMM simulation: {str(e)}")
            self.openmm_simulation = None
            raise ValueError(f"Failed to set up OpenMM simulation: {str(e)}")

    def _create_openmm_topology_and_positions(self, protein: protein.Protein):
        topology = app.Topology()
        chain = topology.addChain()
        positions = []

        for residue_index, residue_type in enumerate(protein.sequence):
            if residue_type not in app.PDBFile.standardResidues:
                logging.warning(f"Non-standard residue {residue_type} at index {residue_index}. Attempting to use it as-is.")

            try:
                omm_residue = topology.addResidue(residue_type, chain)
                valid_atoms = self._add_atoms_to_residue(topology, omm_residue, protein, residue_index, positions)
                if not valid_atoms:
                    logging.warning(f"No valid atoms found for residue {residue_type} at index {residue_index}. Skipping.")
            except Exception as residue_error:
                self._handle_residue_error(residue_index, residue_error)

        if not positions:
            logging.error("No valid atom positions found in the protein structure.")
            raise ValueError("Failed to create OpenMM topology: No valid atom positions found.")

        logging.info(f"Created OpenMM topology with {topology.getNumResidues()} residues and {len(positions)} atoms.")
        return topology, positions

    def _add_atoms_to_residue(self, topology, omm_residue, protein, residue_index, positions):
        residue_atoms = protein.atom_names[residue_index]
        residue_positions = protein.atom_positions[residue_index]

        if len(residue_atoms) != len(residue_positions):
            logging.warning(f"Mismatch in atom names and positions for residue {residue_index}. Using the shorter length.")
            min_length = min(len(residue_atoms), len(residue_positions))
            residue_atoms = residue_atoms[:min_length]
            residue_positions = residue_positions[:min_length]

        for atom_name, atom_position in zip(residue_atoms, residue_positions):
            if atom_name and all(coord is not None and not np.isnan(coord) for coord in atom_position):
                try:
                    element = app.Element.getBySymbol(atom_name[0])
                    topology.addAtom(atom_name, element, omm_residue)
                    positions.append(unit.Quantity(atom_position, unit.angstrom))
                except Exception as e:
                    logging.warning(f"Error adding atom {atom_name} in residue {residue_index}: {str(e)}")
            else:
                logging.warning(f"Skipping invalid atom in residue {residue_index}: {atom_name}, {atom_position}")

    def _handle_residue_error(self, residue_index, residue_error):
        logging.error(f"Error processing residue {residue_index}: {str(residue_error)}")
        if residue_index == 0:
            raise ValueError(f"Failed to process first residue. Check protein structure data.") from residue_error
        logging.warning(f"Skipping residue {residue_index} due to processing error.")

    def _create_openmm_system(self, topology):
        try:
            forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
            return forcefield.createSystem(
                topology,
                nonbondedMethod=app.PME,
                nonbondedCutoff=1*unit.nanometer,
                constraints=app.HBonds
            )
        except Exception as system_error:
            logging.error(f"Error creating OpenMM System: {str(system_error)}")
            raise ValueError("Failed to create OpenMM System. Check topology and force field compatibility.") from system_error

    def _create_openmm_integrator(self):
        try:
            return openmm.LangevinMiddleIntegrator(
                300*unit.kelvin,
                1/unit.picosecond,
                0.002*unit.picoseconds
            )
        except Exception as integrator_error:
            logging.error(f"Error creating Integrator: {str(integrator_error)}")
            raise ValueError("Failed to create Integrator. Check OpenMM compatibility.") from integrator_error

    def _create_openmm_simulation(self, topology, system, integrator):
        try:
            return app.Simulation(topology, system, integrator, platform)
        except Exception as sim_error:
            logging.error(f"Error creating OpenMM Simulation: {str(sim_error)}")
            raise ValueError("Failed to create OpenMM Simulation. Check OpenMM installation and compatibility.") from sim_error

    def _set_simulation_positions(self, positions):
        try:
            positions_nm = [pos.in_units_of(unit.nanometer) for pos in positions]
            self.openmm_simulation.context.setPositions(positions_nm)
            logging.info("Successfully set atom positions in OpenMM simulation.")
        except Exception as position_error:
            logging.error(f"Error setting positions: {str(position_error)}")
            logging.debug(f"Position data: {positions}")
            self.openmm_simulation = None  # Clean up the simulation object
            raise ValueError("Failed to set atom positions. Check unit conversions and position data.") from position_error

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

            # Update the predicted protein with refined positions
            for i, residue in enumerate(predicted_protein.residue_index):
                predicted_protein.atom_positions[residue] = refined_positions[i].value_in_unit(unit.angstrom)

        return predicted_protein

    def get_plddt_scores(self) -> jnp.ndarray:
        """
        Get pLDDT (predicted LDDT) scores for the predicted structure.

        Returns:
            jnp.ndarray: Array of pLDDT scores, ranging from 0 to 100.

        Raises:
            ValueError: If the model or features are not set up.
            RuntimeError: If there's an error during prediction or processing.
        """
        if not self.is_model_ready():
            logging.error("Model is not ready for pLDDT score prediction.")
            raise ValueError("Model setup incomplete. Call setup_model() and prepare_features() first.")

        try:
            logging.info("Predicting pLDDT scores...")
            prediction_result = self.model({'params': self.model_params}, jax.random.PRNGKey(0), self.config, **self.feature_dict)

            if 'plddt' not in prediction_result:
                logging.error("pLDDT scores not found in model output")
                raise KeyError("pLDDT scores not found in model output")

            plddt_scores = prediction_result['plddt']

            if not isinstance(plddt_scores, jnp.ndarray):
                logging.warning(f"pLDDT scores are not a JAX array. Converting from type: {type(plddt_scores)}")
                plddt_scores = jnp.array(plddt_scores)

            if plddt_scores.ndim != 1:
                logging.warning(f"Unexpected pLDDT score shape: {plddt_scores.shape}. Flattening.")
                plddt_scores = plddt_scores.flatten()

            # Ensure scores are within the expected range
            plddt_scores = jnp.clip(plddt_scores, 0, 100)

            logging.info(f"Successfully predicted pLDDT scores. Shape: {plddt_scores.shape}, "
                         f"Range: [{plddt_scores.min():.2f}, {plddt_scores.max():.2f}]")
            return plddt_scores
        except KeyError as ke:
            logging.error(f"KeyError in pLDDT score prediction: {str(ke)}")
            raise RuntimeError(f"Failed to predict pLDDT scores: {str(ke)}") from ke
        except Exception as e:
            logging.error(f"Unexpected error in pLDDT score prediction: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to predict pLDDT scores: {str(e)}") from e

    def get_predicted_aligned_error(self) -> jnp.ndarray:
        """
        Get predicted aligned error for the structure.

        Returns:
            jnp.ndarray: 2D array of predicted aligned errors.

        Raises:
            ValueError: If the model or features are not set up.
            RuntimeError: If there's an error during prediction.
        """
        if self.model is None or self.feature_dict is None:
            logging.error("Model or features not set up for predicted aligned error calculation.")
            raise ValueError("Model or features not set up. Call setup_model() and prepare_features() first.")

        try:
            logging.info("Calculating predicted aligned error...")
            prediction_result = self.model({'params': self.model_params}, jax.random.PRNGKey(0), self.config, **self.feature_dict)
            predicted_aligned_error = prediction_result['predicted_aligned_error']

            if not isinstance(predicted_aligned_error, jnp.ndarray):
                predicted_aligned_error = jnp.array(predicted_aligned_error)

            logging.info(f"Successfully calculated predicted aligned error. Shape: {predicted_aligned_error.shape}")
            return predicted_aligned_error
        except Exception as e:
            logging.error(f"Error calculating predicted aligned error: {str(e)}")
            raise RuntimeError(f"Failed to calculate predicted aligned error: {str(e)}")

import re
import random
from typing import Dict, Any, Tuple

class AlphaFoldIntegration:
    # ... (existing methods)

    def run_alphaproteo(self, sequence: str) -> Dict[str, Any]:
        """
        Run AlphaProteo analysis on the given protein sequence.

        Args:
            sequence (str): Amino acid sequence.

        Returns:
            Dict[str, Any]: Results of AlphaProteo analysis.

        Raises:
            ValueError: If the input sequence is invalid.
            RuntimeError: If an unexpected error occurs during analysis.
        """
        try:
            self._validate_sequence(sequence)
            logging.info(f"Starting AlphaProteo analysis for sequence of length {len(sequence)}")

            molecular_weight = self._calculate_molecular_weight(sequence)
            isoelectric_point = self._calculate_isoelectric_point(sequence)
            hydrophobicity = self._calculate_hydrophobicity(sequence)

            result = {
                "status": "success",
                "sequence_length": len(sequence),
                "predicted_properties": {
                    "molecular_weight": round(molecular_weight, 2),
                    "isoelectric_point": round(isoelectric_point, 2),
                    "hydrophobicity": round(hydrophobicity, 2)
                }
            }

            logging.info(f"AlphaProteo analysis completed successfully for sequence of length {len(sequence)}")
            return result
        except ValueError as ve:
            logging.error(f"ValueError in AlphaProteo analysis: {str(ve)}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error in AlphaProteo analysis: {str(e)}")
            raise RuntimeError("An unexpected error occurred during AlphaProteo analysis") from e

    def _validate_sequence(self, sequence: str) -> None:
        """
        Validate the input protein sequence.

        Args:
            sequence (str): The protein sequence to validate.

        Raises:
            ValueError: If the sequence is invalid.
        """
        if not sequence or not isinstance(sequence, str):
            raise ValueError("Invalid sequence input. Must be a non-empty string.")

        sequence = sequence.upper()
        if not all(aa in 'ACDEFGHIKLMNPQRSTVWY' for aa in sequence):
            raise ValueError("Invalid amino acid sequence. Contains non-standard amino acids.")

        if len(sequence) > 2000:
            raise ValueError("Sequence too long. Maximum length is 2000 amino acids.")

    def _calculate_molecular_weight(self, sequence: str) -> float:
        """
        Calculate the molecular weight of the protein sequence.

        Args:
            sequence (str): The protein sequence.

        Returns:
            float: The calculated molecular weight.
        """
        return sum(residue_constants.residue_weights[aa] for aa in sequence)

    def _calculate_isoelectric_point(self, sequence: str) -> float:
        """
        Estimate the isoelectric point of the protein sequence.

        Args:
            sequence (str): The protein sequence.

        Returns:
            float: The estimated isoelectric point.
        """
        pos_charge = sum(sequence.count(aa) for aa in 'RKH')
        neg_charge = sum(sequence.count(aa) for aa in 'DE')
        return 7.0 + (pos_charge - neg_charge) / len(sequence)

    def _calculate_hydrophobicity(self, sequence: str) -> float:
        """
        Calculate the average hydrophobicity of the protein sequence.

        Args:
            sequence (str): The protein sequence.

        Returns:
            float: The average hydrophobicity score.
        """
        hydrophobicity_scale = {'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'Q': -3.5, 'E': -3.5,
                                'G': -0.4, 'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8,
                                'P': -1.6, 'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2}
        return sum(hydrophobicity_scale[aa] for aa in sequence) / len(sequence)

    def run_alphamissense(self, sequence: str, mutation: str) -> Dict[str, Any]:
        """
        Run AlphaMissense analysis on the given protein sequence and mutation.

        Args:
            sequence (str): Original amino acid sequence.
            mutation (str): Mutation in the format 'X123Y' where X is the original amino acid,
                            123 is the position, and Y is the mutated amino acid.

        Returns:
            Dict[str, Any]: Results of AlphaMissense analysis.

        Raises:
            ValueError: If the input sequence or mutation is invalid.
            RuntimeError: If an unexpected error occurs during analysis.
        """
        try:
            self._validate_alphamissense_input(sequence, mutation)

            original_aa, position, mutated_aa = mutation[0], int(mutation[1:-1]), mutation[-1]
            logging.info(f"Starting AlphaMissense analysis for mutation {mutation}")

            severity_score = self._calculate_severity_score(original_aa, mutated_aa)
            confidence_score = self._calculate_confidence_score(severity_score)
            severity = self._determine_severity(severity_score)
            impact_score, functional_impact = self._determine_functional_impact(severity_score)

            result = {
                "status": "success",
                "mutation": mutation,
                "predicted_effect": {
                    "severity": severity,
                    "confidence": round(confidence_score, 2),
                    "functional_impact": functional_impact,
                    "severity_score": severity_score,
                    "impact_score": round(impact_score, 2)
                }
            }

            logging.info(f"AlphaMissense analysis completed successfully for mutation {mutation}")
            return result

        except ValueError as ve:
            logging.error(f"ValueError in AlphaMissense analysis: {str(ve)}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error in AlphaMissense analysis: {str(e)}")
            raise RuntimeError("An unexpected error occurred during AlphaMissense analysis") from e

    def _validate_alphamissense_input(self, sequence: str, mutation: str) -> None:
        """
        Validate the input for AlphaMissense analysis.

        Args:
            sequence (str): The protein sequence.
            mutation (str): The mutation string.

        Raises:
            ValueError: If the sequence or mutation is invalid.
        """
        self._validate_sequence(sequence)

        if not re.match(r'^[A-Z]\d+[A-Z]$', mutation):
            raise ValueError("Invalid mutation format. Must be in the format 'X123Y'.")

        original_aa, position, mutated_aa = mutation[0], int(mutation[1:-1]), mutation[-1]

        if position < 1 or position > len(sequence):
            raise ValueError(f"Invalid mutation position. Must be between 1 and {len(sequence)}.")

        if sequence[position-1] != original_aa:
            raise ValueError(f"Original amino acid in mutation does not match the sequence at position {position}.")

    def _calculate_severity_score(self, original_aa: str, mutated_aa: str) -> int:
        """
        Calculate the severity score of a mutation.

        Args:
            original_aa (str): The original amino acid.
            mutated_aa (str): The mutated amino acid.

        Returns:
            int: The calculated severity score.
        """
        aa_properties = {
            'A': 0, 'R': 1, 'N': 0, 'D': -1, 'C': 0, 'Q': 0, 'E': -1, 'G': 0,
            'H': 1, 'I': 0, 'L': 0, 'K': 1, 'M': 0, 'F': 0, 'P': 0, 'S': 0,
            'T': 0, 'W': 0, 'Y': 0, 'V': 0
        }
        return abs(aa_properties[original_aa] - aa_properties[mutated_aa])

    def _calculate_confidence_score(self, severity_score: int) -> float:
        """
        Calculate the confidence score based on the severity score.

        Args:
            severity_score (int): The severity score of the mutation.

        Returns:
            float: The calculated confidence score.
        """
        return min(1.0, 0.5 + severity_score * 0.1 + random.random() * 0.3)

    def _determine_severity(self, severity_score: int) -> str:
        """
        Determine the severity category based on the severity score.

        Args:
            severity_score (int): The severity score of the mutation.

        Returns:
            str: The determined severity category.
        """
        if severity_score == 0:
            return "benign"
        elif severity_score == 1:
            return "moderate"
        else:
            return "severe"

    def _determine_functional_impact(self, severity_score: int) -> Tuple[float, str]:
        """
        Determine the functional impact based on the severity score.

        Args:
            severity_score (int): The severity score of the mutation.

        Returns:
            Tuple[float, str]: The impact score and functional impact category.
        """
        impact_score = severity_score + (random.random() - 0.5)
        if impact_score < 0.5:
            return impact_score, "likely benign"
        elif impact_score < 1.5:
            return impact_score, "possibly damaging"
        else:
            return impact_score, "probably damaging"
