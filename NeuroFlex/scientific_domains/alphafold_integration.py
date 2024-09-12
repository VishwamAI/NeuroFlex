import jax
import jax.numpy as jnp
import haiku as hk
from typing import List, Dict, Any, Tuple
from alphafold.model import config, modules
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
import importlib
import importlib.metadata
import openmm
import openmm.app as app
import openmm.unit as unit

# Configure logging only if it hasn't been configured yet
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

def check_version(package_name: str, expected_version: str) -> bool:
    try:
        version = importlib.metadata.version(package_name)
        logger.info(f"{package_name} version: {version}")
        if version != expected_version:
            logger.warning(f"This integration was tested with {package_name} {expected_version}. You are using version {version}. Some features may not work as expected.")
            return False
        return True
    except importlib.metadata.PackageNotFoundError:
        logger.error(f"Unable to determine {package_name} version. Make sure it's installed correctly.")
        return False

# Check versions and set flags for fallback strategies
ALPHAFOLD_COMPATIBLE = check_version("alphafold", "2.0.0")
JAX_COMPATIBLE = check_version("jax", "0.3.25")
HAIKU_COMPATIBLE = check_version("haiku", "0.0.9")
OPENMM_COMPATIBLE = check_version("openmm", "7.7.0")  # Add OpenMM version check

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
        self.openmm_system = None
        self.openmm_simulation = None
        self.openmm_integrator = None
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
            system = forcefield.createSystem(topology, nonbondedMethod=app.PME, nonbondedCutoff=1*unit.nanometer, constraints=app.HBonds)

            # Set up integrator and simulation
            integrator = openmm.LangevinMiddleIntegrator(300*unit.kelvin, 1/unit.picosecond, 0.002*unit.picoseconds)
            self.openmm_simulation = app.Simulation(topology, system, integrator, platform)
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

            # Update the predicted protein with refined positions
            for i, residue in enumerate(predicted_protein.residue_index):
                predicted_protein.atom_positions[residue] = refined_positions[i].value_in_unit(unit.angstrom)

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
