import pytest
from unittest.mock import MagicMock, patch
import logging
import sys
import os
import importlib

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Log current sys.path
logger.debug(f"Current sys.path: {sys.path}")

# Log the location of the Bio package if it exists
try:
    import Bio
    logger.debug(f"Bio package location: {os.path.dirname(Bio.__file__)}")
    logger.debug(f"Bio package version: {Bio.__version__}")
except ImportError:
    logger.debug("Bio package not found in the current environment")

# Log current sys.modules
logger.debug(f"Current sys.modules before patching: {list(sys.modules.keys())}")

# Create a comprehensive mock for the alphafold.model.tf module
mock_alphafold_tf = MagicMock()

# Add the shape_placeholders attribute to the mock with expected contents
mock_alphafold_tf.shape_placeholders = MagicMock()
mock_alphafold_tf.shape_placeholders.NUM_RES = 'num_residues'
mock_alphafold_tf.shape_placeholders.NUM_MSA_SEQ = 'num_msa_sequences'
mock_alphafold_tf.shape_placeholders.NUM_EXTRA_SEQ = 'num_extra_sequences'
mock_alphafold_tf.shape_placeholders.NUM_TEMPLATES = 'num_templates'

# Create a mock for SCOPData
mock_scop_data = MagicMock()
mock_scop_data.protein_letters_3to1 = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}

# Patch the alphafold.model.tf module
@pytest.fixture(scope="session", autouse=True)
def patch_alphafold_tf():
    logger.debug("Applying mock for alphafold.model.tf")
    with patch.dict('sys.modules', {'alphafold.model.tf': mock_alphafold_tf}):
        logger.debug(f"sys.modules after patching alphafold.model.tf: {list(sys.modules.keys())}")
        yield mock_alphafold_tf
    logger.debug("Mock for alphafold.model.tf has been removed")

# Patch the Bio.Data.SCOPData module
@pytest.fixture(scope="session", autouse=True)
def patch_scop_data():
    logger.debug("Applying mock for Bio.Data.SCOPData")
    try:
        with patch.dict('sys.modules', {'Bio.Data.SCOPData': mock_scop_data}):
            logger.debug(f"sys.modules after patching Bio.Data.SCOPData: {list(sys.modules.keys())}")
            # Force reload of Bio.Data to ensure our mock is used
            importlib.reload(importlib.import_module('Bio.Data'))
            yield mock_scop_data
    except Exception as e:
        logger.error(f"Error while patching Bio.Data.SCOPData: {str(e)}")
        raise
    finally:
        logger.debug("Mock for Bio.Data.SCOPData has been removed")

# Ensure the mocks are applied before any test imports
pytest.register_assert_rewrite('alphafold.model.tf')
pytest.register_assert_rewrite('Bio.Data.SCOPData')

logger.debug("conftest.py has been loaded and configured")
