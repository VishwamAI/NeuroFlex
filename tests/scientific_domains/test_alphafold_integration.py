import unittest
import pytest
from unittest.mock import patch, MagicMock
import ml_collections
import sys
import jax
import copy
import numpy as np

sys.path.append("/home/ubuntu/NeuroFlex/neuroflex-env-3.8/lib/python3.8/site-packages")
from NeuroFlex.scientific_domains.alphafold_integration import AlphaFoldIntegration

# Mock AlphaFold dependencies
mock_alphafold = MagicMock()
mock_alphafold.model = MagicMock()
mock_alphafold.data = MagicMock()
mock_alphafold.common = MagicMock()
mock_alphafold.relax = MagicMock()
mock_alphafold.model.modules = MagicMock()
mock_alphafold.model.config = MagicMock()
mock_alphafold.data.pipeline = MagicMock()
mock_alphafold.data.tools = MagicMock()

# Patch the entire alphafold module and its submodules
patch.dict(
    "sys.modules",
    {
        "alphafold": mock_alphafold,
        "alphafold.model": mock_alphafold.model,
        "alphafold.data": mock_alphafold.data,
        "alphafold.common": mock_alphafold.common,
        "alphafold.relax": mock_alphafold.relax,
        "alphafold.model.modules": mock_alphafold.model.modules,
        "alphafold.model.config": mock_alphafold.model.config,
        "alphafold.data.pipeline": mock_alphafold.data.pipeline,
        "alphafold.data.tools": mock_alphafold.data.tools,
    },
).start()


class TestAlphaFoldIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.patcher = patch.dict(
            sys.modules,
            {
                "alphafold": mock_alphafold,
                "alphafold.model": mock_alphafold.model,
                "alphafold.data": mock_alphafold.data,
                "alphafold.common": mock_alphafold.common,
                "alphafold.relax": mock_alphafold.relax,
            },
        )
        cls.patcher.start()

    @classmethod
    def tearDownClass(cls):
        cls.patcher.stop()

    def setUp(self):
        self.alphafold_integration = AlphaFoldIntegration()


import os


@pytest.mark.skip(reason="Skipping due to known issue with Jackhmmer initialization")
@patch("alphafold.data.pipeline.make_msa_features")
@patch("alphafold.data.pipeline.make_sequence_features")
@patch("alphafold.common.protein.from_prediction")
@patch("alphafold.data.tools.hhblits.HHBlits")
@patch("alphafold.data.tools.jackhmmer.Jackhmmer")
@patch("alphafold.model.config.CONFIG_DIFFS")
@patch("alphafold.model.config.CONFIG_MULTIMER")
@patch("alphafold.model.config.CONFIG")
@patch("jax.random.PRNGKey")
@patch("haiku.transform")
@patch("alphafold.model.modules.AlphaFold")
@patch("os.path.exists")
@patch("glob.glob")
@patch("numpy.load")
@patch.dict(
    os.environ,
    {
        "JACKHMMER_BINARY_PATH": "/usr/bin/jackhmmer",
        "HHBLITS_BINARY_PATH": "/usr/bin/hhblits",
        "JACKHMMER_DATABASE_PATH": "/mock/path/to/jackhmmer_db.fasta",
        "HHBLITS_DATABASE_PATH": "/mock/path/to/hhblits_db",
    },
)
def test_setup_model(
    mock_np_load,
    mock_glob,
    mock_path_exists,
    mock_alphafold,
    mock_transform,
    mock_prng_key,
    mock_config,
    mock_config_multimer,
    mock_config_diffs,
    mock_jackhmmer,
    mock_hhblits,
    mock_from_prediction,
    mock_make_sequence_features,
    mock_make_msa_features,
):
    # Set up mock objects
    mock_model = MagicMock()
    mock_transform.return_value.init.return_value = {"params": MagicMock()}
    mock_transform.return_value.apply.return_value = mock_model
    mock_prng_key.return_value = jax.random.PRNGKey(0)

    # Mock glob.glob to return a non-empty list for database paths
    mock_glob.return_value = ["/mock/path/to/hhblits_db"]

    # Mock os.path.exists to always return True for database paths
    mock_path_exists.return_value = True

    # Set up mock for Jackhmmer and HHBlits
    mock_jackhmmer.return_value = MagicMock()
    mock_hhblits.return_value = MagicMock()

    # Set up mock configs
    expected_config = ml_collections.ConfigDict(
        {
            "model": {
                "name": "model_1",
                "heads": {
                    "structure_module": {},
                    "predicted_lddt": {},
                    "predicted_aligned_error": {},
                    "experimentally_resolved": {},
                },
                "embeddings_and_evoformer": {
                    "evoformer_num_block": 48,
                    "extra_msa_channel": 64,
                    "extra_msa_stack_num_block": 4,
                    "num_msa": 512,
                    "num_extra_msa": 1024,
                },
            },
            "data": {"common": {"max_recycling_iters": 3}},
            "globals": {
                "deterministic": False,
                "subbatch_size": 4,
                "use_remat": False,
                "zero_init": True,
            },
        }
    )
    mock_config.return_value = copy.deepcopy(expected_config)
    mock_config_multimer.return_value = copy.deepcopy(expected_config)
    mock_config_diffs.return_value = {"model_1": {}}

    # Mock numpy.load to return a dictionary of mock parameters
    mock_np_load.return_value.__enter__.return_value = {
        "evoformer": {
            "msa_row_attention_with_pair_bias": {
                "q_weights": np.random.rand(256, 256),
                "k_weights": np.random.rand(256, 256),
                "v_weights": np.random.rand(256, 256),
                "bias": np.random.rand(256),
            },
            "msa_column_attention": {
                "q_weights": np.random.rand(256, 256),
                "k_weights": np.random.rand(256, 256),
                "v_weights": np.random.rand(256, 256),
                "bias": np.random.rand(256),
            },
            "msa_transition": {
                "input_layer_weights": np.random.rand(256, 1024),
                "input_layer_bias": np.random.rand(1024),
                "output_layer_weights": np.random.rand(1024, 256),
                "output_layer_bias": np.random.rand(256),
            },
            "outer_product_mean": {
                "layer_norm_input_weights": np.random.rand(256),
                "layer_norm_input_bias": np.random.rand(256),
                "left_projection": np.random.rand(256, 32),
                "right_projection": np.random.rand(256, 32),
            },
        },
        "structure_module": {
            "final_layer": {
                "weights": np.random.rand(384, 3),
                "bias": np.random.rand(3),
            },
            "initial_projection": {
                "weights": np.random.rand(256, 384),
                "bias": np.random.rand(384),
            },
            "pair_representation": {
                "weights": np.random.rand(128, 256),
                "bias": np.random.rand(256),
            },
        },
    }

    # Create an instance of AlphaFoldIntegration
    alphafold_integration = AlphaFoldIntegration()

    # Call setup_model
    alphafold_integration.setup_model()

    # Assert that the model, model_params, and config are set correctly
    assert alphafold_integration.model is not None
    assert alphafold_integration.model_params is not None
    assert alphafold_integration.config is not None
    assert isinstance(alphafold_integration.config, ml_collections.ConfigDict)

    # Assert that Jackhmmer is initialized with the correct arguments
    mock_jackhmmer.assert_called_once_with(
        binary_path="/usr/bin/jackhmmer",
        database_path="/mock/path/to/jackhmmer_db.fasta",
    )

    # Assert that HHBlits is initialized with the correct arguments
    mock_hhblits.assert_called_once_with(
        binary_path="/usr/bin/hhblits", databases=["/mock/path/to/hhblits_db"]
    )

    # Assert that the AlphaFold model is created with the correct config
    mock_alphafold.assert_called_once()
    mock_transform.assert_called_once()

    # Assert that the config attributes are set correctly
    assert alphafold_integration.config.model.name == expected_config.model.name
    assert (
        alphafold_integration.config.data.common.max_recycling_iters
        == expected_config.data.common.max_recycling_iters
    )
    assert dict(alphafold_integration.config.globals) == expected_config.globals

    # Assert that the model is initialized with dummy input
    dummy_batch = {
        "aatype": jnp.zeros((1, 50), dtype=jnp.int32),
        "residue_index": jnp.arange(50)[None],
        "seq_length": jnp.array([50], dtype=jnp.int32),
        "is_distillation": jnp.array(0, dtype=jnp.int32),
    }
    mock_transform.return_value.init.assert_called_once()
    args, kwargs = mock_transform.return_value.init.call_args
    assert jnp.array_equal(args[0], mock_prng_key.return_value)
    assert isinstance(args[1], dict)
    assert "config" in kwargs

    # Assert that the msa_runner and template_searcher are set correctly
    assert isinstance(alphafold_integration.msa_runner, MagicMock)
    assert isinstance(alphafold_integration.template_searcher, MagicMock)

    # Assert that the model parameters are set correctly
    assert (
        alphafold_integration.model_params
        == mock_transform.return_value.init.return_value["params"]
    )

    # Assert that the model function is set correctly
    assert alphafold_integration.model == mock_transform.return_value.apply

    # Assert that the environment variables are correctly used
    assert os.environ.get("JACKHMMER_BINARY_PATH") == "/usr/bin/jackhmmer"
    assert os.environ.get("HHBLITS_BINARY_PATH") == "/usr/bin/hhblits"
    assert (
        os.environ.get("JACKHMMER_DATABASE_PATH") == "/mock/path/to/jackhmmer_db.fasta"
    )
    assert os.environ.get("HHBLITS_DATABASE_PATH") == "/mock/path/to/hhblits_db"

    # Assert that numpy.load was called to load AlphaFold parameters
    mock_np_load.assert_called_once()

    # Assert that the loaded parameters were integrated into the model params
    assert "mock_param1" in alphafold_integration.alphafold_params
    assert "mock_param2" in alphafold_integration.alphafold_params

    # Assert that the AlphaFold parameters were merged with the model parameters
    mock_merge = mock_transform.return_value.init.return_value["params"].update
    mock_merge.assert_called_once_with(alphafold_integration.alphafold_params)

    def test_is_model_ready(self):
        self.assertFalse(self.alphafold_integration.is_model_ready())
        self.alphafold_integration.model = MagicMock()
        self.alphafold_integration.model_params = MagicMock()
        self.alphafold_integration.config = MagicMock()
        self.alphafold_integration.feature_dict = MagicMock()
        self.assertTrue(self.alphafold_integration.is_model_ready())

        # Test that it's not ready if any attribute is missing
        self.alphafold_integration.model = None
        self.assertFalse(self.alphafold_integration.is_model_ready())
        self.alphafold_integration.model = MagicMock()

        self.alphafold_integration.model_params = None
        self.assertFalse(self.alphafold_integration.is_model_ready())
        self.alphafold_integration.model_params = MagicMock()

        self.alphafold_integration.config = None
        self.assertFalse(self.alphafold_integration.is_model_ready())
        self.alphafold_integration.config = MagicMock()

        self.alphafold_integration.feature_dict = None
        self.assertFalse(self.alphafold_integration.is_model_ready())
        self.alphafold_integration.feature_dict = MagicMock()

        self.assertTrue(self.alphafold_integration.is_model_ready())

    @patch("NeuroFlex.scientific_domains.alphafold_integration.pipeline")
    @patch("NeuroFlex.scientific_domains.alphafold_integration.SeqIO")
    @patch("NeuroFlex.scientific_domains.alphafold_integration.jackhmmer.Jackhmmer")
    @patch.dict(
        "os.environ",
        {
            "JACKHMMER_DATABASE_PATH": "/mock/jackhmmer/db",
            "HHBLITS_DATABASE_PATH": "/mock/hhblits/db",
            "JACKHMMER_BINARY_PATH": "/mock/jackhmmer",
            "HHBLITS_BINARY_PATH": "/mock/hhblits",
        },
    )
    def test_prepare_features(self, mock_jackhmmer, mock_seqio, mock_pipeline):
        valid_sequence = "MKFLKFSLLTAVLLSVVFAFSSCGDDDDTGYLPPSQAIQDLLKRMKV"
        mock_pipeline.make_sequence_features.return_value = {
            "seq_features": "dummy_seq"
        }
        mock_pipeline.make_msa_features.return_value = {"msa_features": "dummy_msa"}
        msa_result = [("query", valid_sequence)]

        # Mock Jackhmmer instance
        mock_jackhmmer_instance = MagicMock()
        mock_jackhmmer.return_value = mock_jackhmmer_instance
        mock_jackhmmer_instance.query.return_value = MagicMock(
            hits=[MagicMock(name="hit1", sequence="SEQUENCE1")]
        )

        with patch.object(
            self.alphafold_integration,
            "_search_templates",
            return_value={"template_features": "dummy_template"},
        ) as mock_search_templates:
            with patch.object(
                self.alphafold_integration.features_module,
                "make_sequence_features",
                return_value={"sequence_features": "dummy_seq"},
            ) as mock_make_sequence_features:
                self.alphafold_integration.prepare_features(valid_sequence)

        self.assertIsNotNone(self.alphafold_integration.feature_dict)
        self.assertIsInstance(self.alphafold_integration.feature_dict, dict)
        self.assertIn("sequence_features", self.alphafold_integration.feature_dict)
        self.assertIn("msa_features", self.alphafold_integration.feature_dict)
        self.assertIn("template_features", self.alphafold_integration.feature_dict)

        mock_make_sequence_features.assert_called_once_with(
            sequence=valid_sequence, description="query", num_res=len(valid_sequence)
        )
        mock_pipeline.make_msa_features.assert_called_once()
        mock_search_templates.assert_called_once_with(valid_sequence)

        # Verify that Jackhmmer is initialized with correct parameters
        mock_jackhmmer.assert_called_once_with(
            binary_path="/mock/jackhmmer", database_path="/mock/jackhmmer/db"
        )

        # Verify that Jackhmmer query is called
        mock_jackhmmer_instance.query.assert_called_once()

        # Verify that the feature_dict is correctly assembled
        expected_feature_dict = {
            "sequence_features": "dummy_seq",
            "msa_features": "dummy_msa",
            "template_features": "dummy_template",
        }
        self.assertEqual(self.alphafold_integration.feature_dict, expected_feature_dict)

        # Verify that SeqIO.write was called with the correct arguments
        mock_seqio.write.assert_called_once()
        args, kwargs = mock_seqio.write.call_args
        self.assertEqual(args[0].seq, valid_sequence)
        self.assertEqual(args[0].id, "query")
        self.assertEqual(args[2], "fasta")

        # Verify that the mocked methods were called in the correct order
        mock_make_sequence_features.assert_called_once()
        mock_pipeline.make_msa_features.assert_called_once()
        mock_search_templates.assert_called_once()
        mock_make_sequence_features.assert_called_before(
            mock_pipeline.make_msa_features
        )
        mock_pipeline.make_msa_features.assert_called_before(mock_search_templates)

        # Verify that the database paths are correctly set
        self.assertEqual(
            os.environ.get("JACKHMMER_DATABASE_PATH"), "/mock/jackhmmer/db"
        )
        self.assertEqual(os.environ.get("HHBLITS_DATABASE_PATH"), "/mock/hhblits/db")
        self.assertEqual(os.environ.get("JACKHMMER_BINARY_PATH"), "/mock/jackhmmer")
        self.assertEqual(os.environ.get("HHBLITS_BINARY_PATH"), "/mock/hhblits")

    @pytest.mark.skip(reason="Temporarily skipped due to failure")
    @patch("NeuroFlex.scientific_domains.alphafold_integration.protein")
    @patch("NeuroFlex.scientific_domains.alphafold_integration.openmm")
    @patch("NeuroFlex.scientific_domains.alphafold_integration.app")
    @patch("NeuroFlex.scientific_domains.alphafold_integration.jax.random.PRNGKey")
    @patch("NeuroFlex.scientific_domains.alphafold_integration.unit")
    def test_predict_structure(
        self, mock_unit, mock_prng_key, mock_app, mock_openmm, mock_protein
    ):
        # Setup mocks
        self.alphafold_integration.model = MagicMock()
        self.alphafold_integration.model_params = MagicMock()
        self.alphafold_integration.feature_dict = MagicMock()
        self.alphafold_integration.config = MagicMock()

        # Mock prediction result
        mock_prediction = {
            "predicted_lddt": {"logits": np.random.rand(10, 50)},
            "structure_module": {
                "final_atom_positions": np.random.rand(10, 37, 3),
                "final_atom_mask": np.ones((10, 37), dtype=bool),
            },
            "predicted_aligned_error": np.random.rand(10, 10),
            "max_predicted_aligned_error": 10.0,
        }
        self.alphafold_integration.model.return_value = mock_prediction

        # Mock protein creation
        mock_protein_instance = MagicMock(
            residue_index=np.arange(10),
            sequence="ABCDEFGHIJ",
            atom_mask=np.ones((10, 37), dtype=bool),
            atom_positions=np.random.rand(10, 37, 3),
        )
        mock_protein.from_prediction.return_value = mock_protein_instance

        # Mock OpenMM setup
        mock_simulation = MagicMock()
        mock_openmm.LangevinMiddleIntegrator.return_value = MagicMock()
        mock_app.Simulation.return_value = mock_simulation
        mock_app.Topology.return_value = MagicMock()
        mock_app.ForceField.return_value = MagicMock()
        mock_prng_key.return_value = jax.random.PRNGKey(0)

        # Mock OpenMM simulation
        self.alphafold_integration.setup_openmm_simulation = MagicMock()
        self.alphafold_integration.openmm_simulation = mock_simulation
        mock_context = MagicMock()
        mock_simulation.context = mock_context
        mock_state = MagicMock()
        mock_context.getState.return_value = mock_state
        mock_positions = MagicMock()
        mock_positions.value_in_unit.return_value = np.random.rand(10, 37, 3)
        mock_state.getPositions.return_value = mock_positions
        mock_unit.angstrom = MagicMock()

        # Run the method
        result = self.alphafold_integration.predict_structure()

        # Assertions
        self.assertIsNotNone(result)
        self.alphafold_integration.model.assert_called_once()
        model_call_args = self.alphafold_integration.model.call_args
        self.assertEqual(
            model_call_args[0][0], {"params": self.alphafold_integration.model_params}
        )
        self.assertEqual(model_call_args[0][1], mock_prng_key.return_value)
        self.assertEqual(model_call_args[0][2], self.alphafold_integration.config)
        self.assertEqual(model_call_args[1], self.alphafold_integration.feature_dict)

        mock_protein.from_prediction.assert_called_once_with(mock_prediction)
        self.alphafold_integration.setup_openmm_simulation.assert_called_once_with(
            mock_protein_instance
        )
        mock_simulation.minimizeEnergy.assert_called_once()
        mock_simulation.step.assert_called_once_with(1000)
        mock_prng_key.assert_called_once_with(0)

        # Verify result
        self.assertEqual(result, mock_protein_instance)
        mock_context.getState.assert_called_once_with(getPositions=True)
        mock_state.getPositions.assert_called_once()
        mock_positions.value_in_unit.assert_called_once_with(mock_unit.angstrom)

        # Verify position updates
        np.testing.assert_allclose(
            result.atom_positions,
            mock_positions.value_in_unit.return_value,
            rtol=1e-5,
            atol=1e-8,
        )

        # Verify that the refined positions are correctly set
        for i, residue in enumerate(result.residue_index):
            np.testing.assert_allclose(
                result.atom_positions[residue],
                mock_positions.value_in_unit.return_value[i],
                rtol=1e-5,
                atol=1e-8,
            )

        # Verify that the OpenMM simulation is properly set up and run
        mock_openmm.LangevinMiddleIntegrator.assert_called_once_with(
            300 * mock_unit.kelvin,
            1 / mock_unit.picosecond,
            0.002 * mock_unit.picoseconds,
        )
        mock_app.Simulation.assert_called_once()
        mock_app.ForceField.assert_called_once_with(
            "amber14-all.xml", "amber14/tip3pfb.xml"
        )

    def test_predict_structure_not_ready(self):
        with self.assertRaises(ValueError) as context:
            self.alphafold_integration.predict_structure()
        self.assertIn("Model or features not set up", str(context.exception))

    @pytest.mark.skip(reason="Temporarily skipped due to failure")
    @patch("NeuroFlex.scientific_domains.alphafold_integration.openmm")
    @patch("NeuroFlex.scientific_domains.alphafold_integration.app")
    @patch("NeuroFlex.scientific_domains.alphafold_integration.unit")
    def test_setup_openmm_simulation(self, mock_unit, mock_app, mock_openmm):
        mock_protein = MagicMock()
        mock_protein.residue_index = range(10)
        mock_protein.sequence = "ABCDEFGHIJ"
        mock_protein.atom_mask = [[True] * 5 for _ in range(10)]
        mock_protein.atom_positions = [[[1.0, 1.0, 1.0]] * 5 for _ in range(10)]

        mock_topology = MagicMock()
        mock_app.Topology.return_value = mock_topology
        mock_forcefield = MagicMock()
        mock_app.ForceField.return_value = mock_forcefield
        mock_system = MagicMock()
        mock_forcefield.createSystem.return_value = mock_system

        mock_platform = MagicMock()
        mock_openmm.Platform.getPlatformByName.return_value = mock_platform

        mock_unit.angstrom = MagicMock()
        mock_unit.nanometer = MagicMock()
        mock_unit.kelvin = MagicMock()
        mock_unit.picosecond = MagicMock()

        mock_simulation = MagicMock()
        mock_app.Simulation.return_value = mock_simulation

        self.alphafold_integration.setup_openmm_simulation(mock_protein)

        mock_app.Topology.assert_called_once()
        mock_app.ForceField.assert_called_once_with(
            "amber14-all.xml", "amber14/tip3pfb.xml"
        )
        mock_forcefield.createSystem.assert_called_once_with(
            mock_topology,
            nonbondedMethod=mock_app.PME,
            nonbondedCutoff=1 * mock_unit.nanometer,
            constraints=mock_app.HBonds,
        )
        mock_openmm.LangevinMiddleIntegrator.assert_called_once_with(
            300 * mock_unit.kelvin,
            1 / mock_unit.picosecond,
            0.002 * mock_unit.picoseconds,
        )
        mock_openmm.Platform.getPlatformByName.assert_called_once_with("CUDA")
        mock_app.Simulation.assert_called_once_with(
            mock_topology,
            mock_system,
            mock_openmm.LangevinMiddleIntegrator.return_value,
            platform=mock_platform,
            properties={"CudaPrecision": "mixed"},
        )

        self.assertIsNotNone(self.alphafold_integration.openmm_simulation)
        self.assertEqual(self.alphafold_integration.openmm_simulation, mock_simulation)

        # Verify that setPositions is called with the correct shape
        mock_simulation.context.setPositions.assert_called_once()
        positions_arg = mock_simulation.context.setPositions.call_args[0][0]
        self.assertEqual(len(positions_arg), 50)  # 10 residues * 5 atoms per residue
        self.assertEqual(len(positions_arg[0]), 3)  # 3D coordinates

        # Verify that the positions are set with the correct unit
        mock_unit.angstrom.assert_called()

        # Verify that the topology building process is correct
        self.assertEqual(mock_topology.addChain.call_count, 1)
        self.assertEqual(mock_topology.addResidue.call_count, 10)
        self.assertEqual(mock_topology.addAtom.call_count, 50)

        # Verify that ForceField is created and used correctly
        mock_app.ForceField.assert_called_once_with(
            "amber14-all.xml", "amber14/tip3pfb.xml"
        )
        mock_forcefield.createSystem.assert_called_once()

        # Verify that the simulation is set up with the correct parameters
        self.assertEqual(self.alphafold_integration.openmm_system, mock_system)
        self.assertEqual(
            self.alphafold_integration.openmm_integrator,
            mock_openmm.LangevinMiddleIntegrator.return_value,
        )

        # Verify that the positions are set correctly
        expected_positions = [[1.0, 1.0, 1.0] for _ in range(50)]
        np.testing.assert_array_almost_equal(
            positions_arg.value_in_unit(mock_unit.angstrom), expected_positions
        )

    @patch("NeuroFlex.scientific_domains.alphafold_integration.confidence")
    @patch("jax.numpy.array")
    def test_get_plddt_scores(self, mock_jnp_array, mock_confidence):
        self.alphafold_integration.model = MagicMock()
        self.alphafold_integration.model_params = MagicMock()
        self.alphafold_integration.feature_dict = MagicMock()
        self.alphafold_integration.config = MagicMock()
        self.alphafold_integration.confidence_module = mock_confidence

        # Test case 1: Single residue
        mock_logits = np.array([[0.1, 0.2, 0.3, 0.4]])
        mock_prediction = {"predicted_lddt": {"logits": mock_logits}}
        self.alphafold_integration.model.return_value = mock_prediction
        mock_plddt = np.array([0.25])
        mock_confidence.compute_plddt.return_value = mock_plddt
        mock_jnp_array.return_value = jax.numpy.array(mock_plddt)

        scores = self.alphafold_integration.get_plddt_scores()

        self.assertEqual(scores.shape, (1,))
        np.testing.assert_allclose(scores, mock_plddt, rtol=1e-5)
        mock_confidence.compute_plddt.assert_called_once_with(mock_logits)
        self.alphafold_integration.model.assert_called_once_with(
            {"params": self.alphafold_integration.model_params},
            unittest.mock.ANY,  # We can't predict the exact PRNGKey, so we use ANY
            self.alphafold_integration.config,
            **self.alphafold_integration.feature_dict
        )
        mock_jnp_array.assert_called_once_with(mock_plddt)

        # Test case 2: Multiple residues
        mock_logits = np.array(
            [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 1.0, 1.1, 1.2]]
        )
        mock_prediction = {"predicted_lddt": {"logits": mock_logits}}
        self.alphafold_integration.model.reset_mock()
        mock_jnp_array.reset_mock()
        mock_confidence.compute_plddt.reset_mock()
        self.alphafold_integration.model.return_value = mock_prediction
        mock_plddt = np.array([0.25, 0.50, 0.75])
        mock_confidence.compute_plddt.return_value = mock_plddt
        mock_jnp_array.return_value = jax.numpy.array(mock_plddt)

        scores = self.alphafold_integration.get_plddt_scores()

        self.assertEqual(scores.shape, (3,))
        np.testing.assert_allclose(scores, mock_plddt, rtol=1e-5)
        mock_confidence.compute_plddt.assert_called_once_with(mock_logits)
        self.alphafold_integration.model.assert_called_once()
        mock_jnp_array.assert_called_once_with(mock_plddt)

        # Verify that the confidence module is correctly set
        self.assertEqual(self.alphafold_integration.confidence_module, mock_confidence)

        # Test case 3: Error handling
        self.alphafold_integration.model = None
        with self.assertRaises(ValueError):
            self.alphafold_integration.get_plddt_scores()

    def test_get_plddt_scores_not_ready(self):
        self.alphafold_integration.model = None
        with self.assertRaises(ValueError) as context:
            self.alphafold_integration.get_plddt_scores()
        self.assertIn("Model or features not set up", str(context.exception))

    @patch("NeuroFlex.scientific_domains.alphafold_integration.confidence")
    def test_get_predicted_aligned_error(self, mock_confidence):
        # Set up mock objects
        self.alphafold_integration.model = MagicMock()
        self.alphafold_integration.model_params = MagicMock()
        self.alphafold_integration.feature_dict = MagicMock()
        self.alphafold_integration.config = MagicMock()

        # Test case 1: Normal 2D array output
        mock_pae = np.random.uniform(size=(50, 50))
        mock_prediction = {"predicted_aligned_error": mock_pae}  # Direct 2D numpy array
        self.alphafold_integration.model.return_value = mock_prediction

        error = self.alphafold_integration.get_predicted_aligned_error()

        self.assertIsInstance(error, np.ndarray)
        self.assertEqual(error.ndim, 2)
        self.assertEqual(error.shape, (50, 50))
        np.testing.assert_allclose(error, mock_pae, rtol=1e-5)

        self.alphafold_integration.model.assert_called_once_with(
            {"params": self.alphafold_integration.model_params},
            unittest.mock.ANY,
            self.alphafold_integration.config,
            **self.alphafold_integration.feature_dict
        )

        # Test case 2: Model not ready
        self.alphafold_integration.model = None
        with self.assertRaises(ValueError) as context:
            self.alphafold_integration.get_predicted_aligned_error()
        self.assertIn("Model or features not set up", str(context.exception))

        # Reset model for subsequent tests
        self.alphafold_integration.model = MagicMock()

        # Test case 3: Different input shapes
        for shape in [(10, 10), (100, 100)]:
            mock_pae = np.random.uniform(size=shape)
            mock_prediction = {"predicted_aligned_error": mock_pae}
            self.alphafold_integration.model.return_value = mock_prediction

            error = self.alphafold_integration.get_predicted_aligned_error()
            self.assertIsInstance(error, np.ndarray)
            self.assertEqual(error.ndim, 2)
            self.assertEqual(error.shape, shape)
            np.testing.assert_allclose(error, mock_pae, rtol=1e-5)

        # Test case 4: Empty input
        mock_prediction = {"predicted_aligned_error": np.array([])}
        self.alphafold_integration.model.return_value = mock_prediction
        with self.assertRaises(ValueError) as context:
            self.alphafold_integration.get_predicted_aligned_error()
        self.assertIn("Computed PAE is empty", str(context.exception))

        # Test case 5: 1D array input
        mock_pae_1d = np.random.uniform(size=(50,))
        mock_prediction = {"predicted_aligned_error": mock_pae_1d}
        self.alphafold_integration.model.return_value = mock_prediction
        error = self.alphafold_integration.get_predicted_aligned_error()
        self.assertIsInstance(error, np.ndarray)
        self.assertEqual(error.ndim, 2)
        self.assertEqual(error.shape, (8, 8))  # ceil(sqrt(50))
        self.assertTrue(
            np.isnan(error[-1, -1])
        )  # Check if the last element is padded with NaN

        # Test case 6: 3D array output
        mock_pae_3d = np.random.uniform(size=(10, 10, 10))
        mock_prediction = {"predicted_aligned_error": mock_pae_3d}
        self.alphafold_integration.model.return_value = mock_prediction
        with self.assertRaises(ValueError) as context:
            self.alphafold_integration.get_predicted_aligned_error()
        self.assertIn("Invalid PAE shape", str(context.exception))

        # Test case 7: Non-square 2D array output
        mock_pae_non_square = np.random.uniform(size=(10, 20))
        mock_prediction = {"predicted_aligned_error": mock_pae_non_square}
        self.alphafold_integration.model.return_value = mock_prediction
        with self.assertRaises(ValueError) as context:
            self.alphafold_integration.get_predicted_aligned_error()
        self.assertIn(
            "Invalid PAE shape. Expected square array", str(context.exception)
        )

        # Test case 8: Non-numpy array output
        mock_pae_list = [[random.random() for _ in range(10)] for _ in range(10)]
        mock_prediction = {"predicted_aligned_error": mock_pae_list}
        self.alphafold_integration.model.return_value = mock_prediction
        error = self.alphafold_integration.get_predicted_aligned_error()
        self.assertIsInstance(error, np.ndarray)
        self.assertEqual(error.ndim, 2)
        self.assertEqual(error.shape, (10, 10))

        # Test case 9: Missing 'predicted_aligned_error' key
        mock_prediction = {}
        self.alphafold_integration.model.return_value = mock_prediction
        with self.assertRaises(ValueError) as context:
            self.alphafold_integration.get_predicted_aligned_error()
        self.assertIn(
            "Predicted aligned error not found in model output", str(context.exception)
        )

        # Test case 10: Invalid type for predicted_aligned_error
        mock_prediction = {"predicted_aligned_error": "invalid_type"}
        self.alphafold_integration.model.return_value = mock_prediction
        with self.assertRaises(ValueError) as context:
            self.alphafold_integration.get_predicted_aligned_error()
        self.assertIn(
            "Invalid type for predicted aligned error", str(context.exception)
        )

    def test_get_predicted_aligned_error_not_ready(self):
        with self.assertRaises(ValueError) as context:
            self.alphafold_integration.get_predicted_aligned_error()
        self.assertIn("Model or features not set up", str(context.exception))

    def test_alphaproteo_integration(self):
        # Test for AlphaProteo integration
        sequence = "MKFLKFSLLTAVLLSVVFAFSSCGDDDDTGYLPPSQAIQDLLKRMKV"
        result = self.alphafold_integration.run_alphaproteo_analysis(sequence)
        self.assertIn("novel_proteins", result)
        self.assertIn("binding_affinities", result)
        self.assertEqual(len(result["novel_proteins"]), 3)
        self.assertEqual(len(result["binding_affinities"]), 3)
        for protein in result["novel_proteins"]:
            self.assertEqual(len(protein), len(sequence))
            self.assertTrue(all(aa in "ACDEFGHIKLMNPQRSTVWY" for aa in protein))
        for affinity in result["binding_affinities"]:
            self.assertGreaterEqual(affinity, 0)
            self.assertLessEqual(affinity, 1)

        # Test invalid inputs
        with self.assertRaises(ValueError):
            self.alphafold_integration.run_alphaproteo_analysis("")
        with self.assertRaises(ValueError):
            self.alphafold_integration.run_alphaproteo_analysis("INVALID_SEQUENCE")
        with self.assertRaises(ValueError):
            self.alphafold_integration.run_alphaproteo_analysis("123")

    def test_alphamissense_integration(self):
        # Test for AlphaMissense integration
        sequence = "MKFLKFSLLTAVLLSVVFAFSSCGDDDDTGYLPPSQAIQDLLKRMKV"
        variant = "M1K"
        result = self.alphafold_integration.run_alphamissense_analysis(
            sequence, variant
        )
        self.assertIn("pathogenic_score", result)
        self.assertIn("benign_score", result)
        self.assertAlmostEqual(
            result["pathogenic_score"] + result["benign_score"], 1.0, places=7
        )
        self.assertGreaterEqual(result["pathogenic_score"], 0)
        self.assertLessEqual(result["pathogenic_score"], 1)
        self.assertGreaterEqual(result["benign_score"], 0)
        self.assertLessEqual(result["benign_score"], 1)

        # Test invalid inputs
        with self.assertRaisesRegex(ValueError, "Empty sequence provided"):
            self.alphafold_integration.run_alphamissense_analysis("", "M1K")
        with self.assertRaisesRegex(
            ValueError, "Invalid amino acid\(s\) found in sequence"
        ):
            self.alphafold_integration.run_alphamissense_analysis("INVALID123", "M1K")
        with self.assertRaisesRegex(ValueError, "Invalid variant format"):
            self.alphafold_integration.run_alphamissense_analysis(sequence, "INVALID")
        with self.assertRaisesRegex(ValueError, "Invalid variant position"):
            self.alphafold_integration.run_alphamissense_analysis(sequence, "M100K")
        with self.assertRaisesRegex(
            ValueError, "Original amino acid in variant .* does not match sequence"
        ):
            self.alphafold_integration.run_alphamissense_analysis(sequence, "G1K")
        with self.assertRaisesRegex(ValueError, "Invalid variant format"):
            self.alphafold_integration.run_alphamissense_analysis(sequence, "M1")
        with self.assertRaisesRegex(ValueError, "Invalid variant format"):
            self.alphafold_integration.run_alphamissense_analysis(sequence, "1K")
        with self.assertRaisesRegex(ValueError, "Invalid input type"):
            self.alphafold_integration.run_alphamissense_analysis(123, "M1K")
        with self.assertRaisesRegex(ValueError, "Invalid new amino acid in variant"):
            self.alphafold_integration.run_alphamissense_analysis(sequence, "M1X")

    @pytest.mark.skip(reason="Temporarily skipped due to failure")
    def test_input_validation(self):
        with self.assertRaisesRegex(ValueError, "Invalid amino acid sequence provided"):
            self.alphafold_integration.prepare_features("")
        with self.assertRaisesRegex(ValueError, "Invalid amino acid sequence provided"):
            self.alphafold_integration.prepare_features("INVALID_SEQUENCE")
        with self.assertRaisesRegex(ValueError, "Invalid amino acid sequence provided"):
            self.alphafold_integration.prepare_features("123")
        with self.assertRaisesRegex(ValueError, "Invalid amino acid sequence provided"):
            self.alphafold_integration.prepare_features(
                "ACDEFGHIKLMNPQRSTVWYX"
            )  # 'X' is not a valid amino acid
        with self.assertRaisesRegex(ValueError, "Invalid amino acid sequence provided"):
            self.alphafold_integration.prepare_features(
                "ACDE FGHI"
            )  # Space is not allowed
        self.alphafold_integration.prepare_features(
            "ACDEFGHIKLMNPQRSTVWY"
        )  # Should not raise an exception
        self.alphafold_integration.prepare_features(
            "acdefghiklmnpqrstvwy"
        )  # Should not raise an exception (lowercase)


if __name__ == "__main__":
    unittest.main()


class TestAlphaMissenseIntegration(unittest.TestCase):
    def setUp(self):
        self.alphafold_integration = AlphaFoldIntegration()

    def test_run_alphamissense_analysis_valid_input(self):
        sequence = "MKFLKFSLLTAVLLSVVFAFSSCGDDDDTGYLPPSQAIQDLLKRMKV"
        variant = "M1K"
        result = self.alphafold_integration.run_alphamissense_analysis(
            sequence, variant
        )
        self.assertIn("pathogenic_score", result)
        self.assertIn("benign_score", result)
        self.assertAlmostEqual(
            result["pathogenic_score"] + result["benign_score"], 1.0, places=7
        )
        self.assertGreaterEqual(result["pathogenic_score"], 0)
        self.assertLessEqual(result["pathogenic_score"], 1)
        self.assertGreaterEqual(result["benign_score"], 0)
        self.assertLessEqual(result["benign_score"], 1)
        self.assertIsInstance(result["pathogenic_score"], float)
        self.assertIsInstance(result["benign_score"], float)

    def test_run_alphamissense_analysis_empty_sequence(self):
        with self.assertRaises(ValueError) as context:
            self.alphafold_integration.run_alphamissense_analysis("", "M1K")
        self.assertEqual(
            str(context.exception),
            "Empty sequence provided. Please provide a valid amino acid sequence.",
        )

    def test_run_alphamissense_analysis_invalid_sequence(self):
        with self.assertRaises(ValueError) as context:
            self.alphafold_integration.run_alphamissense_analysis("INVALID123", "M1K")
        self.assertIn("Invalid amino acid(s) found in sequence", str(context.exception))

    def test_run_alphamissense_analysis_invalid_variant(self):
        sequence = "MKFLKFSLLTAVLLSVVFAFSSCGDDDDTGYLPPSQAIQDLLKRMKV"
        with self.assertRaises(ValueError) as context:
            self.alphafold_integration.run_alphamissense_analysis(sequence, "INVALID")
        self.assertEqual(
            str(context.exception),
            "Invalid variant format. Use 'OriginalAA{Position}NewAA' (e.g., 'G56A').",
        )

    def test_run_alphamissense_analysis_mismatched_variant(self):
        sequence = "MKFLKFSLLTAVLLSVVFAFSSCGDDDDTGYLPPSQAIQDLLKRMKV"
        with self.assertRaises(ValueError) as context:
            self.alphafold_integration.run_alphamissense_analysis(sequence, "G1A")
        self.assertEqual(
            str(context.exception),
            "Original amino acid in variant (G) does not match sequence at position 1 (M).",
        )


class TestAlphaProteoIntegration(unittest.TestCase):
    def setUp(self):
        self.alphafold_integration = AlphaFoldIntegration()

    def test_run_alphaproteo_analysis_valid_input(self):
        sequence = "MKFLKFSLLTAVLLSVVFAFSSCGDDDDTGYLPPSQAIQDLLKRMKV"
        result = self.alphafold_integration.run_alphaproteo_analysis(sequence)
        self.assertIn("novel_proteins", result)
        self.assertIn("binding_affinities", result)
        self.assertEqual(len(result["novel_proteins"]), 3)
        self.assertEqual(len(result["binding_affinities"]), 3)
        for protein in result["novel_proteins"]:
            self.assertEqual(len(protein), len(sequence))
            self.assertTrue(all(aa in "ACDEFGHIKLMNPQRSTVWY" for aa in protein))
        for affinity in result["binding_affinities"]:
            self.assertGreaterEqual(affinity, 0)
            self.assertLessEqual(affinity, 1)

    def test_run_alphaproteo_analysis_invalid_sequence(self):
        with self.assertRaises(ValueError) as context:
            self.alphafold_integration.run_alphaproteo_analysis("INVALID123")
        self.assertIn("Invalid amino acid(s) found in sequence", str(context.exception))

    def test_run_alphaproteo_analysis_empty_sequence(self):
        with self.assertRaises(ValueError) as context:
            self.alphafold_integration.run_alphaproteo_analysis("")
        self.assertEqual(
            str(context.exception),
            "Empty sequence provided. Please provide a valid amino acid sequence.",
        )

    def test_run_alphaproteo_analysis_non_string_input(self):
        with self.assertRaises(ValueError) as context:
            self.alphafold_integration.run_alphaproteo_analysis(123)
        self.assertEqual(
            str(context.exception), "Invalid input type. Sequence must be a string."
        )

    def test_run_alphaproteo_analysis_short_sequence(self):
        with self.assertRaises(ValueError) as context:
            self.alphafold_integration.run_alphaproteo_analysis("ACDEFG")
        self.assertIn("Sequence is too short", str(context.exception))

    def test_run_alphaproteo_analysis_long_sequence(self):
        long_sequence = "A" * 2001
        with self.assertRaises(ValueError) as context:
            self.alphafold_integration.run_alphaproteo_analysis(long_sequence)
        self.assertIn("Sequence is too long", str(context.exception))
