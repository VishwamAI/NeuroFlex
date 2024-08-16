import jax.numpy as jnp
from typing import List, Dict, Any
from alphafold.model import model
from alphafold.common import protein
from alphafold.data import pipeline

class AlphaFoldIntegration:
    def __init__(self):
        self.model_runner = None
        self.feature_dict = None

    def setup_model(self, model_params: Dict[str, Any]):
        """
        Set up the AlphaFold model with given parameters.

        Args:
            model_params (Dict[str, Any]): Parameters for the AlphaFold model.
        """
        self.model_runner = model.RunModel(model_params)

    def prepare_features(self, sequence: str):
        """
        Prepare feature dictionary for AlphaFold prediction.

        Args:
            sequence (str): Amino acid sequence.

        Returns:
            Dict: Feature dictionary for AlphaFold.
        """
        self.feature_dict = pipeline.make_sequence_features(sequence)
        self.feature_dict.update(pipeline.make_msa_features([sequence]))

    def predict_structure(self) -> protein.Protein:
        """
        Predict protein structure using AlphaFold.

        Returns:
            protein.Protein: Predicted protein structure.
        """
        if self.model_runner is None or self.feature_dict is None:
            raise ValueError("Model or features not set up. Call setup_model() and prepare_features() first.")

        prediction_result = self.model_runner.predict(self.feature_dict)
        return protein.from_prediction(prediction_result)

    def get_plddt_scores(self) -> jnp.ndarray:
        """
        Get pLDDT (predicted LDDT) scores for the predicted structure.

        Returns:
            jnp.ndarray: Array of pLDDT scores.
        """
        if self.model_runner is None or self.feature_dict is None:
            raise ValueError("Model or features not set up. Call setup_model() and prepare_features() first.")

        prediction_result = self.model_runner.predict(self.feature_dict)
        return prediction_result['plddt']

    def get_predicted_aligned_error(self) -> jnp.ndarray:
        """
        Get predicted aligned error for the structure.

        Returns:
            jnp.ndarray: 2D array of predicted aligned errors.
        """
        if self.model_runner is None or self.feature_dict is None:
            raise ValueError("Model or features not set up. Call setup_model() and prepare_features() first.")

        prediction_result = self.model_runner.predict(self.feature_dict)
        return prediction_result['predicted_aligned_error']
