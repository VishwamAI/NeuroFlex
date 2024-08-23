import jax
import jax.numpy as jnp
import flax.linen as nn
import shap
import matplotlib.pyplot as plt
from typing import List, Tuple, Callable, Any
import logging

class ExplainableAI:
    def __init__(self, model: nn.Module):
        self.model = model
        self.explainer = None
        logging.info("ExplainableAI initialized")

    def setup_shap_explainer(self, background_data: jnp.ndarray):
        """
        Set up the SHAP explainer for the model.

        Args:
            background_data (jnp.ndarray): Background data for the SHAP explainer.
        """
        try:
            def model_predict(x):
                return self.model.apply({'params': self.model.params}, x)

            self.explainer = shap.DeepExplainer(model_predict, background_data)
            logging.info("SHAP explainer set up successfully")
        except Exception as e:
            logging.error(f"Error setting up SHAP explainer: {str(e)}")
            raise

    def explain_instance(self, instance: jnp.ndarray) -> jnp.ndarray:
        """
        Generate SHAP values for a single instance.

        Args:
            instance (jnp.ndarray): The instance to explain.

        Returns:
            jnp.ndarray: SHAP values for the instance.
        """
        if self.explainer is None:
            raise ValueError("SHAP explainer not set up. Call setup_shap_explainer first.")

        try:
            shap_values = self.explainer.shap_values(instance)
            return jnp.array(shap_values)
        except Exception as e:
            logging.error(f"Error generating SHAP values: {str(e)}")
            raise

    def visualize_shap_values(self, instance: jnp.ndarray, feature_names: List[str]):
        """
        Visualize SHAP values for a single instance.

        Args:
            instance (jnp.ndarray): The instance to explain.
            feature_names (List[str]): Names of the features.
        """
        shap_values = self.explain_instance(instance)
        try:
            shap.summary_plot(shap_values, instance, feature_names=feature_names, show=False)
            plt.tight_layout()
            plt.savefig('shap_summary_plot.png')
            plt.close()
            logging.info("SHAP summary plot saved as 'shap_summary_plot.png'")
        except Exception as e:
            logging.error(f"Error visualizing SHAP values: {str(e)}")
            raise

    def get_feature_importance(self, instances: jnp.ndarray) -> jnp.ndarray:
        """
        Calculate global feature importance across multiple instances.

        Args:
            instances (jnp.ndarray): Multiple instances to explain.

        Returns:
            jnp.ndarray: Global feature importance scores.
        """
        try:
            shap_values = self.explainer.shap_values(instances)
            return jnp.abs(shap_values).mean(axis=0)
        except Exception as e:
            logging.error(f"Error calculating feature importance: {str(e)}")
            raise

    def explain_model_decision(self, instance: jnp.ndarray, feature_names: List[str]) -> Tuple[Any, str]:
        """
        Provide a human-readable explanation for a model's decision on a single instance.

        Args:
            instance (jnp.ndarray): The instance to explain.
            feature_names (List[str]): Names of the features.

        Returns:
            Tuple[Any, str]: Model output and a string explanation.
        """
        try:
            model_output = self.model.apply({'params': self.model.params}, instance)
            shap_values = self.explain_instance(instance)

            top_features = jnp.argsort(jnp.abs(shap_values[0]))[-5:]  # Top 5 influential features
            explanation = "The model's decision was most influenced by:\n"
            for idx in reversed(top_features):
                impact = "positively" if shap_values[0][idx] > 0 else "negatively"
                explanation += f"- {feature_names[idx]}, which {impact} impacted the decision\n"

            return model_output, explanation
        except Exception as e:
            logging.error(f"Error explaining model decision: {str(e)}")
            raise

def create_explainable_ai(model: nn.Module) -> ExplainableAI:
    """
    Create an ExplainableAI instance for a given model.

    Args:
        model (nn.Module): The model to explain.

    Returns:
        ExplainableAI: An instance of ExplainableAI.
    """
    return ExplainableAI(model)
