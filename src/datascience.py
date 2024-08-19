import jax
import jax.numpy as jnp
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Dict, Any

class DataScience:
    def __init__(self):
        self.scaler = StandardScaler()

    def preprocess_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Preprocess the input data.

        Args:
            data (pd.DataFrame): Input data

        Returns:
            Dict[str, Any]: Preprocessed data and metadata
        """
        # Handle missing values
        data = data.fillna(data.mean())

        # Normalize numerical features
        numerical_features = data.select_dtypes(include=['float64', 'int64']).columns
        data[numerical_features] = self.scaler.fit_transform(data[numerical_features])

        # One-hot encode categorical features
        categorical_features = data.select_dtypes(include=['object']).columns
        data = pd.get_dummies(data, columns=categorical_features)

        # Convert to JAX arrays
        jax_data = jnp.array(data.values)

        return {
            "preprocessed_data": jax_data,
            "feature_names": data.columns.tolist(),
            "numerical_features": numerical_features.tolist(),
            "categorical_features": categorical_features.tolist()
        }

    def split_data(self, data: jnp.ndarray, test_size: float = 0.2, seed: int = 42) -> Dict[str, jnp.ndarray]:
        """
        Split the data into training and testing sets.

        Args:
            data (jnp.ndarray): Input data
            test_size (float): Proportion of the dataset to include in the test split
            seed (int): Random seed for reproducibility

        Returns:
            Dict[str, jnp.ndarray]: Split data
        """
        X_train, X_test = train_test_split(data, test_size=test_size, random_state=seed)
        return {"train": X_train, "test": X_test}

    def compute_statistics(self, data: jnp.ndarray) -> Dict[str, float]:
        """
        Compute basic statistics of the data.

        Args:
            data (jnp.ndarray): Input data

        Returns:
            Dict[str, float]: Computed statistics
        """
        return {
            "mean": jnp.mean(data).item(),
            "std": jnp.std(data).item(),
            "min": jnp.min(data).item(),
            "max": jnp.max(data).item()
        }

    def correlation_analysis(self, data: pd.DataFrame) -> jnp.ndarray:
        """
        Perform correlation analysis on the data.

        Args:
            data (pd.DataFrame): Input data

        Returns:
            jnp.ndarray: Correlation matrix
        """
        return jnp.array(data.corr())

# Example usage
if __name__ == "__main__":
    # Create a sample dataset
    data = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [5, 4, 3, 2, 1],
        'C': ['x', 'y', 'z', 'x', 'y']
    })

    ds = DataScience()

    # Preprocess the data
    preprocessed = ds.preprocess_data(data)
    print("Preprocessed data shape:", preprocessed["preprocessed_data"].shape)
    print("Feature names:", preprocessed["feature_names"])

    # Split the data
    split_data = ds.split_data(preprocessed["preprocessed_data"])
    print("Train data shape:", split_data["train"].shape)
    print("Test data shape:", split_data["test"].shape)

    # Compute statistics
    stats = ds.compute_statistics(preprocessed["preprocessed_data"])
    print("Data statistics:", stats)

    # Perform correlation analysis
    corr_matrix = ds.correlation_analysis(data[['A', 'B']])
    print("Correlation matrix:\n", corr_matrix)
