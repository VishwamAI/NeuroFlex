import sys
import traceback

def test_import(module_name):
    try:
        __import__(module_name)
        print(f"Successfully imported {module_name}")
    except ImportError as e:
        print(f"Error importing {module_name}: {str(e)}")
        print("Traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    print("Testing NeuroFlex imports:")
    test_import("NeuroFlex")

    print("\nTesting core_neural_networks imports:")
    core_nn_modules = [
        "NeuroFlex.core_neural_networks",
        "NeuroFlex.core_neural_networks.model",
        "NeuroFlex.core_neural_networks.cnn",
        "NeuroFlex.core_neural_networks.lstm",
        "NeuroFlex.core_neural_networks.rnn",
        "NeuroFlex.core_neural_networks.machinelearning",
        "NeuroFlex.core_neural_networks.advanced_thinking",
        "NeuroFlex.core_neural_networks.jax",
        "NeuroFlex.core_neural_networks.pytorch",
    ]

    for module in core_nn_modules:
        test_import(module)

    print("\nTesting AlphaFold integration:")
    test_import("NeuroFlex.scientific_domains.alphafold_integration")
