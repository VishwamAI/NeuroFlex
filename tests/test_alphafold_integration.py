import sys
import os
import traceback

print("Starting AlphaFold integration test")

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    print("Attempting to import AlphaFoldIntegration")
    from NeuroFlex.scientific_domains import AlphaFoldIntegration
    print("AlphaFold integration imported successfully")

    print("Attempting to create an instance of AlphaFoldIntegration")
    alphafold_instance = AlphaFoldIntegration()
    print("AlphaFoldIntegration instance created successfully")

    print("Attempting to call is_model_ready method")
    is_ready = alphafold_instance.is_model_ready()
    print(f"AlphaFoldIntegration is_model_ready method called successfully. Result: {is_ready}")

    if is_ready:
        print("Attempting to call prepare_features method")
        sequence = "ACGT"
        features = alphafold_instance.prepare_features(sequence)
        print("AlphaFoldIntegration prepare_features method called successfully")
        print(f"Prepared features: {features}")
    else:
        print("Model is not ready. Skipping prepare_features method call.")

except ImportError as e:
    print(f"Error importing AlphaFoldIntegration: {e}")
    print("Traceback:")
    traceback.print_exc()
except AttributeError as e:
    print(f"Error creating AlphaFoldIntegration instance or calling method: {e}")
    print("Traceback:")
    traceback.print_exc()
except Exception as e:
    print(f"Unexpected error: {e}")
    print("Traceback:")
    traceback.print_exc()

print("AlphaFold integration test completed")
