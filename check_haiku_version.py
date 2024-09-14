import importlib.metadata

try:
    haiku_version = importlib.metadata.version('dm-haiku')
    print(f"Installed dm-haiku version: '{haiku_version}'")
except importlib.metadata.PackageNotFoundError:
    print("dm-haiku package not found")
