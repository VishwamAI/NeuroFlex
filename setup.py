# MIT License
# 
# Copyright (c) 2024 VishwamAI
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from setuptools import setup, find_packages

setup(
    name="neuroflex",
    version="0.1.4",  # Updated version to 0.1.4
    author="kasinadhsarma",
    author_email="kasinadhsarma@gmail.com",
    description="An advanced neural network framework with interpretability, generalization, robustness, and fairness features",
    long_description=open("README.md").read() + "\n\n## Publishing Instructions\n\nTo publish the package to PyPI, follow these steps:\n\n1. Ensure you have the latest version of `setuptools` and `twine` installed:\n   ```bash\n   pip install --upgrade setuptools twine\n   ```\n\n2. Create the distribution files for the package:\n   ```bash\n   python setup.py sdist bdist_wheel\n   ```\n\n3. Upload the package to PyPI using `twine`:\n   ```bash\n   twine upload dist/*\n   ```\n\n4. Verify the package is published by checking the [PyPI page](https://pypi.org/project/neuroflex/).\n",
    long_description_content_type="text/markdown",
    url="https://github.com/VishwamAI/neuroflex",
    packages=find_packages(include=["NeuroFlex", "NeuroFlex.*"], exclude=["tests*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "absl-py",
        "adversarial-robustness-toolbox",
        "aiohappyeyeballs",
        "aiohttp",
        "aiosignal",
        "alphafold",
        "appdirs",
        "astunparse",
        "attrs",
        "autograd",
        "autoray",
        "biom-format",
        "biopython==1.80",
        "blosc2",
        "cachetools",
        "certifi",
        "charset-normalizer",
        "chex==0.1.6",
        "CHSPy",
        "click==8.0.4",
        "contextlib2",
        "deap",
        "decorator",
        "dill",
        "dm-haiku==0.0.9",
        "dm-sonnet",
        "dm-tree",
        "docker",
        "dpath",
        "ete3",
        "etils",
        "filelock",
        "flatbuffers",
        "flax==0.6.10",
        "frozenlist",
        "fsspec",
        "gast<=0.4.0,>=0.2.1",
        "google-pasta",
        "grpcio",
        "gym",
        "h5py",
        "huggingface-hub",
        "humanize",
        "idna",
        "immutabledict",
        "importlib_resources",
        "iniconfig",
        "jax==0.4.13",
        "jaxlib==0.4.13",
        "Jinja2",
        "jitcdde",
        "jitcxde_common",
        "jmp",
        "joblib",
        "keras<2.13,>=2.12.0",
        "lale==0.6.0",
        "libclang",
        "lightning-utilities",
        "llvmlite",
        "Markdown",
        "markdown-it-py",
        "MarkupSafe",
        "mdurl",
        "ml-collections",
        "ml-dtypes",
        "mpmath",
        "msgpack",
        "multidict",
        "namex",
        "natsort",
        "ndindex",
        "nest-asyncio",
        "networkx",
        "neurolib",
        "numba",
        "numexpr",
        "numpy==1.23.5",
        "nvidia-cublas-cu11",
        "nvidia-cuda-cupti-cu11",
        "nvidia-cuda-nvrtc-cu11",
        "nvidia-cuda-runtime-cu11",
        "nvidia-cudnn-cu11",
        "nvidia-cufft-cu11",
        "nvidia-curand-cu11",
        "nvidia-cusolver-cu11",
        "nvidia-cusparse-cu11",
        "nvidia-nccl-cu11",
        "nvidia-nvtx-cu11",
        "openmm==8.1.1",
        "opt-einsum",
        "optax==0.1.5",
        "optree",
        "orbax-checkpoint==0.2.1",
        "packaging",
        "pandas",
        "patsy",
        "pbr",
        "PennyLane==0.30.0",
        "PennyLane_Lightning==0.30.0",
        "pluggy",
        "protobuf==3.20.3",
        "psutil",
        "py-cpuinfo",
        "Pygments",
        "pypet",
        "pytest",
        "python-dateutil",
        "pytorch-lightning==1.9.5",
        "pytz",
        "PyYAML",
        "qiskit>=1.0.0",
        "qiskit-aer>=0.12.0",
        "qiskit-algorithms>=0.2.0",
        "regex",
        "requests",
        "rich",
        "rustworkx>=0.13.0",
        "safetensors",
        "scikit-bio>=0.5.8,<0.6.0",
        "scikit-learn==1.1.1",
        "scipy==1.9.0",
        "six",
        "statsmodels",
        "stevedore",
        "symengine",
        "sympy",
        "tables",
        "tabulate",
        "tensorboard<2.13,>=2.12",
        "tensorboard-data-server",
        "tensorflow==2.12.0",
        "tensorflow-io-gcs-filesystem",
        "tensorstore",
        "termcolor",
        "threadpoolctl",
        "tokenizers==0.13.3",
        "toml",
        "toolz",
        "torch==1.13.1",
        "torchmetrics",
        "torchvision==0.14.1",
        "tqdm",
        "transformers==4.30.2",
        "triton==2.0.0",
        "typing_extensions>=3.7.4,<4.6.0",
        "tzdata",
        "urllib3",
        "Werkzeug",
        "wrapt",
        "xarray",
        "yarl",
        "zipp",
        "PyQt5",
        "einops",
        "prophet",
        "psutil",
        "aif360",

        "PyWavelets",
        "filterpy"
    ],
    extras_require={
        'dev': [
            'pytest',
            'flake8',
        ],
    },
)
