from setuptools import setup, find_packages

setup(
    name="neuroflex",
    version="0.0.1",
    author="NeuroFlex Team",
    author_email="contact@neuroflex.ai",
    description="An advanced neural network framework with interpretability, generalization, robustness, and fairness features",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/neuroflex/neuroflex",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "jax==0.4.10",
        "jaxlib==0.4.10",
        "flax==0.7.0",
        "optax==0.1.5",
        "tensorflow==2.12.0",
        "keras==2.12.0",
        "gym==0.26.2",
        "numpy==1.23.5",
        "scipy==1.10.1",
        "matplotlib==3.7.1",
        "aif360==0.5.0",
    ],
    extras_require={
        "dev": [
            "pytest==7.3.1",
            "flake8==6.0.0",
        ],
    },
)
