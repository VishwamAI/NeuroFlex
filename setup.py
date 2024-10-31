from setuptools import setup, find_packages

setup(
    name="NeuroFlex",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pytest',
        'jax',
        'ml_collections',
        'Bio',
        'mne',
        'gymnasium',
    ],
)
