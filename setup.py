from setuptools import setup, find_packages

def parse_requirements(filename):
    with open(filename) as f:
        return f.read().splitlines()

setup(
    name="neuroflex",
    version="0.1.3",  # Updated version to 0.1.3
    author="kasinadhsarma",
    author_email="kasinadhsarma@gmail.com",
    description="An advanced neural network framework with interpretability, generalization, robustness, and fairness features",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/VishwamAI/neuroflex",
    packages=find_packages(exclude=["tests*"]),
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
    install_requires=parse_requirements('requirements.txt'),
    extras_require={
        'dev': [
            'pytest',
            'flake8',
        ],
    },
)
