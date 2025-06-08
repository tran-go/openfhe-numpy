import os
import sys
from skbuild import setup
from setuptools import find_packages

# Get custom installation paths from environment variables
openfhe_root = os.environ.get("OPENFHE_ROOT", "")
openfhe_python = os.environ.get("OPENFHE_PYTHON", "")

# Set these for scikit-build environment
if openfhe_root:
    os.environ["CMAKE_PREFIX_PATH"] = openfhe_root
    print(f"Using OpenFHE from: {openfhe_root}")
if openfhe_python:
    print(f"Using OpenFHE-Python from: {openfhe_python}")

# Prepare cmake arguments
cmake_args = [
    "-DCMAKE_VERBOSE_MAKEFILE=ON",  # Add verbosity to debug
]

if openfhe_root:
    cmake_args.extend(["-DWITH_CUSTOM_OPENFHE=ON", f"-DCUSTOM_OPENFHE_ROOT={openfhe_root}"])
if openfhe_python:
    cmake_args.append(f"-DCUSTOM_OPENFHE_PYTHON={openfhe_python}")

# Configuration is mostly from pyproject.toml
setup(
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    cmake_args=cmake_args,
    cmake_install_dir="python",
)
