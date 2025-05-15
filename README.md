# OpenFHE-NumPy

A NumPy-inspired framework for homomorphic encryption operations.

## Table of Contents
- <a href="#introduction">Introduction</a>
- <a href="#features">Features</a>
- <a href="#installation">Installation</a>
  - <a href="#prerequisites">Prerequisites</a>
  - <a href="#installing-openfhe-python">Installing OpenFHE-Python</a>
  - <a href="#building-from-source">Building from source</a>
- <a href="#basic-usage">Basic Usage</a>
- <a href="#license">License</a>

## Introduction

OpenFHE-NumPy provides a familiar NumPy-like interface for performing homomorphic encryption operations. It bridges the gap between data scientists familiar with NumPy and the powerful OpenFHE homomorphic encryption library.

## Features
- NumPy-like API for homomorphic encryption
- Efficient matrix operations on encrypted data
- Seamless integration with OpenFHE

## Installation

### Prerequisites
- CMake 3.16 or newer
- C++20 compatible compiler
- OpenFHE 1.2.3 or newer
- [OpenFHE-Python](https://github.com/openfheorg/openfhe-python) (core Python bindings for OpenFHE)
- Python 3.8 or newer
- NumPy

### Installing OpenFHE-Python
Before installing OpenFHE-NumPy, you must install the OpenFHE-Python bindings:

```bash
# Clone OpenFHE-Python
git clone https://github.com/openfheorg/openfhe-python.git
cd openfhe-python

# Follow the installation instructions in its README
# Typically: pip install .
```

### Building from source

```bash
# Clone the repository
git clone https://github.com/yourusername/openfhe-numpy.git
cd openfhe-numpy

# Create build directory
mkdir build && cd build

# Configure with CMake
# Set OpenFHE_DIR to your OpenFHE installation path if needed
cmake .. -DOpenFHE_DIR=/path/to/openfhe/install

# Build the package
make -j$(nproc)

# Install
make install
```

## Basic Usage

```python
import openfhe_numpy as ofhe
import numpy as np

# Set up encryption parameters
context = ofhe.FHETensor.setup_context(
    poly_modulus_degree=8192,
    plain_modulus=786433,
    security_level=128
)

# Create and encrypt data
plaintext = np.array([[1, 2, 3], [4, 5, 6]])
encrypted = ofhe.array(plaintext).encrypt()

# Perform homomorphic operations
result = ofhe.add(encrypted, encrypted)

# Decrypt results
decrypted = result.decrypt().to_array()
```

## License

*[Add license information]*