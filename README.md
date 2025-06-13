# OpenFHE-NumPy

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Python Versions](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![OpenFHE Version](https://img.shields.io/badge/OpenFHE-1.2.3%2B-green)](https://github.com/openfheorg/openfhe-development)

A NumPy-like API for homomorphic encryption operations, built on top of OpenFHE. This library enables data scientists and machine learning practitioners to perform computations on encrypted data using familiar NumPy syntax.

The project is currently in development, with a planned release shortly.

## Table of Contents
- [OpenFHE-NumPy](#openfhe-numpy)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Project Structure](#project-structure)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Method 1: Building from Source](#method-1-building-from-source)
      - [Step 1: Install OpenFHE (Required)](#step-1-install-openfhe-required)
      - [Step 2: Install OpenFHE-Python from Source](#step-2-install-openfhe-python-from-source)
      - [Step 3: Install OpenFHE-NumPy from Source](#step-3-install-openfhe-numpy-from-source)
    - [Method 2: Installation with pip (for Ubuntu)](#method-2-installation-with-pip-for-ubuntu)
      - [Step 1: Install OpenFHE (Required)](#step-1-install-openfhe-required-1)
      - [Step 2: Install OpenFHE-Python](#step-2-install-openfhe-python)
      - [Step 3: Install OpenFHE-NumPy via pip](#step-3-install-openfhe-numpy-via-pip)
    - [Development Setup](#development-setup)
    - [Running Tests](#running-tests)
  - [Example Usage](#example-usage)
  - [Available Operations](#available-operations)
  - [Documentation](#documentation)
  - [Examples](#examples)
  - [Contributing](#contributing)
  - [License](#license)

## Features

- **NumPy-compatible API**: Use familiar NumPy-style syntax for homomorphic operations
- **Encrypted tensor manipulation**: Create and manipulate encrypted multi-dimensional arrays
- **Matrix operations**: Perform matrix addition, multiplication, transposition on encrypted data
- **Optimized implementation**: Built on top of OpenFHE for optimal performance
- **Type flexibility**: Support for both encrypted (CT) and plaintext (PT) data types
- **Interoperability**: Seamless integration with Python machine learning workflows

## Project Structure

OpenFHE-NumPy is organized as a hybrid C++/Python project with the following structure:

```
openfhe-numpy/
├── core/                # C++ implementation
│   ├── include/         # Public headers
│   │   └── openfhe_numpy/
│   ├── src/             # C++ source code
│   └── examples/        # C++ examples
├── python/              # Python package
│   └── openfhe_numpy/   # Python module code
│       ├── operations/  # Matrix operations
│       ├── tensor/      # Tensor implementations
│       └── utils/       # Utility functions
├── tests/               # Test suite
├── examples/            # Python examples
├── CMakeLists.txt       # Build configuration
└── dev_mode.sh          # Development environment setup
```

## Installation

### Prerequisites

- **C++ compiler**: Supporting C++17 standard
- **CMake**: Version 3.16 or newer
- **Python**: Version 3.8 or newer
- **NumPy**: Recent version

### Method 1: Building from Source

#### Step 1: Install OpenFHE (Required)

OpenFHE must be installed from source with shared libraries:

<!-- ```bash
# Install system dependencies (Ubuntu)
sudo apt update
sudo apt install build-essential cmake

# Clone OpenFHE
git clone https://github.com/openfheorg/openfhe-development.git
cd openfhe-development

# Create build directory
mkdir build && cd build

# Configure with shared libraries enabled
cmake .. -DBUILD_SHARED_LIBS=ON

# Build and install
make -j$(nproc)
sudo make install

# Set environment variable (add to ~/.bashrc for persistence)
echo 'export OpenFHE_DIR=/usr/local/lib/cmake/OpenFHE' >> ~/.bashrc
source ~/.bashrc
``` -->

#### Step 2: Install OpenFHE-Python from Source

Refer to the instruction from OpenFHE-Python

<!-- ```bash
# Clone OpenFHE-Python
git clone https://github.com/openfheorg/openfhe-python.git
cd openfhe-python

# Install from source
cmake -S . -B build \
  -DCMAKE_INSTALL_PREFIX=/path/to/openfhe-python \
  -DCMAKE_PREFIX_PATH=/path/to/openfhe

# Build the package
make

# Install
make install
``` -->

#### Step 3: Install OpenFHE-NumPy from Source

```bash
# Clone the repository
git clone https://github.com/openfheorg/openfhe-numpy.git
cd openfhe-numpy

# Create build directory
mkdir build && cd build

# Configure with CMake
cmake .. \
  -DWITH_CUSTOM_OPENFHE=ON \
  -DCUSTOM_OPENFHE_ROOT=/path/to/openfhe \
  -DCUSTOM_OPENFHE_PYTHON=/path/to/openfhe-python \
  -DCMAKE_INSTALL_PREFIX=/path/to/openfhe_numpy

# Build the package
make

# Install
make install
```

### Method 2: Installation with pip (for Ubuntu)

#### Step 1: Install OpenFHE (Required)

Follow the same OpenFHE installation instructions from Method 1, Step 1.

#### Step 2: Install OpenFHE-Python

```bash
# Install OpenFHE-Python from PyPI
pip install openfhe-python
```

#### Step 3: Install OpenFHE-NumPy via pip

```bash
# Install from PyPI (once available)
pip install openfhe-numpy

# Or install directly from GitHub
pip install git+https://github.com/openfheorg/openfhe-numpy.git
```

### Development Setup

For development without installation, use the provided script:

```bash
# Clone the repository
git clone https://github.com/openfheorg/openfhe-numpy.git
cd openfhe-numpy

# Make the script executable
chmod +x dev_mode.sh

# Run the script (IMPORTANT: use source to preserve environment variables)
source ./dev_mode.sh
```

This will:
1. Build the C++ extension
2. Create a development environment in dev_build
3. Set up necessary environment variables (PYTHONPATH, LD_LIBRARY_PATH)

After running this script, you can run tests and import the package in your Python code without formal installation.

### Running Tests

```bash
# Run all tests
python -m tests

# Run a specific test
python -m tests.test_matrix_addition
```

## Example Usage

```python
import numpy as np
import openfhe_numpy as onp
from openfhe import (
    CCParamsCKKSRNS,
    GenCryptoContext,
    PKESchemeFeature,
    FIXEDAUTO,
    HYBRID,
    UNIFORM_TERNARY,
)

# Initialize CKKS context
params = CCParamsCKKSRNS()
params.SetMultiplicativeDepth(7)
params.SetScalingModSize(59)
params.SetFirstModSize(60)
params.SetScalingTechnique(FIXEDAUTO)
params.SetSecretKeyDist(UNIFORM_TERNARY)

cc = GenCryptoContext(params)
cc.Enable(PKESchemeFeature.PKE)
cc.Enable(PKESchemeFeature.LEVELEDSHE)
cc.Enable(PKESchemeFeature.ADVANCEDSHE)

# Generate keys
keys = cc.KeyGen()
cc.EvalMultKeyGen(keys.secretKey)
cc.EvalSumKeyGen(keys.secretKey)

# Create matrix and encrypt it
A = np.array([[1, 2], [3, 4]])

ring_dim = cc.GetRingDimension()
total_slots = ring_dim // 2

# Encrypt with OpenFHE-NumPy
tensor_A = onp.array(cc, A, total_slots, public_key=keys.publicKey)

# Generate keys
onp.EvalSquareMatMultRotateKeyGen(keys.secretKey, tensor_A.ncols)

# Perform encrypted operations
tensor_product = tensor_A @ tensor_A  # Matrix multiplication
tensor_sum = onp.add(tensor_A, tensor_A)  # Element-wise addition

# Decrypt results
decrypted_product = tensor_product.decrypt(keys.secretKey, unpack_type = True)
decrypted_sum = tensor_sum.decrypt(keys.secretKey, unpack_type = True)

print("Result of A @ A:")
print(decrypted_product)

print("Result of A + A:")
print(decrypted_sum)
```

## Available Operations

OpenFHE-NumPy currently supports the following operations:

| Operation   | Description                 | Example                         |
| ----------- | --------------------------- | ------------------------------- |
| `add`       | Element-wise addition       | `onp.add(a, b)` or `a + b`      |
| `subtract`  | Element-wise subtraction    | `onp.subtract(a, b)` or `a - b` |
| `multiply`  | Element-wise multiplication | `onp.multiply(a, b)` or `a * b` |
| `matmul`    | Matrix multiplication       | `onp.matmul(a, b)` or `a @ b`   |
| `transpose` | Matrix transposition        | `onp.transpose(a)`              |
| `cumsum`    | Cumulative sum along axis   | `onp.cumsum(a, axis)`           |
| `power`     | Element-wise power          | `onp.power(a, exp)`             |
| `dot`       | Dot product                 | `onp.dot(a, b)`                 |

## Documentation

For detailed documentation on the API, please visit our [documentation site](https://openfheorg.github.io/openfhe-numpy).

## Examples

We provide several examples showcasing the library's functionality:

- [Matrix Addition](https://github.com/openfheorg/openfhe-numpy/blob/main/examples/demo_matrix_addition.py)
- [Matrix Transpose](https://github.com/openfheorg/openfhe-numpy/blob/main/examples/demo_matrix_transpose.py)
- [Matrix-Vector Multiplication](https://github.com/openfheorg/openfhe-numpy/blob/main/examples/demo_matvec_product.py)
- [Square Matrix Multiplication](https://github.com/openfheorg/openfhe-numpy/blob/main/examples/demo_square_matrix_product.py)
- [Cumulative Matrix Operations](https://github.com/openfheorg/openfhe-numpy/blob/main/examples/demo_matrix_accumulation.py)


## Contributing
[OpenFHE Development - Contributing Guide](https://openfhe-development.readthedocs.io/en/latest/sphinx_rsts/contributing/contributing_workflow.html)

## License

OpenFHE-NumPy is licensed under the BSD 2-Clause License. See the LICENSE file for details.

---

OpenFHE-NumPy is an independent project and is not officially affiliated with NumPy.
