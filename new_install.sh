#!/bin/bash

# Exit on error
set -e

# Define variables
OPENFHE_DIR="/home/tango/x/work/share_dir/openfhe"
OPENFHE_PYTHON_DIR="/home/tango/x/work/share_dir/openfhe_python"
INSTALL_DIR="/home/tango/x/work/share_dir/openfhe_numpy" 
PYTHON_EXECUTABLE=$(which python3)
BUILD_DIR="$(pwd)/build"

# Print configuration
echo "OpenFHE is installed at: ${OPENFHE_DIR}"
echo "OpenFHE Python is installed at: ${OPENFHE_PYTHON_DIR}"
echo "OpenFHE NumPy will be installed at: ${INSTALL_DIR}"

# Check if directories exist
if [ ! -d "${OPENFHE_DIR}" ]; then
    echo "Error: OpenFHE directory does not exist: ${OPENFHE_DIR}"
    exit 1
fi

if [ ! -d "${OPENFHE_PYTHON_DIR}" ]; then
    echo "Error: OpenFHE Python directory does not exist: ${OPENFHE_PYTHON_DIR}"
    exit 1
fi

# Create build directory
mkdir -p ${BUILD_DIR}
cd ${BUILD_DIR}

# Configure with custom OpenFHE paths and install to local directory
echo "Configuring OpenFHE-NumPy..."
cmake .. \
  -DWITH_CUSTOM_OPENFHE=ON \
  -DCUSTOM_OPENFHE_ROOT=${OPENFHE_DIR} \
  -DCUSTOM_OPENFHE_PYTHON=${OPENFHE_PYTHON_DIR} \
  -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}

# Build using all available cores
echo "Building OpenFHE-NumPy..."
make -j$(nproc)

# Install to specified prefix
echo "Installing OpenFHE-NumPy to ${INSTALL_DIR}..."
make install

echo "Installation complete!"
echo "To use OpenFHE-NumPy, add the following to your environment:"
echo "export PYTHONPATH=${INSTALL_DIR}:\$PYTHONPATH"
