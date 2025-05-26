#!/bin/bash
#
# OpenFHE-NumPy Installation Script Template
# -----------------------------------------
# This script builds and installs OpenFHE-NumPy in a custom location.
# Customize the paths below to match your environment.

# Exit immediately if any command fails
set -e

#=====================================================================
# CONFIGURATION SECTION - MODIFY THESE VARIABLES FOR YOUR ENVIRONMENT
#=====================================================================

# Path to OpenFHE C++ installation
OPENFHE_DIR="/path/to/openfhe/installation"

# Path to OpenFHE Python wrapper installation
OPENFHE_PYTHON_DIR="/path/to/openfhe_python/installation"

# Where to install OpenFHE-NumPy
INSTALL_DIR="/path/to/openfhe_numpy/installation"

# Python executable to use (default: system Python3)
PYTHON_EXECUTABLE=$(which python3)

# Build directory (default: ./build)
BUILD_DIR="$(pwd)/build"

#=====================================================================
# BUILD SCRIPT - GENERALLY NO NEED TO MODIFY BELOW THIS LINE
#=====================================================================

# Print configuration information
echo "========================================================"
echo "OpenFHE-NumPy Installation Configuration:"
echo "========================================================"
echo "OpenFHE C++ location: ${OPENFHE_DIR}"
echo "OpenFHE Python location: ${OPENFHE_PYTHON_DIR}"
echo "Installation target: ${INSTALL_DIR}"
echo "Python executable: ${PYTHON_EXECUTABLE}"
echo "Build directory: ${BUILD_DIR}"
echo "========================================================"

# Validate required dependencies
echo "Checking dependencies..."

# Check if OpenFHE C++ exists
if [ ! -d "${OPENFHE_DIR}" ]; then
    echo "Error: OpenFHE directory does not exist: ${OPENFHE_DIR}"
    echo "Please install OpenFHE C++ or update OPENFHE_DIR variable."
    exit 1
fi

# Check if OpenFHE Python wrapper exists
if [ ! -d "${OPENFHE_PYTHON_DIR}" ]; then
    echo "Error: OpenFHE Python directory does not exist: ${OPENFHE_PYTHON_DIR}"
    echo "Please install OpenFHE Python wrapper or update OPENFHE_PYTHON_DIR variable."
    exit 1
fi

# Create build directory if it doesn't exist
echo "Creating build directory..."
mkdir -p ${BUILD_DIR}
cd ${BUILD_DIR}

# Configure project with CMake
echo "Configuring OpenFHE-NumPy..."
cmake .. \
  -DWITH_CUSTOM_OPENFHE=ON \
  -DCUSTOM_OPENFHE_ROOT=${OPENFHE_DIR} \
  -DCUSTOM_OPENFHE_PYTHON=${OPENFHE_PYTHON_DIR} \
  -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}

# Build the project using all available CPU cores
echo "Building OpenFHE-NumPy..."
make -j$(nproc)

# Install to the specified location
echo "Installing OpenFHE-NumPy to ${INSTALL_DIR}..."
make install

# Installation complete
echo "========================================================"
echo "Installation complete!"
echo "========================================================"
echo "To use OpenFHE-NumPy, add the following to your environment:"
echo "export PYTHONPATH=\"${INSTALL_DIR}:\$PYTHONPATH\""
echo "export LD_LIBRARY_PATH=\"${OPENFHE_DIR}/lib:\$LD_LIBRARY_PATH\""
echo ""
echo "For permanent setup, add these lines to your ~/.bashrc file."
echo "========================================================"