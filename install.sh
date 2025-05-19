#!/bin/bash

# Exit on error
set -e

# Define variables
INSTALL_PREFIX="/home/tango/x/work/share_dir"
BUILD_DIR="build"
PYTHON_EXECUTABLE=$(which python3)

# Clean and create build directory
rm -rf ${BUILD_DIR}
mkdir -p ${BUILD_DIR}
cd ${BUILD_DIR}

# Configure project (removed comment markers inside the command)
cmake .. \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX}

# Build
cmake --build . --parallel

# Install
cmake --install .

echo "openfhe_matrix installed to: ${INSTALL_PREFIX}"
echo "To use the Python module, run your code with:"
echo "PYTHONPATH=\"${INSTALL_PREFIX}\" python3 your_script.py"

# Create a helper script for running examples
cat > ../run-example.sh << EOL
#!/bin/bash
PYTHONPATH="${INSTALL_PREFIX}" python3 \$@
EOL
chmod +x ../run-example.sh

echo "Or use the generated helper script: ./run-example.sh examples/demo_matrix_addition.py"