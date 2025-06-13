# Running Tests for OpenFHE-NumPy

You have several options for running tests in the OpenFHE-NumPy project:

## Option 1: Using run_test.py

```bash
# Run all tests
python run_test.py
```

This script automatically discovers and runs all test files in the tests directory.

## Option 2: Running tests as a Python module

```bash
# Run all tests
python -m tests

# Run a specific test file
python -m tests tests/test_matrix_addition.py

# Run without stopping on first failure
python -m tests --keep-going

# Run with verbose output
python -m tests --verbose
```

## Option 3: Run a specific test class

```bash
# Run a specific test file directly
python -m tests.test_matrix_addition
```

This executes `TestMatrixAddition.run_test_summary()` which runs all tests in that class with a nice summary output.

## Option 4: Using unittest directly

```bash
# Run a specific test method
python -m unittest tests.test_matrix_addition.TestMatrixAddition.test_addition_1_ring_16384_size_2
```

## Environment Setup

Make sure you've set up your environment correctly:

```bash
# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install the package in development mode
pip install -e .
```

This ensures your C++ extension module (`_onp_cpp`) will be properly linked and available during testing.

## Viewing Test Results

Test results are stored in:
- logs: Contains summary logs of all test runs
- debug_logs: Contains detailed logs for any failed tests
