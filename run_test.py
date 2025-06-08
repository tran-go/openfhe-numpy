#!/usr/bin/env python3

import unittest
from pathlib import Path
import importlib
import sys


def discover_and_run_tests():
    tests_dir = Path(__file__).parent / "tests"

    # Load all test modules directly
    test_files = list(tests_dir.glob("test_*.py"))

    # No tests found
    if not test_files:
        print("No test_*.py files found in tests directory!")
        return 1

    # Import each test module
    for test_file in test_files:
        module_name = f"tests.{test_file.stem}"
        try:
            module = importlib.import_module(module_name)
            print(f"Loaded: {module_name}")
        except ImportError as e:
            print(f"Error importing {module_name}: {e}")

    # Create test suite from all TestCase classes
    loader = unittest.defaultTestLoader
    test_suite = loader.discover(start_dir="tests", pattern="test_*.py")

    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(discover_and_run_tests())
