"""OpenFHE-NumPy test suite."""

import argparse
import sys
import os
import unittest
import importlib.util
from datetime import datetime


def _print_test_summary(result, test_name, start_time):
    """Print a formatted test summary.

    Parameters
    ----------
    result : unittest.TestResult
        The test result object
    test_name : str
        Name to display in the summary
    start_time : datetime
        When the test run started
    """
    duration = datetime.now() - start_time

    # Print summary
    print("\n" + "=" * 50)
    print(f"{test_name} Test Summary:")
    print(f"  Total tests:  {result.testsRun}")
    print(f"  Passed:       {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"  Failed:       {len(result.failures)}")
    print(f"  Errors:       {len(result.errors)}")
    print(f"  Duration:     {duration.total_seconds():.2f} seconds")
    print("=" * 50)

    if len(result.failures) > 0 or len(result.errors) > 0:
        print("\nFailed tests:")
        for test, _ in result.failures:
            print(f"  - {test.id()}")
        for test, _ in result.errors:
            print(f"  - {test.id()} (ERROR)")


def run_tests(test_path=None, verbose=False, stop_on_first_failure=True):
    """Run OpenFHE-NumPy tests with configurable options.

    Parameters
    ----------
    test_path : str, optional
        Specific test path to run (default: run all tests)
    verbose : bool, optional
        Run tests in verbose mode
    stop_on_first_failure : bool, optional
        Exit immediately on first test failure
    """
    # Import the suppress_stdout function from main_unittest
    from .main_unittest import suppress_stdout

    # Configure verbosity - keep it low regardless of verbose flag to prevent matrix output
    verbosity = 0

    # Handle single test file case specially
    if test_path and test_path.endswith(".py"):
        # Import the module directly to ensure tests are registered
        module_name = os.path.basename(test_path)[:-3]  # Remove .py

        # Add parent directory to path if needed
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(test_path)))
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)

        # Import and execute module code ONCE to register tests
        spec = importlib.util.spec_from_file_location(module_name, test_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # Only execute once!

        # Look for test classes in the imported module
        test_classes = [
            obj
            for name, obj in vars(module).items()
            if isinstance(obj, type)
            and issubclass(obj, unittest.TestCase)
            and obj != unittest.TestCase
        ]

        if test_classes:
            # Run the tests with clean output
            print(f"Running tests from {test_path}...")
            start_time = datetime.now()

            with suppress_stdout():
                suite = unittest.TestSuite()
                for test_class in test_classes:
                    suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(test_class))

                runner = unittest.TextTestRunner(
                    verbosity=verbosity, failfast=stop_on_first_failure
                )
                result = runner.run(suite)

            # Print summary using the shared function
            _print_test_summary(result, module_name.replace("_", " ").title(), start_time)

            return 0 if result.wasSuccessful() else 1
        else:
            print(f"No tests found in {test_path}")
            return 1

    # Default discovery-based testing
    print("Running OpenFHE-NumPy tests...")
    start_time = datetime.now()

    # Configure test discovery
    loader = unittest.TestLoader()
    if test_path:
        if test_path.endswith(".py"):
            # Extract directory and filename for proper test discovery
            dir_path = os.path.dirname(test_path) or "."
            file_pattern = os.path.basename(test_path)
            tests = loader.discover(start_dir=dir_path, pattern=file_pattern)
        else:
            # Load specific test directory
            tests = loader.discover(start_dir=test_path)
    else:
        # Load all tests
        tests = loader.discover(start_dir=".")

    # Run tests with suppressed output
    with suppress_stdout():
        runner = unittest.TextTestRunner(verbosity=verbosity, failfast=stop_on_first_failure)
        result = runner.run(tests)

    # Print summary using the shared function
    _print_test_summary(result, "OpenFHE-NumPy", start_time)

    # Return appropriate exit code
    return 0 if result.wasSuccessful() else 1


def main():
    """Parse command-line arguments and run tests."""
    parser = argparse.ArgumentParser(description="Run OpenFHE-NumPy tests")
    parser.add_argument("test_path", nargs="?", help="Specific test path to run")
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Display more test information (not recommended due to matrix output)",
    )
    parser.add_argument(
        "--keep-going",
        "-k",
        action="store_false",
        dest="stop_on_first_failure",
        help="Continue testing after first failure",
    )

    args = parser.parse_args()
    exit_code = run_tests(args.test_path, args.verbose, args.stop_on_first_failure)
    sys.exit(exit_code)


if __name__ == "__main__":
    """
    # Run all tests with default settings
    python -m tests

    # Run a specific test file
    python -m tests tests/test_matrix_addition.py

    # Run without stopping on first failure  
    python -m tests --keep-going
    """
    main()
