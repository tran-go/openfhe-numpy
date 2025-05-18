"""
OpenFHE-NumPy Test Framework

This module provides a framework for testing OpenFHE-NumPy operations
with different parameter sets and input configurations.
"""

# Standard library imports
import csv
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from contextlib import contextmanager

# Third-party imports
import numpy as np
import unittest

# Local imports
from openfhe import (
    CCParamsCKKSRNS,
    GenCryptoContext,
    PKESchemeFeature,
    UNIFORM_TERNARY,
    FIXEDAUTO,
    FLEXIBLEAUTOEXT,
    FLEXIBLEAUTO,
    FIXEDMANUAL,
    HYBRID,
    BV,
    HEStd_128_classic,
    HEStd_192_classic,
    HEStd_256_classic,
    HEStd_NotSet,
)
import openfhe_numpy as onp


#  absolute paths
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PARAMS_CSV = os.path.join(TESTS_DIR, "ckks_params.csv")
# PARAMS_CSV = os.path.join(TESTS_DIR, "ckks_params_high_depth.csv")
LOG_DIR = Path(os.path.join(TESTS_DIR, "logs"))
DEBUG_LOG_DIR = Path(os.path.join(TESTS_DIR, "debug_logs"))

# Cache
_crypto_context_cache = {}


@contextmanager
def suppress_stdout():
    """Context manager to suppress stdout temporarily."""
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")
    try:
        yield
    finally:
        sys.stdout.close()
        sys.stdout = original_stdout


def generate_random_array(
    rows: int,
    cols: Optional[int] = None,
    low: float = 0,
    high: float = 10,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate a random array with specified dimensions.

    Parameters
    ----------
    rows : int
        Number of rows
    cols : int, optional
        Number of columns (defaults to rows if None)
    low : float, optional
        Minimum value in array
    high : float, optional
        Maximum value in array
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    np.ndarray
        Randomly generated array
    """
    rng = np.random.default_rng(seed)
    if cols is None:
        cols = rows
    if cols == 1:
        return rng.uniform(low, high, size=rows)
    return rng.uniform(low, high, size=(rows, cols))


def load_ckks_params() -> List[Dict[str, Any]]:
    """
    Load CKKS parameters from CSV file.

    Returns
    -------
    List[Dict[str, Any]]
        List of parameter dictionaries
    """
    try:
        with open(PARAMS_CSV, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            params_list = []
            for row in reader:
                params = {
                    "ptModulus": int(row["ptModulus"]),
                    "digitSize": int(row["digitSize"]),
                    "standardDeviation": float(row["standardDeviation"]),
                    "secretKeyDist": row["secretKeyDist"],
                    "maxRelinSkDeg": int(row["maxRelinSkDeg"]),
                    "ksTech": row["ksTech"],
                    "scalTech": row["scalTech"],
                    "firstModSize": int(row["firstModSize"]),
                    "batchSize": int(row["batchSize"]),
                    "numLargeDigits": int(row["numLargeDigits"]),
                    "multiplicativeDepth": int(row["multiplicativeDepth"]),
                    "scalingModSize": int(row["scalingModSize"]),
                    "securityLevel": row["securityLevel"],
                    "ringDim": int(row["ringDim"]),
                }
                params_list.append(params)
            return params_list
    except FileNotFoundError:
        raise FileNotFoundError(f"Parameter file {PARAMS_CSV} not found")


def gen_crypto_context_from_params(p: Dict[str, Any]) -> Tuple[Any, Any]:
    """
    Generate a crypto context and keys from parameters.

    Parameters
    ----------
    p : Dict[str, Any]
        Dictionary of parameters

    Returns
    -------
    Tuple[Any, Any]
        CryptoContext and keys
    """
    # params = p.copy()

    # # Check for HYBRID key switching compatibility
    # if params["ksTech"] == "HYBRID":
    #     estimated_towers = params["multiplicativeDepth"] + 1
    #     original = params["numLargeDigits"]
    #     found_compatible = False

    #     # If towers can't be distributed evenly, fix the configuration
    #     if estimated_towers % params["numLargeDigits"] != 0:
    #         # Try common divisors
    #         for candidate in [1, 2, 4]:
    #             if estimated_towers % candidate == 0:
    #                 params["numLargeDigits"] = candidate
    #                 print(
    #                     f"[info] Adjusted numLargeDigits from {original} to {candidate} for compatibility with HYBRID (towers={estimated_towers})"
    #                 )
    #                 found_compatible = True
    #                 break

    #         # If no compatible value found, switch to BV technique
    #         if not found_compatible:
    #             print(
    #                 f"[warn] HYBRID key switching incompatible with {estimated_towers} towers and {original} digits. Switching to BV."
    #             )
    #             params["ksTech"] = "BV"

    parameters = CCParamsCKKSRNS()
    parameters.SetRingDim(p["ringDim"])
    parameters.SetMultiplicativeDepth(p["multiplicativeDepth"])
    parameters.SetScalingModSize(p["scalingModSize"])
    parameters.SetBatchSize(p["batchSize"])
    parameters.SetFirstModSize(p["firstModSize"])
    parameters.SetStandardDeviation(p["standardDeviation"])
    parameters.SetSecretKeyDist(eval(p["secretKeyDist"]))
    parameters.SetScalingTechnique(eval(p["scalTech"]))
    parameters.SetKeySwitchTechnique(eval(p["ksTech"]))
    parameters.SetSecurityLevel(eval(p["securityLevel"]))
    parameters.SetNumLargeDigits(p["numLargeDigits"])
    parameters.SetMaxRelinSkDeg(p["maxRelinSkDeg"])
    parameters.SetDigitSize(p["digitSize"])

    cc = GenCryptoContext(parameters)
    cc.Enable(PKESchemeFeature.PKE)
    cc.Enable(PKESchemeFeature.LEVELEDSHE)
    cc.Enable(PKESchemeFeature.ADVANCEDSHE)

    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)
    cc.EvalSumKeyGen(keys.secretKey)

    return cc, keys


def get_cached_crypto_context(params):
    """Get or create a crypto context for the given parameters.

    Parameters
    ----------
    params : Dict[str, Any]
        Parameter dictionary for crypto context

    Returns
    -------
    Tuple[Any, Any]
        Cached or newly created crypto context and keys
    """
    # Create a hashable key from the parameters dictionary
    param_key = frozenset(params.items())

    if param_key not in _crypto_context_cache:
        # Cache miss - create a new context
        with suppress_stdout():
            cc, keys = gen_crypto_context_from_params(params)
            _crypto_context_cache[param_key] = (cc, keys)

    return _crypto_context_cache[param_key]


def write_test_failure(
    test_name: str,
    input_data: List[np.ndarray],
    expected: np.ndarray,
    result: np.ndarray,
    error: Any,
) -> None:
    """
    Write test failure information to a debug log file.

    Parameters
    ----------
    test_name : str
        Name of the test
    input_data : List[np.ndarray]
        Input data for the test
    expected : np.ndarray
        Expected result
    result : np.ndarray
        Actual result
    error : Any
        Error information
    """
    DEBUG_LOG_DIR.mkdir(exist_ok=True)
    log_path = DEBUG_LOG_DIR / f"{test_name}.log"

    with open(log_path, "w") as f:
        f.write(f"Test Name: {test_name}\n\n")
        f.write("Input:\n")
        for i, x in enumerate(input_data):
            f.write(f"Input {i}:\n")
            f.write(np.array2string(np.array(x), separator=", ") + "\n\n")
        f.write("Expected Result:\n")
        f.write(np.array2string(np.array(expected), separator=", ") + "\n\n")
        f.write("Actual Result:\n")
        f.write(np.array2string(np.array(result), separator=", ") + "\n\n")
        f.write("Error:\n")
        f.write(str(error) + "\n")


def log_test_result(
    name: str,
    test_name: str,
    input_data: List[np.ndarray],
    expected: np.ndarray,
    result: np.ndarray,
    error_size: float,
    passed: bool,
) -> None:
    """
    Log test result to a file.

    Parameters
    ----------
    name : str
        Name of the log file
    test_name : str
        Name of the test
    input_data : List[np.ndarray]
        Input data for the test
    expected : np.ndarray
        Expected result
    result : np.ndarray
        Actual result
    error_size : float
        Size of the error
    passed : bool
        Whether the test passed
    """
    LOG_DIR.mkdir(exist_ok=True)
    log_path = LOG_DIR / f"{name}.log"

    with open(log_path, "a") as f:
        status = "PASS" if passed else "FAIL"
        f.write(f"--- {datetime.now().isoformat()} ---\n")
        f.write(f"Test: {test_name}\n")
        f.write(f"Status: {status}\n")
        f.write("Input:\n")
        for i, x in enumerate(input_data):
            f.write(f"Input {i}:\n")
            f.write(np.array2string(np.array(x), separator=", ") + "\n\n")
        f.write("Expected Result:\n")
        f.write(np.array2string(np.array(expected), separator=", ") + "\n")
        f.write("Actual Result:\n")
        f.write(np.array2string(np.array(result), separator=", ") + "\n\n")


class MainUnittest(unittest.TestCase):
    """Base class for OpenFHE-NumPy tests with dynamic test generation."""

    @classmethod
    def setUpClass(cls):
        """Set up test parameters once before all tests.

        This method ensures test cases are generated when the test class is used.
        Each subclass must implement _generate_test_cases() method.
        """
        if not hasattr(cls, "_test_cases_generated"):
            if not hasattr(cls, "_generate_test_cases"):
                raise NotImplementedError(
                    f"Test class {cls.__name__} must implement _generate_test_cases()"
                )
            cls._generate_test_cases()
            cls._test_cases_generated = True

    @classmethod
    def run_test_summary(cls, test_name=""):
        """Run tests with matrix output suppressed and print summary."""
        from datetime import datetime

        print(f"Running {test_name} tests...")
        start_time = datetime.now()

        # Execute tests with suppressed output
        with suppress_stdout():
            suite = unittest.defaultTestLoader.loadTestsFromTestCase(cls)
            result = unittest.TextTestRunner(verbosity=0).run(suite)

        duration = datetime.now() - start_time

        # Print clean summary
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

        return 0 if result.wasSuccessful() else 1

    @classmethod
    def generate_test_case(
        cls,
        func_name: Callable,
        name: str,
        test_name: str,
        params: Dict[str, Any],
        input_data: List[np.ndarray],
        expected: np.ndarray,
        eps: float = onp.EPSILON,
    ) -> Callable:
        """
        Generate a test case function.

        Parameters
        ----------
        func_name : Callable
            Function to test
        name : str
            Test group name
        test_name : str
            Specific test name
        params : Dict[str, Any]
            Parameters for the crypto context
        input_data : List[np.ndarray]
            Input data for the test
        expected : np.ndarray
            Expected result
        eps : float, optional
            Epsilon for comparison

        Returns
        -------
        Callable
            Test function
        """

        def test(self):
            try:
                with suppress_stdout():  # Suppress matrix output during test
                    result = func_name(params, input_data)
                    flag, error_size = onp.check_equality_matrix(result, expected, eps)

                log_test_result(
                    name, test_name, input_data, expected, result, error_size, passed=flag
                )

                if not flag:
                    write_test_failure(test_name, input_data, expected, result, onp.ERROR_MATCHING)
                    raise AssertionError(f"Matrix equality failed with error {error_size}")
            except Exception as e:
                # Log exceptions to help with debugging
                DEBUG_LOG_DIR.mkdir(exist_ok=True)
                error_log_path = DEBUG_LOG_DIR / f"{test_name}_exception.log"

                with open(error_log_path, "w") as f:
                    f.write(f"Test Name: {test_name}\n\n")
                    f.write("Input:\n")
                    for i, x in enumerate(input_data):
                        f.write(f"Input {i}:\n")
                        f.write(np.array2string(np.array(x), separator=", ") + "\n\n")
                    f.write("Expected Result:\n")
                    f.write(np.array2string(np.array(expected), separator=", ") + "\n\n")
                    f.write("Exception:\n")
                    f.write(f"{type(e).__name__}: {str(e)}\n")

                    # Include traceback information
                    import traceback

                    f.write("\nTraceback:\n")
                    f.write(traceback.format_exc())
                # Re-raise the exception
                raise

        return test
