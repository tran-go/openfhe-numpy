"""
OpenFHE-NumPy Test Framework

This module provides a framework for testing OpenFHE-NumPy operations
with different parameter sets and input configurations.
"""

# Standard library imports
import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Third-party imports
import numpy as np
import unittest

# Local imports
from openfhe import CCParamsCKKSRNS, GenCryptoContext, PKESchemeFeature
import openfhe_numpy as onp


# Constants
PARAMS_CSV = "ckks_params.csv"
LOG_DIR = Path("logs")
DEBUG_LOG_DIR = Path("debug_logs")


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
    def generate_test_case(
        cls,
        func_name: Callable,
        name: str,
        test_name: str,
        params: Dict[str, Any],
        input_data: List[np.ndarray],
        expected: np.ndarray,
        eps: float = onp.matlib.EPSILON,
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
            result = func_name(params, input_data)
            flag, error_size = onp.matlib.check_equality_matrix(result, expected, eps)

            log_test_result(name, test_name, input_data, expected, result, error_size, passed=flag)

            if not flag:
                write_test_failure(
                    test_name, input_data, expected, result, onp.matlib.FHEErrorCodes.ERROR_MATCHING
                )
                raise AssertionError(f"Matrix equality failed with error {error_size}")

        return test


# Add standard test runner
if __name__ == "__main__":
    unittest.main()
