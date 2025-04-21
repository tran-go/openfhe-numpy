import ast
import csv
import os
import random
import json
import unittest
import numpy as np
from datetime import datetime

from openfhe import *
from openfhe_matrix import *
import openfhe_numpy as fp
from openfhe_numpy.utils import *
from openfhe_numpy.matlib import *

PARAMS_CSV = "ckks_params.csv"

# Counters for test summary
total_tests = 0
passed_tests = 0


def load_ckks_params():
    """Load CKKS parameters from a CSV file.

    Returns
    -------
    list of dict
        A list of parameter dictionaries.
    """
    with open(PARAMS_CSV, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        return [
            {
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
            for row in reader
        ]


def gen_crypto_context_from_params(p):
    """Generate a CryptoContext and keys from CKKS parameters.

    Parameters
    ----------
    p : dict
        CKKS parameter set.

    Returns
    -------
    tuple
        (CryptoContext, KeyPair)
    """
    parameters = CCParamsCKKSRNS()
    parameters.SetRingDim(p["ringDim"])
    parameters.SetMultiplicativeDepth(p["multiplicativeDepth"])
    parameters.SetScalingModSize(p["scalingModSize"])
    parameters.SetBatchSize(p["batchSize"])
    parameters.SetFirstModSize(p["firstModSize"])
    parameters.SetScalingTechnique(eval(p["scalTech"]))
    parameters.SetKeySwitchTechnique(eval(p["ksTech"]))
    parameters.SetSecurityLevel(eval(p["securityLevel"]))

    cc = GenCryptoContext(parameters)
    cc.Enable(PKESchemeFeature.PKE)
    cc.Enable(PKESchemeFeature.LEVELEDSHE)
    cc.Enable(PKESchemeFeature.ADVANCEDSHE)

    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)
    cc.EvalSumKeyGen(keys.secretKey)

    return cc, keys


def fhe_square_matrix_product(params, a, b):
    """Perform encrypted square matrix multiplication.

    Parameters
    ----------
    params : dict
        CKKS parameter set.
    a, b : list of list
        Input matrices.

    Returns
    -------
    np.ndarray
        Decrypted result after homomorphic matrix multiplication.
    """
    total_slots = params["ringDim"] // 2
    cc, keys = gen_crypto_context_from_params(params)
    ctm_a = fp.array(cc, np.array(a), total_slots, pub_key=keys.publicKey)
    ctm_b = fp.array(cc, np.array(b), total_slots, pub_key=keys.publicKey)
    ct_result = fp.matmul_square(cc, keys, ctm_a, ctm_b)
    return ct_result.decrypt(cc, keys.secretKey)


def generate_random_matrix(n):
    """Generate a square matrix with random integers in [0, 10].

    Parameters
    ----------
    n : int
        Dimension of the square matrix.

    Returns
    -------
    list of list
        Random integer matrix.
    """
    return [[random.randint(0, 10) for _ in range(n)] for _ in range(n)]


def OPENFHE_failure_to_file(test_name, A, B, expected, result, error):
    """Log details of a failed test case to a debug log.

    Parameters
    ----------
    test_name : str
    A, B, expected, result : list of list
    error : Any
    """
    os.makedirs("debug_logs", exist_ok=True)
    with open(f"debug_logs/{test_name}.log", "w") as f:
        f.write(f"Test Name: {test_name}\n\n")
        f.write("Matrix A:\n" + np.array2string(np.array(A), separator=", ") + "\n\n")
        f.write("Matrix B:\n" + np.array2string(np.array(B), separator=", ") + "\n\n")
        f.write("Expected Result:\n" + np.array2string(np.array(expected), separator=", ") + "\n\n")
        f.write("Actual Result:\n" + np.array2string(np.array(result), separator=", ") + "\n\n")
        f.write("Error:\n" + str(error) + "\n")


def OPENFHE_test_result(test_name, A, B, expected, result, error_size, passed):
    """Log the outcome of a test case.

    Parameters
    ----------
    test_name : str
    A, B, expected, result : list of list
    error_size : float
    passed : bool
    """
    os.makedirs("logs", exist_ok=True)
    with open("logs/TestSquareMatrixProduct.log", "a") as f:
        status = "PASS" if passed else "FAIL"
        f.write(f"--- {datetime.now().isoformat()} ---\n")
        f.write(f"Test: {test_name}\nStatus: {status}\n")
        f.write("Matrix A:\n" + np.array2string(np.array(A), separator=", ") + "\n")
        f.write("Matrix B:\n" + np.array2string(np.array(B), separator=", ") + "\n")
        f.write("Expected Result:\n" + np.array2string(np.array(expected), separator=", ") + "\n")
        f.write("Actual Result:\n" + np.array2string(np.array(result), separator=", ") + "\n\n")


def OPENFHE_ckks_parameters(params, filename):
    """Log CKKS parameter dictionary to a file in JSON format.

    Parameters
    ----------
    params : dict
        CKKS parameters.
    filename : str
        Destination file name.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        json.dump(params, f, indent=4)


def rerun_failed_tests():
    """Re-run failed test cases from debug logs."""
    rerun_total = 0
    rerun_passed = 0

    if not os.path.exists("debug_logs"):
        print("No debug logs to rerun.")
        return

    for filename in os.listdir("debug_logs"):
        if not filename.endswith(".log"):
            continue

        with open(os.path.join("debug_logs", filename), "r") as f:
            lines = f.read().split("\n")

        def extract_matrix(lines, keyword):
            start = lines.index(keyword) + 1
            raw = ""
            while start < len(lines) and lines[start].strip():
                raw += lines[start].strip()
                start += 1
            return ast.literal_eval(raw)

        test_name = lines[0].split(":")[1].strip()
        A = extract_matrix(lines, "Matrix A:")
        B = extract_matrix(lines, "Matrix B:")
        expected = extract_matrix(lines, "Expected Result:")

        for param in load_ckks_params():
            if f"ring_{param['ringDim']}" in test_name:
                rerun_total += 1
                print(f"Re-running {test_name}...")
                result = fhe_square_matrix_product(param, A, B)
                flag, error_size = check_equality_matrix(result, expected)
                if flag:
                    rerun_passed += 1
                OPENFHE_test_result(test_name, A, B, expected, result, error_size, passed=flag)
                break

    print(f"\nRerun result: {rerun_passed}/{rerun_total} tests passed.")


class TestSquareMatrixProduct(unittest.TestCase):
    @classmethod
    def generate_test_case(cls, test_name, params, A, B, expected, eps=EPSILON):
        """Dynamically generates a test case method for unittest.

        Parameters
        ----------
        test_name : str
        params : dict
        A, B, expected : list of list
        eps : float
        """

        def test(self):
            global total_tests, passed_tests
            total_tests += 1
            result = fhe_square_matrix_product(params, A, B)
            flag, error_size = check_equality_matrix(result, expected, eps)
            OPENFHE_test_result(test_name, A, B, expected, result, error_size, passed=flag)
            if not flag:
                OPENFHE_failure_to_file(
                    test_name, A, B, expected, result, ErrorCodes.ERROR_MATCHING
                )
                OPENFHE_ckks_parameters(param, f"debug_logs/{test_name}_params.json")
            else:
                passed_tests += 1
            self.assertTrue(flag, f"Test {test_name} failed with error size: {error_size}")

        return test


if __name__ == "__main__":
    ckks_param_list = load_ckks_params()
    matrix_sizes = [2, 3, 8, 16]
    test_counter = 1

    for param in ckks_param_list:
        for size in matrix_sizes:
            A = generate_random_matrix(size)
            B = generate_random_matrix(size)
            expected = np.array(A) @ np.array(B)
            test_name = f"test_case_{test_counter}_ring_{param['ringDim']}_size_{size}"
            test_method = TestSquareMatrixProduct.generate_test_case(
                test_name, param, A, B, expected
            )
            setattr(TestSquareMatrixProduct, test_name, test_method)
            test_counter += 1

    unittest.main(argv=[""], exit=False)
    print(f"\nTotal tests passed: {passed_tests}/{total_tests}")
    # print("\nRe-running failed tests from debug logs (if any)...")
    # rerun_failed_tests()
