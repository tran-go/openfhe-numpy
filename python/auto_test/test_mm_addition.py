import csv
import os
import random
import unittest
import ast
import numpy as np
from datetime import datetime

from openfhe import *
from openfhe_matrix import *
import openfhe_numpy as fp
from openfhe_numpy.utils import *


def gen_crypto_context(ringDimension, mult_depth):
    parameters = CCParamsCKKSRNS()
    parameters.SetSecurityLevel(HEStd_NotSet)
    parameters.SetRingDim(ringDimension)
    parameters.SetMultiplicativeDepth(mult_depth)
    parameters.SetScalingModSize(59)
    parameters.SetBatchSize(ringDimension // 2)
    parameters.SetScalingTechnique(FIXEDAUTO)
    parameters.SetKeySwitchTechnique(HYBRID)
    parameters.SetFirstModSize(60)
    parameters.SetSecretKeyDist(UNIFORM_TERNARY)

    cc = GenCryptoContext(parameters)
    cc.Enable(PKESchemeFeature.PKE)
    cc.Enable(PKESchemeFeature.LEVELEDSHE)
    cc.Enable(PKESchemeFeature.ADVANCEDSHE)

    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)
    cc.EvalSumKeyGen(keys.secretKey)

    return cc, keys


def fhe_matrix_addition(ringDimension, mult_depth, a, b, precision=2):
    total_slots = ringDimension // 2

    cc, keys = gen_crypto_context(ringDimension, mult_depth)
    a = np.array(a)
    b = np.array(b)

    ctm_a = fp.array(cc, a, total_slots, pub_key=keys.publicKey)
    ctm_b = fp.array(cc, b, total_slots, pub_key=keys.publicKey)

    ct_sum = fp.add(cc, ctm_a, ctm_b)
    result = ct_sum.decrypt(cc, keys.secretKey)

    return np.round(result, decimals=precision)


def generate_random_matrix(n):
    return [[random.randint(0, 10) for _ in range(n)] for _ in range(n)]


def log_failure_to_file(test_name, A, B, expected, result, error):
    os.makedirs("debug_logs", exist_ok=True)
    log_path = f"debug_logs/{test_name}.log"
    with open(log_path, "w") as f:
        f.write(f"Test Name: {test_name}\n\n")
        f.write("Matrix A:\n")
        f.write(np.array2string(np.array(A), separator=", ") + "\n\n")
        f.write("Matrix B:\n")
        f.write(np.array2string(np.array(B), separator=", ") + "\n\n")
        f.write("Expected Result:\n")
        f.write(np.array2string(np.array(expected), separator=", ") + "\n\n")
        f.write("Actual Result:\n")
        f.write(np.array2string(np.array(result), separator=", ") + "\n\n")
        f.write("Error:\n")
        f.write(str(error) + "\n")


def log_test_result(test_name, A, B, expected, result, passed):
    os.makedirs("logs", exist_ok=True)
    log_file = "logs/TestMatrixAddition.log"
    with open(log_file, "a") as f:
        status = "PASS" if passed else "FAIL"
        f.write(f"--- {datetime.now().isoformat()} ---\n")
        f.write(f"Test: {test_name}\n")
        f.write(f"Status: {status}\n")
        f.write("Matrix A:\n")
        f.write(np.array2string(np.array(A), separator=", ") + "\n")
        f.write("Matrix B:\n")
        f.write(np.array2string(np.array(B), separator=", ") + "\n")
        f.write("Expected Result:\n")
        f.write(np.array2string(np.array(expected), separator=", ") + "\n")
        f.write("Actual Result:\n")
        f.write(np.array2string(np.array(result), separator=", ") + "\n")
        f.write("\n")


class TestMatrixAddition(unittest.TestCase):
    @classmethod
    def generate_test_case(cls, test_name, ring_dim, A, B, expected):
        def test(self):
            result = fhe_matrix_addition(ring_dim, 3, A, B, precision=1)
            try:
                np.testing.assert_array_almost_equal(result, expected, decimal=1)
                log_test_result(test_name, A, B, expected, result, passed=True)
            except AssertionError as e:
                log_test_result(test_name, A, B, expected, result, passed=False)
                log_failure_to_file(test_name, A, B, expected, result, e)
                raise

        return test


if __name__ == "__main__":
    ring_dims = [2**15]
    matrix_sizes = [30, 32, 64]  # Add more sizes if needed 128,  2**16
    test_counter = 1

    for ring_dim in ring_dims:
        for size in matrix_sizes:
            for _ in range(2):  # Two random tests per configuration
                A = generate_random_matrix(size)
                B = generate_random_matrix(size)
                expected = (np.array(A) + np.array(B)).tolist()
                test_name = f"test_case_{test_counter}_ring_{ring_dim}_size_{size}"
                test_method = TestMatrixAddition.generate_test_case(
                    test_name, ring_dim, A, B, expected
                )
                setattr(TestMatrixAddition, test_name, test_method)
                test_counter += 1

    unittest.main(argv=[""], exit=False)
