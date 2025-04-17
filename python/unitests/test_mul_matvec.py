import os
import random
import unittest
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


def fhe_matrix_vector_product(ringDimension, mult_depth, matrix, vector, precision=2):
    total_slots = ringDimension // 2

    cc, keys = gen_crypto_context(ringDimension, mult_depth)
    matrix = np.array(matrix)
    vector = np.array(vector).reshape(-1, 1)

    block_size = len(matrix[0])
    sum_col_keys = fp.gen_sum_col_keys(cc, keys.secretKey, block_size)

    ctm_matrix = fp.array(cc, matrix, total_slots, pub_key=keys.publicKey)
    ctm_vector = fp.array(cc, vector, total_slots, block_size, "C", pub_key=keys.publicKey)
    ct_result = fp.matvec(cc, keys, sum_col_keys, ctm_a, ctv_c, block_size)
    result = ct_result.decrypt(cc, keys.secretKey)

    return np.round(result, decimals=precision)


def generate_random_matrix_vector(n):
    matrix = [[random.randint(0, 10) for _ in range(n)] for _ in range(n)]
    vector = [random.randint(0, 10) for _ in range(n)]
    return matrix, vector


def log_failure_to_file(test_name, A, v, expected, result, error):
    os.makedirs("debug_logs", exist_ok=True)
    log_path = f"debug_logs/{test_name}.log"
    with open(log_path, "w") as f:
        f.write(f"Test Name: {test_name}\n\n")
        f.write("Matrix A:\n")
        f.write(np.array2string(np.array(A), separator=", ") + "\n\n")
        f.write("Vector v:\n")
        f.write(np.array2string(np.array(v), separator=", ") + "\n\n")
        f.write("Expected Result:\n")
        f.write(np.array2string(np.array(expected), separator=", ") + "\n\n")
        f.write("Actual Result:\n")
        f.write(np.array2string(np.array(result), separator=", ") + "\n\n")
        f.write("Error:\n")
        f.write(str(error) + "\n")


def log_test_result(test_name, A, v, expected, result, passed):
    os.makedirs("logs", exist_ok=True)
    log_file = "logs/TestMatrixVectorProduct.log"
    with open(log_file, "a") as f:
        status = "PASS" if passed else "FAIL"
        f.write(f"--- {datetime.now().isoformat()} ---\n")
        f.write(f"Test: {test_name}\n")
        f.write(f"Status: {status}\n")
        f.write("Matrix A:\n")
        f.write(np.array2string(np.array(A), separator=", ") + "\n")
        f.write("Vector v:\n")
        f.write(np.array2string(np.array(v), separator=", ") + "\n")
        f.write("Expected Result:\n")
        f.write(np.array2string(np.array(expected), separator=", ") + "\n")
        f.write("Actual Result:\n")
        f.write(np.array2string(np.array(result), separator=", ") + "\n")
        f.write("\n")


class TestMatrixVectorProduct(unittest.TestCase):
    @classmethod
    def generate_test_case(cls, test_name, ring_dim, matrix, vector, expected):
        def test(self):
            result = fhe_matrix_vector_product(ring_dim, 3, matrix, vector, precision=1)
            try:
                np.testing.assert_array_almost_equal(result, expected, decimal=1)
                log_test_result(test_name, matrix, vector, expected, result, passed=True)
            except AssertionError as e:
                log_test_result(test_name, matrix, vector, expected, result, passed=False)
                log_failure_to_file(test_name, matrix, vector, expected, result, e)
                raise

        return test


if __name__ == "__main__":
    ring_dims = [2**12, 2**13, 2**14]
    matrix_sizes = [2, 4, 8]
    test_counter = 1

    for ring_dim in ring_dims:
        for size in matrix_sizes:
            for _ in range(2):  # Two tests per setting
                A, v = generate_random_matrix_vector(size)
                expected = (np.matmul(np.array(A), np.array(v).reshape(-1, 1))).tolist()
                test_name = f"test_case_{test_counter}_ring_{ring_dim}_size_{size}"
                test_method = TestMatrixVectorProduct.generate_test_case(
                    test_name, ring_dim, A, v, expected
                )
                setattr(TestMatrixVectorProduct, test_name, test_method)
                test_counter += 1

    unittest.main(argv=[""], exit=False)
