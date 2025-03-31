import csv
import random
import unittest
import ast
import numpy as np

# import pytest
from openfhe import *
from openfhe_matrix import *

import openfhe_numpy as fp
from openfhe_numpy.utils import *


def gen_crypto_context(ringDimension, mult_depth):
    # Setup CryptoContext for CKKS
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

    # Enable the features that you wish to use
    cc = GenCryptoContext(parameters)
    cc.Enable(PKESchemeFeature.PKE)
    cc.Enable(PKESchemeFeature.LEVELEDSHE)
    cc.Enable(PKESchemeFeature.ADVANCEDSHE)

    # Generate encryption keys
    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)
    cc.EvalSumKeyGen(keys.secretKey)

    return cc, keys


# Function to multiply two matrices A and B in FHE


def fhe_matrix_addition(ringDimension, mult_depth, a, b, precision=2):
    print("Simple Test for Matrix Addition")

    total_slots = ringDimension // 2

    cc, keys = gen_crypto_context(ringDimension, mult_depth)
    a = np.array(a)
    b = np.array(b)

    ctm_a = fp.array(cc, a, total_slots, pub_key=keys.publicKey)
    ctm_b = fp.array(cc, b, total_slots, pub_key=keys.publicKey)

    ct_sum = fp.add(cc, ctm_a, ctm_b)
    result = ct_sum.decrypt(cc, keys.secretKey)
    result = np.round(result, decimals=1)

    return result


# Generate 40 test cases and write them to a CSV file
def generate_test_cases_csv(filename, num_tests=1):
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["A", "B", "expected"])  # Header row

        for _ in range(num_tests):
            n = random.randint(4, 4)  # Randomly choose matrix size (2x2 or 3x3)
            A = generate_random_matrix(n)
            B = generate_random_matrix(n)
            expected = np.array(A) + np.array(B)
            print(expected)
            writer.writerow([str(A), str(B), str(expected)])


# Test case class
class TestMatrixMultiplication(unittest.TestCase):
    # Dynamically load test cases from a CSV file
    @classmethod
    def load_test_cases_from_csv(cls, filename):
        test_cases = []
        with open(filename, mode="r") as file:
            reader = csv.reader(file)
            next(reader)  # Skip header row
            for row in reader:
                # Parse matrices and expected result using ast.literal_eval
                A = ast.literal_eval(row[0])
                B = ast.literal_eval(row[1])

                expected = ast.literal_eval(row[2])

                test_cases.append((A, B, expected))
        return test_cases

    # Function to dynamically generate test methods
    @classmethod
    def generate_test_case(cls, A, B, expected):
        def test(self):
            result = fhe_matrix_addition(2**5, 9, A, B, precision=4)
            self.assertEqual(result, expected)

        return test


# Main function to generate CSV and run tests
if __name__ == "__main__":
    # Generate the test cases and write them to the CSV file
    generate_test_cases_csv("tests/mulmat_tests.csv", 1)

    # Load the test cases from the CSV file
    test_cases = TestMatrixMultiplication.load_test_cases_from_csv("tests/mulmat_tests.csv")

    # Dynamically add test methods to the TestMatrixMultiplication class
    for i, (A, B, expected) in enumerate(test_cases):
        test_name = f"test_case_{i + 1}"
        test_method = TestMatrixMultiplication.generate_test_case(A, B, expected)
        setattr(TestMatrixMultiplication, test_name, test_method)

    # Run the unittest framework
    unittest.main()
