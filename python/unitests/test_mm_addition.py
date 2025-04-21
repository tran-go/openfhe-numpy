import csv
import os
import random
import unittest
import numpy as np
from datetime import datetime


from openfhe import *
from openfhe_matrix import *
import openfhe_numpy as fp
from openfhe_numpy.utils import *

PARAMS_CSV = "ckks_params.csv"


def load_ckks_params():
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


def gen_crypto_context_from_params(p):
    parameters = CCParamsCKKSRNS()
    parameters.SetRingDim(p["ringDim"])
    parameters.SetMultiplicativeDepth(p["multiplicativeDepth"])
    parameters.SetScalingModSize(p["scalingModSize"])
    parameters.SetBatchSize(p["batchSize"])
    parameters.SetFirstModSize(p["firstModSize"])
    # parameters.SetStandardDeviation(p["standardDeviation"])
    # parameters.SetSecretKeyDist(eval(p["secretKeyDist"]))
    parameters.SetScalingTechnique(eval(p["scalTech"]))
    parameters.SetKeySwitchTechnique(eval(p["ksTech"]))
    parameters.SetSecurityLevel(eval(p["securityLevel"]))
    # parameters.SetNumLargeDigits(p["numLargeDigits"])
    # parameters.SetMaxRelinSkDeg(p["maxRelinSkDeg"])
    # parameters.SetDigitSize(p["digitSize"])

    cc = GenCryptoContext(parameters)
    cc.Enable(PKESchemeFeature.PKE)
    cc.Enable(PKESchemeFeature.LEVELEDSHE)
    cc.Enable(PKESchemeFeature.ADVANCEDSHE)

    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)
    cc.EvalSumKeyGen(keys.secretKey)

    return cc, keys


def fhe_matrix_addition(params, a, b, precision=2):
    total_slots = params["ringDim"] // 2
    cc, keys = gen_crypto_context_from_params(params)

    a = np.array(a)
    b = np.array(b)

    ctm_a = fp.array(cc, a, total_slots, pub_key=keys.publicKey)
    ctm_b = fp.array(cc, b, total_slots, pub_key=keys.publicKey)

    ct_sum = fp.add(cc, ctm_a, ctm_b)
    result = ct_sum.decrypt(cc, keys.secretKey)

    return result


def generate_random_matrix(n):
    return [[random.randint(0, 10) for _ in range(n)] for _ in range(n)]


def OPENFHE_failure_to_file(test_name, A, B, expected, result, error):
    os.makedirs("debug_logs", exist_ok=True)
    OPENFHE_path = f"debug_logs/{test_name}.log"
    with open(OPENFHE_path, "w") as f:
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


def OPENFHE_test_result(test_name, A, B, expected, result, error_size, passed):
    os.makedirs("logs", exist_ok=True)
    OPENFHE_file = "logs/TestMatrixAddition.log"
    with open(OPENFHE_file, "a") as f:
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
        f.write(np.array2string(np.array(result), separator=", ") + "\n\n")


class TestMatrixAddition(unittest.TestCase):
    @classmethod
    def generate_test_case(cls, test_name, params, A, B, expected, eps=EPSILON):
        def test(self):
            result = fhe_matrix_addition(params, A, B, precision=1)
            flag, error_size = check_equality_matrix(result, expected, eps)
            if flag:
                OPENFHE_test_result(test_name, A, B, expected, result, error_size, passed=True)
            else:
                OPENFHE_test_result(test_name, A, B, expected, result, error_size, passed=False)
                OPENFHE_failure_to_file(
                    test_name, A, B, expected, result, FHEErrorCodes.ERROR_MATCHING
                )
                raise

        return test


if __name__ == "__main__":
    ckks_param_list = load_ckks_params()
    matrix_sizes = [2, 3, 8, 16]
    test_counter = 1

    for param in ckks_param_list:
        for size in matrix_sizes:
            A = generate_random_matrix(size)
            B = generate_random_matrix(size)
            expected = (np.array(A) + np.array(B)).tolist()
            test_name = f"test_case_{test_counter}_ring_{param['ringDim']}_size_{size}"
            test_method = TestMatrixAddition.generate_test_case(test_name, param, A, B, expected)
            setattr(TestMatrixAddition, test_name, test_method)
            test_counter += 1

    unittest.main(argv=[""], exit=False)
