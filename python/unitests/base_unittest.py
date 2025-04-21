import csv
import os
import random
import unittest
import numpy as np
from datetime import datetime
import unittest_helper

from openfhe import *
from openfhe_matrix import *
import openfhe_numpy as fp
from openfhe_numpy.utils import *

PARAMS_CSV = "ckks_params.csv"


def generate_random_matrix(n):
    return [[random.randint(0, 10) for _ in range(n)] for _ in range(n)]


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


def OPENFHE_failure_to_file(test_name, input, expected, result, error):
    os.makedirs("debug_logs", exist_ok=True)
    OPENFHE_path = f"debug_logs/{test_name}.log"
    expected
    with open(OPENFHE_path, "w") as f:
        f.write(f"Test Name: {test_name}\n\n")
        f.write("Input:\n")
        for i, x in enumerate(input):
            f.write(f"Input {i}:\n")
            f.write(np.array2string(np.array(x), separator=", ") + "\n\n")
        f.write("Expected Result:\n")
        f.write(np.array2string(np.array(expected), separator=", ") + "\n\n")
        f.write("Actual Result:\n")
        f.write(np.array2string(np.array(result), separator=", ") + "\n\n")
        f.write("Error:\n")
        f.write(str(error) + "\n")


def OPENFHE_test_result(test_name, input, expected, result, error_size, passed):
    os.makedirs("logs", exist_ok=True)
    OPENFHE_file = "logs/TestMatrixAddition.log"
    with open(OPENFHE_file, "a") as f:
        status = "PASS" if passed else "FAIL"
        f.write(f"--- {datetime.now().isoformat()} ---\n")
        f.write(f"Test: {test_name}\n")
        f.write(f"Status: {status}\n")
        f.write("Input:\n")
        for i, x in enumerate(input):
            f.write(f"Input {i}:\n")
            f.write(np.array2string(np.array(x), separator=", ") + "\n\n")
        f.write("Expected Result:\n")
        f.write(np.array2string(np.array(expected), separator=", ") + "\n")
        f.write("Actual Result:\n")
        f.write(np.array2string(np.array(result), separator=", ") + "\n\n")


class Unittests(unittest.TestCase):
    @classmethod
    def generate_test_case(
        cls, func_name, test_name, params, input, expected, eps=unittest_helper.EPSILON
    ):
        def test(self):
            result = func_name(params, input)
            flag, error_size = unittest_helper.check_equality_matrix(result, expected, eps)
            if flag:
                OPENFHE_test_result(test_name, input, expected, result, error_size, passed=True)
            else:
                OPENFHE_test_result(test_name, input, expected, result, error_size, passed=False)
                OPENFHE_failure_to_file(
                    test_name, input, expected, result, unittest_helper.FHEErrorCodes.ERROR_MATCHING
                )
                raise

        return test
