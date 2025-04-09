import os, random, unittest
import numpy as np
from datetime import datetime
from openfhe import *
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


def fhe_vector_add(ringDimension, mult_depth, a, b, precision=2):
    cc, keys = gen_crypto_context(ringDimension, mult_depth)
    total_slots = ringDimension // 2

    a = np.array(a)
    b = np.array(b)

    block_size = len(a)

    cta = fp.array(cc, a, total_slots, block_size, "C", pub_key=keys.publicKey)
    ctb = fp.array(cc, b, total_slots, block_size, "C", pub_key=keys.publicKey)

    result = fp.add(cc, cta, ctb).decrypt(cc, keys.secretKey)
    return np.round(result, decimals=precision)


def log_failure(test_name, a, b, expected, result, error):
    os.makedirs("debug_logs", exist_ok=True)
    with open(f"debug_logs/{test_name}.log", "w") as f:
        f.write(f"Test: {test_name}\nVector A: {a}\nVector B: {b}\nExpected: {expected}\nResult: {result}\nError: {error}\n")


def log_result(test_name, a, b, expected, result, passed):
    os.makedirs("logs", exist_ok=True)
    with open("logs/TestVectorAddition.log", "a") as f:
        f.write(f"--- {datetime.now().isoformat()} ---\n")
        f.write(f"{test_name} | {'PASS' if passed else 'FAIL'}\nA: {a}\nB: {b}\nExpected: {expected}\nResult: {result}\n\n")


class TestVectorAddition(unittest.TestCase):
    @classmethod
    def generate_test(cls, test_name, ring_dim, a, b, expected):
        def test(self):
            result = fhe_vector_add(ring_dim, 2, a, b)
            try:
                np.testing.assert_array_almost_equal(result, expected, decimal=1)
                log_result(test_name, a, b, expected, result, True)
            except AssertionError as e:
                log_result(test_name, a, b, expected, result, False)
                log_failure(test_name, a, b, expected, result, e)
                raise
        return test


if __name__ == "__main__":
    ring_dims = [2**10, 2**12, 2**14]
    sizes = [4, 8, 16]
    test_counter = 1

    for r in ring_dims:
        for size in sizes:
            for _ in range(2):
                a = [random.randint(0, 10) for _ in range(size)]
                b = [random.randint(0, 10) for _ in range(size)]
                expected = (np.array(a) + np.array(b)).tolist()
                name = f"test_case_{test_counter}"
                method = TestVectorAddition.generate_test(name, r, a, b, expected)
                setattr(TestVectorAddition, name, method)
                test_counter += 1

    unittest.main(argv=[""], exit=False)

