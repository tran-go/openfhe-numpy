# import openfhe related libraries
import numpy as np
from openfhe import *
from openfhe_matrix import *

# import openfhe_numpy library
import openfhe_numpy as fp
from openfhe_numpy.utils import *


#  TODO: add separate examples
#  TODO: transpose
#  TODO: tests
#  TODO: clean code + comments


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


def demo():
    # TODO check with different ringDimension, write test cases
    ringDimension = 2**12
    total_slots = ringDimension // 2
    mult_depth = 9

    cc, keys = gen_crypto_context(ringDimension, mult_depth)

    # a = np.array([[9, 4, 1], [9, 7, 3], [7, 6, 10]])
    # b = np.array([[0, 2, 10], [5, 6, 10], [3, 9, 9]])
    # a = np.array([[5, 8, 7], [2, 0, 0], [9, 10, 7]])

    # b = np.array([[2, 0, 6], [2, 5, 7], [5, 3, 2]])

    a = np.array(
        [
            [0, 7, 8, 10, 1, 2, 7, 6],
            [0, 1, 1, 9, 7, 5, 1, 7],
            [8, 8, 4, 5, 8, 2, 6, 1],
            [1, 0, 0, 1, 10, 3, 1, 7],
            [7, 8, 2, 5, 3, 2, 10, 9],
            [0, 3, 4, 10, 10, 5, 2, 5],
            [2, 5, 0, 2, 8, 8, 5, 9],
            [5, 1, 10, 6, 2, 8, 6, 3],
        ]
    )

    b = np.array(
        [
            [7, 0, 1, 3, 5, 0, 1, 8],
            [0, 5, 10, 3, 9, 0, 2, 10],
            [10, 8, 9, 8, 4, 9, 8, 8],
            [2, 9, 7, 9, 3, 8, 2, 8],
            [2, 8, 2, 2, 10, 7, 6, 0],
            [8, 7, 3, 0, 3, 10, 6, 5],
            [6, 6, 5, 9, 10, 5, 4, 7],
            [1, 4, 3, 4, 3, 9, 9, 4],
        ]
    )

    print("a: \n", a)
    print("b: \n", b)

    ctm_a = fp.array(cc, a, total_slots, pub_key=keys.publicKey)
    ctm_b = fp.array(cc, b, total_slots, pub_key=keys.publicKey)

    print()
    print("*" * 10, "MULTIPLICATION", "*" * 10)

    print("\n1.Matrix multiplication:")
    ct_prod = fp.matmul_square(cc, keys, ctm_a, ctm_b)
    result = ct_prod.decrypt(cc, keys.secretKey)
    print(f"Expected: {a @ b}")
    print(f"Obtained: {result}")
    is_corrected, error_size = check_equality_matrix(result, a @ b)
    print(f"Matching = [{is_corrected}] with precision: [{error_size}]")


demo()
