import time

# Import OpenFHE and matrix utilities
import numpy as np
from openfhe import *
import openfhe_numpy as onp


def demo():
    """
    Run a demonstration of homomorphic matrix multiplication using OpenFHE-NumPy.
    """

    mult_depth = 4
    params = CCParamsCKKSRNS()
    params.SetMultiplicativeDepth(mult_depth)
    params.SetScalingModSize(59)
    params.SetFirstModSize(60)
    params.SetScalingTechnique(FIXEDAUTO)
    params.SetKeySwitchTechnique(HYBRID)
    params.SetSecretKeyDist(UNIFORM_TERNARY)

    cc = GenCryptoContext(params)
    cc.Enable(PKESchemeFeature.PKE)
    cc.Enable(PKESchemeFeature.LEVELEDSHE)
    cc.Enable(PKESchemeFeature.ADVANCEDSHE)

    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)
    cc.EvalSumKeyGen(keys.secretKey)

    # Sample input matrix (8x8)
    matrix = np.array(
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

    print("Matrix:\n", matrix)
    slots = params.GetBatchSize() if params.GetBatchSize() else cc.GetRingDimension() // 4
    print(params.GetBatchSize(), params.GetRingDim())

    # Encrypt matrix A
    ctm_matA = onp.array(cc, matrix, slots, public_key=keys.publicKey)

    print("\n********** HOMOMORPHIC MATRIX TRANSPOSE **********")

    # Perform matrix tranpose on ciphertexts
    onp.gen_transpose_keys(keys.secretKey, ctm_matA)
    ctm_result = onp.transpose(ctm_matA)  # ctm_matA.T

    # Decrypt the result
    result = ctm_result.decrypt(keys.secretKey)
    # Compare with plain result
    expected = matrix.T
    print(f"\nExpected:\n{expected}")
    print(f"\nDecrypted Result:\n{result}")

    is_match, error = onp.check_equality_matrix(result, expected)
    print(f"\nMatch: {is_match}, Total Error: {error:.6f}")


if __name__ == "__main__":
    demo()
