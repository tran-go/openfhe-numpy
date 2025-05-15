# Organize imports logically
import time
import numpy as np
from openfhe import (
    CCParamsCKKSRNS,
    GenCryptoContext,
    PKESchemeFeature,
    FIXEDAUTO,
    HYBRID,
    UNIFORM_TERNARY,
)
import openfhe_numpy as onp
from openfhe_numpy.utils import check_equality_matrix


def gen_crypto_context(mult_depth):
    """
    Generate a CryptoContext and key pair for CKKS encryption.

    Parameters
    ----------
    mult_depth : int
        Maximum multiplicative depth for the ciphertext.

    Returns
    -------
    tuple
        (CryptoContext, KeyPair)
    """

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

    return cc, keys


def demo():
    """
    Run a demonstration of homomorphic matrix addition using OpenFHE-NumPy.
    """

    cc, keys = gen_crypto_context(4)
    ring_dim = cc.GetRingDimension()
    total_slots = ring_dim // 2

    # Sample input matrices (8x8)
    matA = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]])

    matB = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]])

    # matA = np.array(
    #     [
    #         [0, 7, 8, 10, 1, 2, 7, 6],
    #         [0, 1, 1, 9, 7, 5, 1, 7],
    #         [8, 8, 4, 5, 8, 2, 6, 1],
    #         [1, 0, 0, 1, 10, 3, 1, 7],
    #         [7, 8, 2, 5, 3, 2, 10, 9],
    #         [0, 3, 4, 10, 10, 5, 2, 5],
    #         [2, 5, 0, 2, 8, 8, 5, 9],
    #         [5, 1, 10, 6, 2, 8, 6, 3],
    #     ]
    # )

    # matB = np.array(
    #     [
    #         [7, 0, 1, 3, 5, 0, 1, 8],
    #         [0, 5, 10, 3, 9, 0, 2, 10],
    #         [10, 8, 9, 8, 4, 9, 8, 8],
    #         [2, 9, 7, 9, 3, 8, 2, 8],
    #         [2, 8, 2, 2, 10, 7, 6, 0],
    #         [8, 7, 3, 0, 3, 10, 6, 5],
    #         [6, 6, 5, 9, 10, 5, 4, 7],
    #         [1, 4, 3, 4, 3, 9, 9, 4],
    #     ]
    # )

    print("Matrix A:\n", matA)
    print("Matrix B:\n", matB)

    # Encrypt both matrices
    ctm_matA = onp.array(cc, matA, total_slots, public_key=keys.publicKey)
    ctm_matB = onp.array(cc, matB, total_slots, public_key=keys.publicKey)

    print("\n********** HOMOMORPHIC ADDITIONS **********")
    print("1. Matrix Additions...")

    # Perform matrix additions on ciphertexts
    ctm_result = onp.add(ctm_matA, ctm_matB)
    # ctm_result = ctm_matA + ctm_matB
    result = ctm_result.decrypt(keys.secretKey)
    result = np.round(result, decimals=1)

    # Compare with plain result
    expected = matA + matB
    # expected = np.add(matA, matB)

    print(f"\nExpected:\n{expected}")
    print(f"\nDecrypted Result:\n{result}")

    is_match, error = check_equality_matrix(result, expected)
    print(f"\nMatch: {is_match}, Total Error: {error:.6f}")


if __name__ == "__main__":
    demo()
