import numpy as np

from openfhe import (
    CCParamsCKKSRNS,
    GenCryptoContext,
    PKESchemeFeature,
    HEStd_NotSet,
    FIXEDAUTO,
    HYBRID,
    UNIFORM_TERNARY,
)

import openfhe_numpy as onp
from openfhe_numpy.utils import check_equality_matrix


def gen_crypto_context(mult_depth: int, ring_dim: int = 0):
    """
    Generate a CryptoContext and key pair for CKKS encryption.

    Parameters
    ----------
    mult_depth : int
        Maximum multiplicative depth for the ciphertext.
    ring_dim : int, optional
        Ring dimension (must be power of two). If 0, it's chosen by default.

    Returns
    -------
    Tuple[CryptoContext, KeyPair]
        The generated crypto context and its associated key pair.
    """
    params = CCParamsCKKSRNS()

    if ring_dim > 0:
        params.SetRingDim(ring_dim)
        params.SetBatchSize(ring_dim // 2)
        params.SetSecurityLevel(HEStd_NotSet)

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
    Run a demonstration of homomorphic matrix-matrix multiplication using OpenFHE-NumPy.
    """
    ring_dim = 2**12
    mult_depth = 4
    total_slots = ring_dim // 2

    cc, keys = gen_crypto_context(mult_depth, ring_dim)

    # Sample 8×8 input matrices
    A = np.array(
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
    B = np.array(
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

    print("Matrix A:\n", A)
    print("Matrix B:\n", B)

    # Encrypt both matrices
    ctm_A = onp.array(cc, A, total_slots, public_key=keys.publicKey)
    ctm_B = onp.array(cc, B, total_slots, public_key=keys.publicKey)

    print("\n********** HOMOMORPHIC MULTIPLICATION **********")
    print("1. Matrix Multiplication...")

    # Generate rotation/sum keys for square matmul
    onp.gen_square_matrix_product(keys.secretKey, ctm_A.ncols)
    ctm_result = onp.matmul(ctm_A, ctm_B)

    # Decrypt the result
    result = ctm_result.decrypt(keys.secretKey)

    # Compare with plaintext
    expected = A @ B
    print(f"\nExpected:\n{expected}")
    print(f"\nDecrypted Result:\n{result}")

    is_match, error = check_equality_matrix(result, expected)
    print(f"\nMatch: {is_match}, Total Error: {error:.6f}")


if __name__ == "__main__":
    demo()
