# Import OpenFHE and matrix utilities
import numpy as np
from openfhe import *
from openfhe_matrix import *
from openfhe_numpy import utils

# Import OpenFHE NumPy-style interface
import openfhe_numpy as fp
from openfhe_numpy.utils import check_equality_matrix


def gen_crypto_context(ring_dim, mult_depth):
    """
    Generate a CryptoContext and key pair for CKKS encryption.

    Parameters
    ----------
    ring_dim : int
        Ring dimension (must be power of two).
    mult_depth : int
        Maximum multiplicative depth for the ciphertext.

    Returns
    -------
    tuple
        (CryptoContext, KeyPair)
    """
    params = CCParamsCKKSRNS()
    params.SetSecurityLevel(HEStd_NotSet)
    params.SetRingDim(ring_dim)
    params.SetMultiplicativeDepth(mult_depth)
    params.SetScalingModSize(59)
    params.SetFirstModSize(60)
    params.SetBatchSize(ring_dim // 2)
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
    Run a demonstration of homomorphic matrix multiplication using OpenFHE-NumPy.
    """
    ring_dim = 2**12
    mult_depth = 4
    total_slots = ring_dim // 2

    cc, keys = gen_crypto_context(ring_dim, mult_depth)

    # Sample input matrices (8x8)
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

    b = np.array(
        [7, 0, 1, 3, 5, 0, 1, 8],
    )

    print("Matrix A:\n", A)
    print("Vector b:\n", b)

    # Encrypt both matrices
    ct_matrix = fp.array(cc, A, total_slots, pub_key=keys.publicKey)
    block_size = ct_matrix.ncols
    sum_col_keys = fp.gen_sum_col_keys(cc, keys.secretKey, block_size)
    ct_vector = fp.array(cc, b, total_slots, block_size, "C", pub_key=keys.publicKey)

    print("\n********** HOMOMORPHIC Matrix Vector Product **********")
    ct_result = fp.matvec(cc, keys, sum_col_keys, ct_matrix, ct_vector, block_size)
    result = ct_result.decrypt(cc, keys.secretKey)
    # Compare with plain result
    expected = utils.pack_vec_row_wise((A @ b), block_size, total_slots)
    print(f"\nExpected:\n{expected}")
    print(f"\nDecrypted Result:\n{result}")

    is_match, error = check_equality_matrix(result, expected)
    print(f"\nMatch: {is_match}, Total Error: {error:.6f}")


if __name__ == "__main__":
    demo()
