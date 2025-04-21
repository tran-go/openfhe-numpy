# Import OpenFHE and matrix utilities
import numpy as np
from openfhe import *
from openfhe_matrix import *

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
    ctm_A = fp.array(cc, A, total_slots, pub_key=keys.publicKey)
    ctm_B = fp.array(cc, B, total_slots, pub_key=keys.publicKey)

    print("\n********** HOMOMORPHIC MULTIPLICATION **********")
    print("1. Matrix Multiplication...")

    # Perform matrix multiplication on ciphertexts
    ctm_result = fp.matmul_square(cc, keys, ctm_A, ctm_B)

    # Decrypt the result
    result = ctm_result.decrypt(cc, keys.secretKey)

    # Compare with plain result
    expected = A @ B
    print(f"\nExpected:\n{expected}")
    print(f"\nDecrypted Result:\n{result}")

    is_match, error = check_equality_matrix(result, expected)
    print(f"\nMatch: {is_match}, Total Error: {error:.6f}")


def gen_transpose_diag(total_slots: int, row_size: int, i: int) -> list:
    """
    Generate a diagonal vector for simulating matrix transpose via diagonal encoding.

    Parameters
    ----------
    total_slots : int
        Number of available slots in the ciphertext.
    row_size : int
        Dimension of the square matrix.
    i : int
        Diagonal index (can be negative).

    Returns
    -------
    list of int
        A diagonal vector with 1s on the `i`-th diagonal and 0s elsewhere.

    Example
    -------
    >>> gen_transpose_diag(32, 4, 1)
    [0, 1, 0, 0, 0, 0, 1, 0, ...]
    """
    n = row_size * row_size
    diag = [0] * total_slots

    for t in range(total_slots // n):
        offset = t * n
        if i >= 0:
            for j in range(row_size - i):
                idx = offset + (row_size + 1) * j + i
                print(f"idx = {idx}")
                if idx < total_slots:
                    diag[idx] = 1
        else:
            for j in range(abs(i), row_size):
                idx = offset + ((row_size + 1) * j + i)
                print(f"idx = {idx}")
                if idx < total_slots:
                    diag[idx] = 1

    return diag


def transpose(v, total_slots, row_size):
    import numpy as np
    from openfhe_numpy.utils import rotate_vector

    final = [0] * total_slots

    for i in range(-row_size + 1, row_size):
        idx = (row_size - 1) * i
        diag = gen_transpose_diag(total_slots, row_size, i)
        print(idx, diag)
        print(idx, rotate_vector(v, idx))
        tmp = np.array(diag) * np.array(rotate_vector(v, idx))
        print(idx, tmp, "\n")
        final += np.array(diag) * np.array(rotate_vector(v, idx))
    print("====================")
    print(final)


def demo1():
    from openfhe_numpy.constructors import ravel_mat

    A = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]
    )
    A = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]
    )
    A = np.array(
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
        ]
    )
    # A = np.array([[1, 2], [4, 5]])
    row_size = 4
    total_slots = 2**5
    print("Matrix A:\n", A)
    print("row_size: ", row_size)

    # Encrypt both matrices
    packed = ravel_mat(A, total_slots, row_size)
    print(packed)
    transpose(packed, total_slots, row_size)


if __name__ == "__main__":
    demo1()
