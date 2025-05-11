# Import OpenFHE and matrix utilities
import numpy as np
from openfhe import *
from openfhe_matrix import *

# Import OpenFHE NumPy-style interface
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

    return cc, params, keys


def demo():
    """
    Run a demonstration of homomorphic matrix multiplication using OpenFHE-NumPy.
    """

    mult_depth = 4
    cc, params, keys = gen_crypto_context(mult_depth)

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
    slots = params.GetBatchSize() if params.GetBatchSize() else cc.GetRingDimension() // 2
    print(f"slots = {slots}, dim = {cc.GetRingDimension()}")

    # Encrypt matrix A
    ctm_matA = onp.array(cc, matrix, slots, public_key=keys.publicKey)
    print(ctm_matA)

    print("\n********** HOMOMORPHIC MATRIX Accumulation **********")

    # Perform matrix tranpose on ciphertexts
    print("11111111111111111111111111111111111")
    onp.gen_accumulate_cols_key(cc, keys.secretKey, ctm_matA.rowsize)
    print("222222222222222222222222222222222222222")
    # onp.gen_sum_row_keys(cc, keys.secretKey, ctm_matA.rowsize)
    # print("333333333333333333333333333333333333333333333")
    ctm_result = onp.add_accumulate(ctm_matA, 1)

    # Decrypt the result
    result = ctm_result.decrypt(keys.secretKey)

    # Compare with plain result
    expected = matrix.T
    print(f"\nExpected:\n{expected}")
    print(f"\nDecrypted Result:\n{result}")

    is_match, error = check_equality_matrix(result, expected)
    print(f"\nMatch: {is_match}, Total Error: {error:.6f}")


if __name__ == "__main__":
    demo()
