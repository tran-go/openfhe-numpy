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
        (CryptoContext, CCParamsCKKSRNS, KeyPair)
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
    Run a demonstration of homomorphic matrix accumulation using OpenFHE-NumPy.
    """
    mult_depth = 8
    cc, params, keys = gen_crypto_context(mult_depth)

    # Sample input matrix (8x8)
    matrix = np.array([[1.4080462, 4.79377952], [6.42504974, 9.33786052]])

    print("Matrix:\n", matrix)
    slots = params.GetBatchSize() if params.GetBatchSize() else cc.GetRingDimension() // 2

    # Encrypt matrix A
    ctm_matA = onp.array(cc, matrix, slots, public_key=keys.publicKey)

    print(f"slots = {slots}, dim = {cc.GetRingDimension()}, ncols = {ctm_matA.ncols}")

    print("\n********** HOMOMORPHIC ACCUMULATION BY ROWS **********")
    #  Generate rotation keys for column operations
    start_keygen = time.time()
    onp.gen_accumulate_rows_key(keys.secretKey, ctm_matA.ncols)
    end_keygen = time.time()

    # Perform homomorphic column accumulation
    start_acc = time.time()
    ctm_result = onp.cumsum(ctm_matA, 0, True)
    end_acc = time.time()

    # Perform decryption
    start_dec = time.time()
    result = ctm_result.decrypt(keys.secretKey, True)
    end_dec = time.time()
    result = np.round(result, decimals=1)
    print(f"Row Accumulation Time (KeyGen): {(end_keygen - start_keygen) * 1000:.2f} ms")
    print(f"Row Accumulation Time (Eval): {(end_acc - start_acc) * 1000:.2f} ms")
    print(f"Time for decryption: {(end_dec - start_dec) * 1000:.2f} ms")

    # Compare with plain result
    expected = np.cumsum(matrix, axis=1)
    print(f"\nExpected:\n{expected}")
    print(f"\nDecrypted Result:\n{result}")

    is_match, error = onp.check_equality_matrix(result, expected)
    print(f"\nMatch: {is_match}, Total Error: {error:.6f}")

    print("\n********** HOMOMORPHIC ACCUMULATION BY COLUMNS **********")

    #  Generate rotation keys for column operations
    start_keygen = time.time()
    onp.gen_accumulate_cols_key(keys.secretKey, ctm_matA.ncols)
    end_keygen = time.time()

    # Perform homomorphic column accumulation
    start_acc = time.time()
    ctm_result = onp.cumsum(ctm_matA, 1, True)
    end_acc = time.time()

    # Perform decryption
    start_dec = time.time()
    result = ctm_result.decrypt(keys.secretKey, True)
    end_dec = time.time()
    result = np.round(result, decimals=1)
    print(f"Col Accumulation Time (KeyGen): {(end_keygen - start_keygen) * 1000:.2f} ms")
    print(f"Col Accumulation Time (Eval): {(end_acc - start_acc) * 1000:.2f} ms")
    print(f"Time for decryption: {(end_dec - start_dec) * 1000:.2f} ms")

    # Compare with plain result
    expected = np.cumsum(matrix, axis=1)
    print(f"\nExpected:\n{expected}")
    print(f"\nDecrypted Result:\n{result}")

    is_match, error = onp.check_equality_matrix(result, expected)
    print(f"\nMatch: {is_match}, Total Error: {error:.6f}")


if __name__ == "__main__":
    demo()
