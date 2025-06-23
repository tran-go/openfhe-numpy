import time

import numpy as np
from openfhe import *
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
    # matrix = np.array(
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

    matrix = np.array([[1.80521609, 0.46970757], [7.82405472, 8.52768494]])

    print("Matrix:\n", matrix)
    if params.GetBatchSize():
        batch_size = params.GetBatchSize()
    else:
        batch_size = cc.GetRingDimension() // 2

    # Encrypt matrix A
    tensor = onp.array(cc, matrix, batch_size, onp.ROW_MAJOR, "C", mode="zero", public_key=keys.publicKey)

    print(f"batch_size = {batch_size}, dim = {cc.GetRingDimension()}, shape = {tensor.shape}")

    print("\n********** HOMOMORPHIC SUM BY ALL ENTRIES **********")
    #  Generate rotation keys for column operations
    start_keygen = time.time()
    onp.gen_accumulate_rows_key(keys.secretKey, tensor.ncols)
    end_keygen = time.time()

    # Perform homomorphic column accumulation
    start_acc = time.time()
    result_tensor = onp.sum(tensor)
    end_acc = time.time()

    # Perform decryption
    start_dec = time.time()
    result = result_tensor.decrypt(keys.secretKey, unpack_type="original")
    end_dec = time.time()

    expected = np.sum(matrix)

    is_match, error = onp.check_single_equality(result, expected)

    # Timing
    print(f"Sum Time (KeyGen): {(end_keygen - start_keygen) * 1000:.2f} ms")
    print(f"Sum Time (Eval): {(end_acc - start_acc) * 1000:.2f} ms")
    print(f"Time for decryption: {(end_dec - start_dec) * 1000:.2f} ms")

    # Print out result
    print(f"\nExpected:\n{expected}")
    print(f"\nDecrypted Result:\n{result}")
    print(f"\nMatch: {is_match}, Total Error: {error:.6f}")

    print("\n********** HOMOMORPHIC SUM BY ROWS **********")
    #  Generate rotation keys for column operations
    start_keygen = time.time()
    tensor.extra["rowkey"] = onp.sum_row_keys(keys.secretKey, tensor.ncols)
    end_keygen = time.time()

    # Perform homomorphic column accumulation
    start_acc = time.time()
    result_tensor = onp.sum(tensor, axis=0)
    end_acc = time.time()

    # Perform decryption
    start_dec = time.time()
    result = result_tensor.decrypt(keys.secretKey, unpack_type="original")
    end_dec = time.time()
    # result = np.round(result, decimals=1)
    print(f"Row Accumulation Time (KeyGen): {(end_keygen - start_keygen) * 1000:.2f} ms")
    print(f"Row Accumulation Time (Eval): {(end_acc - start_acc) * 1000:.2f} ms")
    print(f"Time for decryption: {(end_dec - start_dec) * 1000:.2f} ms")

    # Compare with plain result
    expected = np.sum(matrix, axis=0)
    print(f"\nExpected:\n{expected}")
    print(f"\nDecrypted Result:\n{result}")

    is_match, error = onp.check_equality_vector(result, expected)
    print(f"\nMatch: {is_match}, Total Error: {error:.6f}")

    print("\n********** HOMOMORPHIC SUM BY COLUMNS **********")

    #  Generate rotation keys for column operations
    start_keygen = time.time()
    tensor.extra["colkey"] = onp.sum_col_keys(keys.secretKey, tensor.ncols)
    end_keygen = time.time()

    # Perform homomorphic column accumulation
    start_acc = time.time()
    result_tensor = onp.sum(tensor, axis=1)
    end_acc = time.time()

    # Perform decryption
    start_dec = time.time()
    result = result_tensor.decrypt(keys.secretKey, unpack_type="original")
    end_dec = time.time()
    # result = np.round(result, decimals=1)
    print(f"Col Accumulation Time (KeyGen): {(end_keygen - start_keygen) * 1000:.2f} ms")
    print(f"Col Accumulation Time (Eval): {(end_acc - start_acc) * 1000:.2f} ms")
    print(f"Time for decryption: {(end_dec - start_dec) * 1000:.2f} ms")

    # Compare with plain result
    expected = np.sum(matrix, axis=1)
    print(f"\nExpected:\n{expected}")
    print(f"\nDecrypted Result:\n{result}")

    is_match, error = onp.check_equality_vector(result, expected)
    print(f"\nMatch: {is_match}, Total Error: {error:.6f}")


if __name__ == "__main__":
    demo()
