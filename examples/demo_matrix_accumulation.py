import time

import numpy as np

from openfhe import *
import openfhe_numpy as onp


# push the params inside the demo
def demo():
    """
    Run a demonstration of homomorphic matrix accumulation using OpenFHE-NumPy.
    """

    mult_depth = 8
    params = CCParamsCKKSRNS()
    params.SetMultiplicativeDepth(mult_depth)
    params.SetScalingModSize(59)  # add comments
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

    # Sample input matrix (3x8)
    matrix = np.array(
        [
            [0, 7, 8, 10, 1, 2, 7, 6],
            [0, 1, 1, 9, 7, 5, 1, 7],
            [8, 8, 4, 5, 8, 2, 6, 1],
            # [1, 0, 0, 1, 10, 3, 1, 7],
            # [7, 8, 2, 5, 3, 2, 10, 9],
            # [0, 3, 4, 10, 10, 5, 2, 5],
            # [2, 5, 0, 2, 8, 8, 5, 9],
            # [5, 1, 10, 6, 2, 8, 6, 3],
        ]
    )

    print("Matrix:\n", matrix)
    # slots = params.GetBatchSize() if params.GetBatchSize() else cc.GetRingDimension() // 2
    slots = cc.GetBatchSize() if params.GetBatchSize() else cc.GetRingDimension() // 2

    # Encrypt matrix A
    ctm_A = onp.array(cc, matrix, slots, order=onp.ROW_MAJOR, public_key=keys.publicKey)

    print(f"slots = {slots}, dim = {cc.GetRingDimension()}, ncols = {ctm_A.ncols}")

    print("\n********** HOMOMORPHIC ACCUMULATION BY ROWS **********")

    #  Generate rotation keys for column operations
    onp.gen_accumulate_rows_key(keys.secretKey, ctm_A.ncols)

    # Perform homomorphic column accumulation
    ctm_result = onp.cumsum(ctm_A, axis=0)
    result = ctm_result.decrypt(keys.secretKey, unpack_type="original")

    # Compare with plain result
    expected = np.cumsum(matrix, axis=0)
    is_match, error = onp.check_equality_matrix(result, expected)

    print(f"\nExpected:\n{expected}")
    print(f"\nDecrypted Result:\n{result}")
    print(f"\nMatch: {is_match}, Total Error: {error:.6f}")

    print("\n********** HOMOMORPHIC ACCUMULATION BY COLUMNS **********")

    #  Generate rotation keys for column operations
    onp.gen_accumulate_cols_key(keys.secretKey, ctm_A.ncols)

    # Perform homomorphic column accumulation
    ctm_result = onp.cumsum(ctm_A, axis=1)
    result = ctm_result.decrypt(keys.secretKey, unpack_type="original")

    # Compare with plain result
    expected = np.cumsum(matrix, axis=1)
    is_match, error = onp.check_equality_matrix(result, expected)

    print(f"\nExpected:\n{expected}")
    print(f"\nDecrypted Result:\n{result}")
    print(f"\nMatch: {is_match}, Total Error: {error:.6f}")


if __name__ == "__main__":
    demo()
