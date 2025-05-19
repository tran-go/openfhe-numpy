import time
import numpy as np
from openfhe import (
    CCParamsCKKSRNS,
    GenCryptoContext,
    PKESchemeFeature,
    HEStd_NotSet,
    FIXEDAUTO,
    FIXEDMANUAL,
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
    ptModulus = 0
    digitSize = 0
    standardDeviation = 3.19
    secretKeyDist = UNIFORM_TERNARY
    maxRelinSkDeg = 2
    ksTech = HYBRID
    scalTech = FIXEDMANUAL
    firstModSize = 60
    batchSize = 512
    numLargeDigits = 3
    multiplicativeDepth = 9
    scalingModSize = 59
    securityLevel = HEStd_NotSet
    ringDim = 1024

    p = CCParamsCKKSRNS()
    p.SetRingDim(ringdim)
    p.SetMultiplicativeDepth(multiplicativeDepth)
    p.SetScalingModSize(scalingModSize)
    p.SetBatchSize(batchSize)
    p.SetFirstModSize(firstModSize)
    p.SetStandardDeviation(standardDeviation)
    p.SetSecretKeyDist(secretKeyDist)
    p.SetScalingTechnique(scalTech)
    p.SetKeySwitchTechnique(ksTech)
    p.SetSecurityLevel(securityLevel)
    p.SetNumLargeDigits(numLargeDigits)
    p.SetMaxRelinSkDeg(maxRelinSkDeg)
    p.SetDigitSize(digitSize)
    cc = GenCryptoContext(p)

    for feature in [
        PKESchemeFeature.PKE,
        PKESchemeFeature.LEVELEDSHE,
        PKESchemeFeature.ADVANCEDSHE,
    ]:
        cc.Enable(feature)

    keys = cc.KeyGen()

    return cc, p, keys


def demo():
    """
    Run a demonstration of homomorphic matrix accumulation using OpenFHE-NumPy.
    """
    mult_depth = 8
    cc, params, keys = gen_crypto_context(mult_depth)

    # Sample input matrix (8x8)
    matrix = np.array([[8.43884633, 9.13095135], [1.82624548, 1.7901978]])
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
    result = ctm_result.decrypt(keys.secretKey, format_type="reshape")
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
    result = ctm_result.decrypt(keys.secretKey, format_type="reshape")
    end_dec = time.time()

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
