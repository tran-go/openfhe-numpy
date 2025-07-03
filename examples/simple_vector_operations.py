import numpy as np
from openfhe import *
import openfhe_numpy as onp


def validate_and_print_results(computed, expected, operation_name):
    """Helper function to validate and print vector results."""
    print("\n" + "*" * 60)
    print(f"* {operation_name} *")
    print("*" * 60)
    print(f"\nExpected:\n{expected}")
    print(f"\nDecrypted Result:\n{computed}")
    is_match, error = onp.check_equality_vector(computed, expected)
    print(f"\nMatch: {is_match}, Total Error: {error:.6f}")
    return is_match, error


def main():
    """
    Run a demonstration of homomorphic vector operations using OpenFHE-NumPy:
      • addition
      • subtraction
      • transpose
      • elementwise multiplication
      • inner product via *
    """
    # Cryptographic setup
    params = CCParamsCKKSRNS()

    cc = GenCryptoContext(params)
    cc.Enable(PKESchemeFeature.PKE)
    cc.Enable(PKESchemeFeature.LEVELEDSHE)
    cc.Enable(PKESchemeFeature.ADVANCEDSHE)

    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)
    cc.EvalSumKeyGen(keys.secretKey)

    ring_dim = cc.GetRingDimension()
    total_slots = ring_dim // 2
    print(f"\nCKKS ring dimension: {ring_dim}")
    print(f"Available slots:    {total_slots}")

    # Sample input vectors
    vector_a = [1.0, 2.0, 3.0, 4.0, 5.0]
    vector_b = [4.0, 0.0, 1.0, 3.0, 6.0]

    print("\nInput vectors")
    print("vector_a:", vector_a)
    print("vector_b:", vector_b)

    # Encrypt vector_a directly to ciphertext
    ctv_a = onp.array(cc, vector_a, total_slots, public_key=keys.publicKey)
    ctv_b = onp.array(cc, vector_a, total_slots, public_key=keys.publicKey)

    # 1) Addition
    ctv_add = ctv_a + ctv_b
    res_add = ctv_add.decrypt(keys.secretKey, unpack_type="original")
    validate_and_print_results(res_add, np.add(vector_a, vector_b), "Vector Addition")

    # 2) Subtraction
    ctv_sub = ctv_a - ctv_b
    res_sub = ctv_sub.decrypt(keys.secretKey, unpack_type="original")
    validate_and_print_results(res_sub, np.subtract(vector_a, vector_b), "Vector Subtraction")

    # 3) Transpose (no-op for 1D, but shows API usage)
    ctv_a_T = onp.transpose(ctv_a)
    res_T = ctv_a_T.decrypt(keys.secretKey, unpack_type="original")
    validate_and_print_results(res_T, np.transpose(vector_a), "Tranpose(vector_a)")

    # 4) Elementwise multiplication
    ctv_mul = ctv_a * ctv_b
    res_mul = ctv_mul.decrypt(keys.secretKey, unpack_type="original")
    validate_and_print_results(res_mul, np.multiply(vector_a, vector_b), "Encrypt(a) * Encrypt(b)")

    # 4) Inner product
    ctv_inner = ctv_a @ ctv_b
    inner_decrypted = ctv_inner.decrypt(keys.secretKey, unpack_type="original")
    validate_and_print_results(inner_decrypted, np.dot(vector_a, vector_b), "ctv_a@ctv_b")

    # 5) Sum
    ctv_sum = onp.sum(ctv_a)
    sum_decrypted = ctv_sum.decrypt(keys.secretKey, unpack_type="original")
    validate_and_print_results(sum_decrypted, np.sum(vector_a), "onp.sum(ctv_a)")


if __name__ == "__main__":
    main()
