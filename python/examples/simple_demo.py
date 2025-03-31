# import openfhe related libraries
import numpy as np
from openfhe import *
from openfhe_matrix import *

# import fhepy library
import fhepy as fp
from fhepy.utils import *


def gen_crypto_context(ringDimension, mult_depth):
    # Setup CryptoContext for CKKS
    parameters = CCParamsCKKSRNS()
    parameters.SetSecurityLevel(HEStd_NotSet)
    parameters.SetRingDim(ringDimension)
    parameters.SetMultiplicativeDepth(mult_depth)
    parameters.SetScalingModSize(59)
    parameters.SetBatchSize(ringDimension // 2)
    parameters.SetScalingTechnique(FIXEDAUTO)
    parameters.SetKeySwitchTechnique(HYBRID)
    parameters.SetFirstModSize(60)
    parameters.SetSecretKeyDist(UNIFORM_TERNARY)

    # Enable the features that you wish to use
    cc = GenCryptoContext(parameters)
    cc.Enable(PKESchemeFeature.PKE)
    cc.Enable(PKESchemeFeature.LEVELEDSHE)
    cc.Enable(PKESchemeFeature.ADVANCEDSHE)

    # Generate encryption keys
    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)
    cc.EvalSumKeyGen(keys.secretKey)

    return cc, keys


def demo():
    # TODO check with different ringDimension, write test cases
    ringDimension = 2**5
    total_slots = ringDimension // 2
    block_size = 4
    mult_depth = 9

    cc, keys = gen_crypto_context(ringDimension, mult_depth)

    a = np.array([[1, 1, 1, 0], [2, 2, 2, 0], [3, 3, 3, 0], [4, 4, 4, 0]])
    b = np.array([[1, 0, 1, 0], [1, 1, 0, 0], [3, 0, 3, 0], [3, 0, 2, 0]])

    c = np.array([1, 2, 3, 4])
    d = np.array([5, 6, 7, 8])

    print("a: \n", a)
    print("b: \n", b)
    print("c: ", c)
    print("d: ", d)

    ctm_a = fp.array(cc, a, total_slots, pub_key=keys.publicKey)
    ctm_b = fp.array(cc, b, total_slots, pub_key=keys.publicKey)

    ctv_c = fp.array(cc, c, total_slots, block_size, "C", pub_key=keys.publicKey)
    ctv_d = fp.array(cc, d, total_slots, block_size, "C", pub_key=keys.publicKey)

    print()
    print("*" * 10, "ADDITION", "*" * 10)
    print("\n1. Matrix addition:")
    ct_sum = fp.add(cc, ctm_a, ctm_b)
    result = ct_sum.decrypt(cc, keys.secretKey)
    result = np.round(result, decimals=1)
    print(f"Expected: {a + b}")
    print(f"Obtained: {result}")
    print(f"Matching = [{np.array_equal(result, a + b)}]")

    print("\n2. Vector addition:")
    ct_sum = fp.add(cc, ctv_c, ctv_d)
    result = ct_sum.decrypt(cc, keys.secretKey)
    result = np.round(result, decimals=1)
    print(f"Expected: {c + d}")
    print(f"Obtained: {result}")
    print(f"Matching = [{np.array_equal(result, c + d)}]")

    print()
    print("*" * 10, "MULTIPLICATION", "*" * 10)

    print("\n1.Matrix multiplication:")
    ct_prod = fp.matmul_square(cc, keys, ctm_a, ctm_b)
    result = ct_prod.decrypt(cc, keys.secretKey)
    result = np.round(result, decimals=1)
    print(f"Expected: {a @ b}")
    print(f"Obtained: {result}")
    print(f"Matching = [{np.array_equal(result, a @ b)}]")

    print("\n2.Matrix Vector multiplication: A@c")
    vec_ac = pack_vec_row_wise((a @ c), block_size, total_slots)
    sum_col_keys = fp.gen_sum_col_keys(cc, keys.secretKey, block_size)
    ct_prod = fp.matvec(cc, keys, sum_col_keys, ctm_a, ctv_c, block_size)
    result = ct_prod.decrypt(cc, keys.secretKey, format=0)
    result = np.round(result, decimals=1)
    print(f"Expected: {vec_ac}")
    print(f"Obtained: {result}")
    print(f"Matching = [{np.array_equal(result, vec_ac)}]")

    print("\n3.Dot product c.d = <c,d>:")
    dot_prod = np.dot(c, d)
    sum_col_keys = fp.gen_sum_col_keys(cc, keys.secretKey, block_size)
    ct_prod = fp.dot(cc, keys, sum_col_keys, ctv_c, ctv_d)
    result = ct_prod.decrypt(cc, keys.secretKey, format=0)
    result = np.round(result, decimals=1)
    print(f"Expected: {dot_prod}")
    print(f"Obtained: {result}")
    print(f"Matching = [{np.array_equal(result[0], dot_prod)}]")

    print("\n4.Hadamard Product: a.b:")
    print(np.multiply(a, b))

    # print()
    # print("*" * 10, "TRANSPOSE", "*" * 10)
    # print("\n4.Matrix Transpose: A^T:")
    # a1 = np.array([[1, 2], [3, 4]])
    # aa = np.array([1, 2, 3, 4])
    # ctm_a1 = fp.array(cc, a, total_slots, pub_key=keys.publicKey)
    # matI = np.array(gen_comm_mat(len(aa), len(aa), 0))

    # print(matI, aa)
    # print("commutation matrix = ", matI)
    # expected = np.matmul(matI, aa)

    # pt_matI = fp.array(cc, data=matI, size=total_slots, block_size=len(a) * len(a))

    # size = block_size * block_size

    # sum_col_keys = fp.gen_sum_col_keys(cc, keys.secretKey, size)
    # ct_prod = fp.matvec(cc, keys, sum_col_keys, pt_matI, ctm_a, size)
    # result = ct_prod.decrypt(cc, keys.secretKey, format=0)
    # result = np.round(result, decimals=1)
    # print(f"Expected: {expected}")
    # print(f"Obtained: {result}")
    # print(f"Matching = [{np.array_equal(result, vec_ac)}]")

    # # %%
    # print()
    # print("*" * 10, "SUM", "*" * 10)
    # print("\n1. Matrix addition:")
    # ct_sum = fp.add(cc, ctm_a, ctm_b)
    # result = ct_sum.decrypt(cc, keys.secretKey)
    # result = np.round(result, decimals=1)
    # print(f"Expected: {a + b}")
    # print(f"Obtained: {result}")
    # print(f"Matching = [{np.array_equal(result, a + b)}]")

    # # %%
    # print("\nMean of array elements: mean(a)")
    # print(np.mean(a))

    # # %%
    # print("\nAddition: a+b")
    # print(np.add(a, b))

    # # %%
    # print("\nSubtraction: a - b")
    # print(np.subtract(a, b))

    # # %%
    # print("\nReduce by addition:")
    # print("Before addition: ", c)
    # print(np.add.reduce(c))

    # # %%
    # print("\nReduce by subtraction:")
    # print("Before subtraction: ", c)
    # print(np.subtract.reduce(c))

    # # %%
    # print("\nAccumulate by addition:")
    # print("Before accumulate by addition: ", c)
    # print(np.add.accumulate(c))

    # # %%
    # print("\nAccumulate by subtraction:")
    # print("Before accumulate by subtraction: ", c)
    # print(np.subtract.accumulate(c))


demo()
