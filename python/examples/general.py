# import openfhe related libraries
import numpy as np
from openfhe import *
from openfhe_matrix import *

# import openfhe_numpy library
import openfhe_numpy as onp
from openfhe_numpy.utils import *


#  TODO: add separate examples
#  TODO: transpose
#  TODO: tests
#  TODO: clean code + comments


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
    ringDimension = 2**7
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

    ctm_a = onp.array(cc, a, total_slots, public_key=keys.publicKey)
    ctm_b = onp.array(cc, b, total_slots, public_key=keys.publicKey)

    ctv_c = onp.array(cc, c, total_slots, block_size, "C", public_key=keys.publicKey)
    ctv_d = onp.array(cc, d, total_slots, block_size, "C", public_key=keys.publicKey)

    print()
    print("*" * 10, "ADDITION", "*" * 10)
    print("\n1. Matrix addition:")
    ct_sum = onp.add(cc, ctm_a, ctm_b)
    result = ct_sum.decrypt(keys.secretKey)
    result = np.round(result, decimals=1)
    print(f"Expected: {a + b}")
    print(f"Obtained: {result}")
    print(f"Matching = [{np.array_equal(result, a + b)}]")

    print("\n2. Vector addition:")
    ct_sum = onp.add(cc, ctv_c, ctv_d)
    result = ct_sum.decrypt(keys.secretKey)
    result = np.round(result, decimals=1)
    print(f"Expected: {c + d}")
    print(f"Obtained: {result}")
    print(f"Matching = [{np.array_equal(result, c + d)}]")

    print()
    print("*" * 10, "MULTIPLICATION", "*" * 10)

    print("\n1.Matrix multiplication:")
    ct_product = onp.square_matmul(ctm_a, ctm_b)
    result = ct_product.decrypt(keys.secretKey)
    result = np.round(result, decimals=1)
    print(f"Expected: {a @ b}")
    print(f"Obtained: {result}")
    print(f"Matching = [{np.array_equal(result, a @ b)}]")

    print("\n2.Matrix Vector multiplication: A@c")
    vec_ac = pack_vec_row_wise((a @ c), block_size, total_slots)
    sumkey = onp.gen_sum_col_keys(cc, keys.secretKey, block_size)
    ct_product = onp.matvec(cc, keys, sumkey, ctm_a, ctv_c, block_size)
    result = ct_product.decrypt(cc, keys.secretKey, format=0)
    result = np.round(result, decimals=1)
    print(f"Expected: {vec_ac}")
    print(f"Obtained: {result}")
    print(f"Matching = [{np.array_equal(result, vec_ac)}]")

    print("\n3.Dot product c.d = <c,d>:")
    dot_prod = np.dot(c, d)
    sumkey = onp.gen_sum_col_keys(cc, keys.secretKey, block_size)
    ct_product = onp.dot(cc, keys, sumkey, ctv_c, ctv_d)
    result = ct_product.decrypt(cc, keys.secretKey, format=0)
    result = np.round(result, decimals=1)
    print(f"Expected: {dot_prod}")
    print(f"Obtained: {result}")
    print(f"Matching = [{np.array_equal(result[0], dot_prod)}]")

    # print("\n4.Hadamard Product: a.b:")
    # print(np.multiply(a, b))

    # print()
    # print("*" * 10, "TRANSPOSE", "*" * 10)
    # print("\n4.Matrix Transpose: A^T:")
    # a1 = np.array([[1, 2], [3, 4]])
    # aa = np.array([1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4])

    # # aa = np.array([1, 2, 3, 4])
    # # ctm_a1 = onp.array(cc, a, total_slots, public_key=keys.publicKey)
    # matI = np.array(gen_comm_mat(2, 2, 0))

    # print(a)
    # print("commutation matrix = ", matI)
    # expected = np.matmul(matI, aa)
    # print(f"Expected: {expected}")

    # def gen_transpose_diag(total_slots, rowsize, i):
    #     n = rowsize * rowsize
    #     diag = np.zeros(n)

    #     if i >= 0:
    #         for l in range(n):
    #             for j in range(rowsize):
    #                 if (l - i == (rowsize + 1) * j) and (j < rowsize - i):
    #                     diag[i] = 1
    #     else:
    #         for l in range(n):
    #             for j in range(rowsize):
    #                 if l - i == (rowsize + 1) * j and -i <= j and j < rowsize:
    #                     diag[i] = 1
    #     return diag

    # def gen_transpose_diag(d, i):
    #     """Compute the non-zero diagonal vector t_{(d-1)*i}."""
    #     t_vec = np.zeros(d * d, dtype=int)
    #     if i >= 0:
    #         for j in range(0, d):
    #             if j < d - i:
    #                 l = (d + 1) * j + i
    #                 if l < d * d:
    #                     t_vec[l] = 1
    #     else:
    #         for j in range(-i, d):
    #             l = (d + 1) * j + i
    #             if l >= 0 and l < d * d:
    #                 t_vec[l] = 1
    #     return t_vec

    # # u = gen_transpose_diag(total_slots, 2, 0)
    # print("u = ", gen_transpose_diag(2, 0))
    # print("u = ", gen_transpose_diag(2, 1))
    # print("u = ", gen_transpose_diag(2, 2))
    # print("u = ", gen_transpose_diag(2, 3))

    # pt_matI = onp.array(cc, data=matI, size=total_slots, block_size=len(a) * len(a))

    # size = block_size * block_size

    # sumkey = onp.gen_sum_col_keys(cc, keys.secretKey, size)
    # ct_product = onp.matvec(cc, keys, sumkey, pt_matI, ctm_a, size)
    # result = ct_product.decrypt(cc, keys.secretKey, format=0)
    # result = np.round(result, decimals=1)
    # print(f"Expected: {expected}")
    # print(f"Obtained: {result}")
    # print(f"Matching = [{np.array_equal(result, vec_ac)}]")

    # # %%
    # print()
    print("*" * 10, "SUM", "*" * 10)
    print("\n1. Matrix addition:")
    ct_sum = onp.add(cc, ctm_a, ctm_b)
    result = ct_sum.decrypt(keys.secretKey)
    result = np.round(result, decimals=1)
    print(f"Expected: {a + b}")
    print(f"Obtained: {result}")
    print(f"Matching = [{np.array_equal(result, a + b)}]")

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
