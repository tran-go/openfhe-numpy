import unittest
import numpy as np

import openfhe_numpy as onp

from main_unittest import MainUnittest
from main_unittest import load_ckks_params
from main_unittest import generate_random_array
from main_unittest import gen_crypto_context_from_params


def fhe_square_matrix_product(params, input):
    total_slots = params["ringDim"] // 2
    cc, keys = gen_crypto_context_from_params(params)
    public_key = keys.publicKey
    matrixA = np.array(input[0])
    matrixB = np.array(input[1])

    ct_matrixA = onp.array(cc, matrixA, total_slots, public_key=public_key)
    ct_matrixB = onp.array(cc, matrixB, total_slots, public_key=public_key)
    block_size = ct_matrixA.ncols
    onp.gen_square_matmult_key(keys.secretKey, block_size)
    ct_result = onp.matmul(ct_matrixA, ct_matrixB)
    return ct_result.decrypt(keys.secretKey)


if __name__ == "__main__":
    ckks_param_list = load_ckks_params()
    matrix_sizes = [2, 3, 8, 16]
    test_counter = 1

    for param in ckks_param_list:
        for size in matrix_sizes:
            matrixA = generate_random_array(size)
            matrixB = generate_random_array(size)
            expected = np.array(matrixA) @ np.array(matrixB)
            name = "TestSquareMatrixProduct"
            test_name = f"test_case_{test_counter}_ring_{param['ringDim']}_size_{size}"
            test_method = MainUnittest.generate_test_case(
                fhe_square_matrix_product, name, test_name, param, [matrixA, matrixB], expected
            )
            setattr(MainUnittest, test_name, test_method)
            test_counter += 1

    unittest.main(argv=[""], exit=False)
