import unittest
import numpy as np

import openfhe_numpy as onp

# from openfhe_matrix import *
from main_unittest import MainUnittest
from main_unittest import load_ckks_params
from main_unittest import generate_random_array
from main_unittest import gen_crypto_context_from_params


def fhe_matrix_sumcum(params, input):
    total_slots = params["ringDim"] // 2
    cc, keys = gen_crypto_context_from_params(params)
    public_key = keys.publicKey
    matrixA = np.array(input[0])
    ct_matrixA = onp.array(cc, matrixA, total_slots, public_key=public_key)
    onp.gen_accumulate_cols_key(keys.secretKey, ct_matrixA.ncols)
    ct_result = onp.sum(ct_matrixA, 1)
    result = ct_result.decrypt(keys.secretKey)
    return result


if __name__ == "__main__":
    ckks_param_list = load_ckks_params()
    matrix_sizes = [2, 3, 8, 16]
    test_counter = 1

    for param in ckks_param_list:
        for size in matrix_sizes:
            A = generate_random_array(size)
            expected = np.sum(A, axis=1)
            print("expected = \n", expected)
            name = "TestMatrixTranspose"
            test_name = f"test_case_{test_counter}_ring_{param['ringDim']}_size_{size}"
            test_method = MainUnittest.generate_test_case(
                fhe_matrix_sumcum, name, test_name, param, [A], expected
            )
            setattr(MainUnittest, test_name, test_method)
            test_counter += 1

    unittest.main(argv=[""], exit=False)
