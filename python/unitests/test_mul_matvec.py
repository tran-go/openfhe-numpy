import unittest
import numpy as np

import openfhe_numpy as fp

from main_unittest import MainUnittest
from main_unittest import load_ckks_params
from main_unittest import generate_random_array
from main_unittest import gen_crypto_context_from_params


def fhe_matrix_vector_product(params, input):
    total_slots = params["ringDim"] // 2
    cc, keys = gen_crypto_context_from_params(params)
    public_key = keys.publicKey
    matrix = np.array(input[0])
    vector = np.array(input[1])

    ct_matrix = fp.array(cc, matrix, total_slots, public_key=public_key)
    block_size = ct_matrix.rowsize
    sumkey = fp.gen_sum_col_keys(cc, keys.secretKey, block_size)

    ct_vector = fp.array(cc, vector, total_slots, block_size, "C", public_key=public_key)

    ct_result = fp.matvec(cc, keys, sumkey, ct_matrix, ct_vector, block_size)
    result = ct_result.decrypt(keys.secretKey)
    return result


if __name__ == "__main__":
    ckks_param_list = load_ckks_params()
    matrix_sizes = [2, 3, 8, 16]
    test_counter = 1

    for param in ckks_param_list:
        for size in matrix_sizes:
            A = generate_random_array(size)
            b = generate_random_array(size, 1)
            expected = np.matmul(np.array(A), np.array(b)).tolist()
            name = "TestMatrixVecProduct"
            test_name = f"test_case_{test_counter}_ring_{param['ringDim']}_size_{size}"
            test_method = MainUnittest.generate_test_case(
                fhe_matrix_vector_product, name, test_name, param, [A, b], expected
            )
            setattr(MainUnittest, test_name, test_method)
            test_counter += 1

    unittest.main(argv=[""], exit=False)
