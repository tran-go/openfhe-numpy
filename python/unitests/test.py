import unittest
import numpy as np

import openfhe_numpy as fp

from main_unittest import MainUnittest
from main_unittest import load_ckks_params
from main_unittest import generate_random_array
from main_unittest import gen_crypto_context_from_params


def fhe_vector_dot(params, input):
    total_slots = params["ringDim"] // 2
    cc, keys = gen_crypto_context_from_params(params)
    pub_key = keys.publicKey
    input0 = np.array(input[0])
    input1 = np.array(input[1])
    if input0.ndim == 1:
        ct_input0 = fp.array(cc, input0, total_slots, 1, "C", pub_key=keys.publicKey)
        ct_input1 = fp.array(cc, input1, total_slots, 1, "C", pub_key=keys.publicKey)
    else:
        ct_input0 = fp.array(cc, input0, total_slots, pub_key=keys.publicKey)
        ct_input1 = fp.array(cc, input1, total_slots, pub_key=keys.publicKey)

    return fp.dot(cc, ct_input0, ct_input1).decrypt(cc, keys.secretKey)


if __name__ == "__main__":
    ckks_param_list = load_ckks_params()
    matrix_sizes = [2, 3, 8, 16]
    dim = [1, 2]
    test_counter = 1

    for param in ckks_param_list:
        for size in matrix_sizes:
            for d in dims:
                input0 = generate_random_array(size, d)
                input1 = generate_random_array(size, d)
                expected = np.dot(input0, input1)
                name = "TestSquareMatrixProduct"
                test_name = f"test_case_{test_counter}_ring_{param['ringDim']}_size_{size}"
                test_method = MainUnittest.generate_test_case(
                    fhe_vector_dot, name, test_name, param, [input0, input1], expected
                )
                setattr(MainUnittest, test_name, test_method)
                test_counter += 1

    unittest.main(argv=[""], exit=False)
