import unittest
import numpy as np
import openfhe_numpy as fp
from main_unittest import MainUnittest
from main_unittest import load_ckks_params
from main_unittest import generate_random_array
from main_unittest import gen_crypto_context_from_params


def fhe_matrix_addition(params, input):
    total_slots = params["ringDim"] // 2
    cc, keys = gen_crypto_context_from_params(params)

    matrixA = np.array(input[0])
    matrixB = np.array(input[1])

    ct_matrixA = fp.array(cc, matrixA, total_slots, public_key=keys.publicKey)
    ct_matrixB = fp.array(cc, matrixB, total_slots, public_key=keys.publicKey)

    ct_sum = fp.add(ct_matrixA, ct_matrixB)
    result = ct_sum.decrypt(keys.secretKey)

    return result


if __name__ == "__main__":
    ckks_param_list = load_ckks_params()
    matrix_sizes = [2, 3, 8, 16]
    test_counter = 1

    for param in ckks_param_list:
        for size in matrix_sizes:
            A = generate_random_array(size)
            B = generate_random_array(size)
            expected = (np.array(A) + np.array(B)).tolist()
            name = "TestMatrixAddition"
            test_name = f"test_case_{test_counter}_ring_{param['ringDim']}_size_{size}"
            test_method = MainUnittest.generate_test_case(
                fhe_matrix_addition, name, test_name, param, [A, B], expected
            )
            setattr(MainUnittest, test_name, test_method)
            test_counter += 1

    unittest.main(argv=[""], exit=False)
