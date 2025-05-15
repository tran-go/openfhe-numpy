import unittest
import numpy as np
import openfhe_numpy as onp
from main_unittest import MainUnittest
from main_unittest import load_ckks_params
from main_unittest import generate_random_array
from main_unittest import gen_crypto_context_from_params


def fhe_addition(params, input):
    total_slots = params["ringDim"] // 2
    cc, keys = gen_crypto_context_from_params(params)

    ciphext1 = onp.array(cc, np.array(input[0]), total_slots, public_key=keys.publicKey)
    ciphext2 = onp.array(cc, np.array(input[1]), total_slots, public_key=keys.publicKey)

    ct_sum = onp.add(ciphext1, ciphext2)
    result = ct_sum.decrypt(keys.secretKey)

    return result


if __name__ == "__main__":
    ckks_param_list = load_ckks_params()
    matrix_sizes = [2, 3, 8, 16]
    test_counter = 1

    for param in ckks_param_list:
        for size in matrix_sizes:
            A = generate_random_array(size, 1)
            B = generate_random_array(size, 1)
            expected = (np.array(A) + np.array(B)).tolist()
            name = "TestMatrixAddition"
            test_name = f"test_case_{test_counter}_ring_{param['ringDim']}_size_{size}"
            test_method = MainUnittest.generate_test_case(
                fhe_addition, name, test_name, param, [A, B], expected
            )
            setattr(MainUnittest, test_name, test_method)
            test_counter += 1

    unittest.main(argv=[""], exit=False)
