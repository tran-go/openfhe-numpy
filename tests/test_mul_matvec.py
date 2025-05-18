# At the top of each test file
import numpy as np
import openfhe_numpy as onp

# Single import line replaces all the try/except logic
from tests.test_imports import *


def fhe_matrix_vector_product(params, input):
    """Execute matrix-vector product with suppressed output."""
    total_slots = params["ringDim"] // 2

    with suppress_stdout():
        cc, keys = get_cached_crypto_context(params)
        public_key = keys.publicKey
        matrix = np.array(input[0])
        vector = np.array(input[1])

        ctm_matrix = onp.array(cc, matrix, total_slots, public_key=keys.publicKey)
        ncols = ctm_matrix.ncols
        sumkey = onp.sum_col_keys(cc, keys.secretKey)
        ctm_matrix.extra["colkey"] = sumkey

        ctv_vector = onp.array(cc, vector, total_slots, ncols, "C", public_key=keys.publicKey)

        ctv_result = ctm_matrix @ ctv_vector

        # todo: write return format
        result = ctv_result.decrypt(keys.secretKey, format_type="re")

    return result


# Create test class to be discoverable for module running
class TestMatrixVectorProduct(MainUnittest):
    """Test class for matrix-vector product operations."""

    @classmethod
    def _generate_test_cases(cls):
        """Generate test cases for matrix-vector product."""
        ckks_param_list = load_ckks_params()
        matrix_sizes = [2, 3, 8, 16]
        test_counter = 1

        for param in ckks_param_list:
            for size in matrix_sizes:
                A = generate_random_array(size)
                b = generate_random_array(size, 1)
                # expected = np.matmul(np.array(A), np.array(b)).tolist()
                size = onp.next_power_of_two(size)
                expected = onp.pack_vec_row_wise((A @ b), size, total_slots)

                name = "TestMatrixVectorProduct"
                test_name = f"test_case_{test_counter}_ring_{param['ringDim']}_size_{size}"
                test_method = MainUnittest.generate_test_case(
                    fhe_matrix_vector_product, name, test_name, param, [A, b], expected
                )
                # Add to TestMatrixVectorProduct class for module discovery
                setattr(cls, test_name, test_method)
                test_counter += 1


if __name__ == "__main__":
    # Generate test cases and run with summary
    TestMatrixVectorProduct.setUpClass()
    TestMatrixVectorProduct.run_test_summary("Matrix-Vector Product")
else:
    # For module-based execution, generate test cases immediately
    TestMatrixVectorProduct.setUpClass()
