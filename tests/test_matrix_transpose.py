# At the top of each test file
import numpy as np
import openfhe_numpy as onp

# Single import line replaces all the try/except logic
from tests.test_imports import *


def fhe_matrix_transpose(params, input):
    """Execute matrix transpose with suppressed output."""
    total_slots = params["ringDim"] // 2

    with suppress_stdout():
        cc, keys = get_cached_crypto_context(params)
        cc.Get
        public_key = keys.publicKey
        matrixA = np.array(input[0])
        ct_matrixA = onp.array(cc, matrixA, total_slots, public_key=public_key)
        onp.gen_transpose_keys(keys.secretKey, ct_matrixA)
        ct_result = onp.transpose(ct_matrixA)
        result = ct_result.decrypt(keys.secretKey, True)

    return result


# Create test class to be discoverable for module running
class TestMatrixTranspose(MainUnittest):
    """Test class for matrix transpose operations."""

    @classmethod
    def _generate_test_cases(cls):
        """Generate test cases for matrix transpose."""
        ckks_param_list = load_ckks_params()
        matrix_sizes = [2, 3, 8, 16]
        test_counter = 1

        for param in ckks_param_list:
            for size in matrix_sizes:
                A = generate_random_array(size)
                expected = A.T
                name = "TestMatrixTranspose"
                test_name = f"test_case_{test_counter}_ring_{param['ringDim']}_size_{size}"
                test_method = MainUnittest.generate_test_case(
                    fhe_matrix_transpose, name, test_name, param, [A], expected
                )
                # Add to TestMatrixTranspose class for module discovery
                setattr(cls, test_name, test_method)
                test_counter += 1


if __name__ == "__main__":
    # Generate test cases and run with summary
    TestMatrixTranspose.setUpClass()
    TestMatrixTranspose.run_test_summary("Matrix Transpose")
else:
    # For module-based execution, generate test cases immediately
    TestMatrixTranspose.setUpClass()
