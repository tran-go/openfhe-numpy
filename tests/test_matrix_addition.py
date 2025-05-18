# At the top of each test file
import numpy as np
import openfhe_numpy as onp

# Single import line replaces all the try/except logic
from tests.test_imports import *


def fhe_matrix_addition(params, input):
    """Execute matrix addition with suppressed output."""
    total_slots = params["ringDim"] // 2

    with suppress_stdout():
        cc, keys = get_cached_crypto_context(params)
        matrixA = np.array(input[0])
        matrixB = np.array(input[1])
        ctm_A = onp.array(cc, matrixA, total_slots, public_key=keys.publicKey)
        ctm_B = onp.array(cc, matrixB, total_slots, public_key=keys.publicKey)
        ctm_sum = onp.add(ctm_A, ctm_B)
        result = ctm_sum.decrypt(keys.secretKey, True)

    return result


# Create test class to be discoverable for module running
class TestMatrixAddition(MainUnittest):
    """Test class for matrix addition operations."""

    @classmethod
    def _generate_test_cases(cls):
        """Generate test cases for matrix addition."""
        ckks_param_list = load_ckks_params()
        matrix_sizes = [2, 3, 8, 16]
        test_counter = 1

        for param in ckks_param_list:
            for size in matrix_sizes:
                A = generate_random_array(size)
                B = generate_random_array(size)
                expected = np.array(A) + np.array(B)
                name = "TestMatrixAddition"
                test_name = f"test_case_{test_counter}_ring_{param['ringDim']}_size_{size}"
                test_method = MainUnittest.generate_test_case(
                    fhe_matrix_addition, name, test_name, param, [A, B], expected
                )
                # Add to TestMatrixAddition class for module discovery
                setattr(cls, test_name, test_method)
                test_counter += 1


if __name__ == "__main__":
    # Generate test cases and run with summary
    TestMatrixAddition.setUpClass()
    TestMatrixAddition.run_test_summary("Matrix Addition")
else:
    # For module-based execution, generate test cases immediately
    TestMatrixAddition.setUpClass()
