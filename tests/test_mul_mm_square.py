# At the top of each test file
import numpy as np
import openfhe_numpy as onp

# Single import line replaces all the try/except logic
from tests.test_imports import *


def fhe_square_matrix_product(params, input):
    """Execute square matrix product with suppressed output."""
    total_slots = params["ringDim"] // 2

    with suppress_stdout():
        cc, keys = get_cached_crypto_context(params)
        public_key = keys.publicKey
        A = np.array(input[0])
        B = np.array(input[1])
        ctm_matA = onp.array(cc, A, total_slots, public_key=keys.publicKey)
        ctm_matB = onp.array(cc, B, total_slots, public_key=keys.publicKey)
        ncols = ctm_matA.ncols

        onp.EvalSquareMatMultRotateKeyGen(keys.secretKey, ctm_matA.ncols)
        ctm_result = ctm_matA @ ctm_matB

        result = ctm_result.decrypt(keys.secretKey, format_type="reshape")

    return result


# Create test class to be discoverable for module running
class TestSquareMatrixProduct(MainUnittest):
    """Test class for square matrix product operations."""

    @classmethod
    def _generate_test_cases(cls):
        """Generate test cases for square matrix product."""
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
                # Add to TestSquareMatrixProduct class for module discovery
                setattr(cls, test_name, test_method)
                test_counter += 1


if __name__ == "__main__":
    # Generate test cases and run with summary
    TestSquareMatrixProduct.setUpClass()
    TestSquareMatrixProduct.run_test_summary("Square Matrix Product")
else:
    # For module-based execution, generate test cases immediately
    TestSquareMatrixProduct.setUpClass()
