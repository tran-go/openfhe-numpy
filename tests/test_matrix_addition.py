import numpy as np
import openfhe_numpy as onp

# Import directly from main_unittest - aligned with new framework
from tests.main_unittest import (
    generate_random_array,
    gen_crypto_context,
    load_ckks_params,
    suppress_stdout,
    MainUnittest,
)


def fhe_matrix_addition(params, input):
    """Execute matrix addition with FHE."""
    total_slots = params["ringDim"] // 2

    # Get crypto context with optimized function
    cc, keys = gen_crypto_context(params)

    # Parse input matrices
    matrixA = np.array(input[0])
    matrixB = np.array(input[1])

    # Encrypt matrices with explicit encoding order
    ctm_A = onp.array(cc, matrixA, total_slots, onp.ROW_MAJOR, public_key=keys.publicKey)
    ctm_B = onp.array(cc, matrixB, total_slots, onp.ROW_MAJOR, public_key=keys.publicKey)

    # Perform homomorphic addition
    ctm_sum = onp.add(ctm_A, ctm_B)

    # Decrypt and format result
    result = ctm_sum.decrypt(keys.secretKey, unpack_type="original")
    result = np.round(result, decimals=1)

    return result


class TestMatrixAddition(MainUnittest):
    """Test class for matrix addition operations."""

    @classmethod
    def _generate_test_cases(cls):
        """Generate test cases for matrix addition."""
        # Load parameters and define test dimensions
        ckks_param_list = load_ckks_params()
        matrix_sizes = [2, 3, 8, 16]
        test_counter = 1

        for param in ckks_param_list:
            for size in matrix_sizes:
                # Generate random test matrices
                A = generate_random_array(size)
                B = generate_random_array(size)

                # Calculate expected result directly
                expected = np.array(A) + np.array(B)

                # Create test name with descriptive format
                name = "TestMatrixAddition"
                test_name = f"test_id_{test_counter:03d}_ring_{param['ringDim']}_size_{size}"

                # Generate the test case with debug output
                test_method = MainUnittest.generate_test_case(
                    func=fhe_matrix_addition,
                    name=name,
                    test_name=test_name,
                    params=param,
                    input_data=[A, B],
                    expected=expected,
                    compare_fn=onp.check_equality_matrix,  # Use matrix comparison
                    tolerance=0.1,  # Add tolerance for floating-point comparison
                    debug=True,  # Enable debug output
                )

                # Register the test method
                setattr(cls, test_name, test_method)
                test_counter += 1


TestMatrixAddition._generate_test_cases()

if __name__ == "__main__":
    # Interactive test execution mode
    TestMatrixAddition.run_test_summary("Matrix Addition", debug=True)
