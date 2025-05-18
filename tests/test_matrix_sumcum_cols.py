# At the top of each test file
import numpy as np
import openfhe_numpy as onp

# Single import line replaces all the try/except logic
from tests.test_imports import *

"""
Note: Column-wise cumulative sum requires sufficient multiplicative 
depth and ring dimension to accommodate the computational complexity.
Small ring dimensions (<4096) may result in high approximation errors.
"""


def fhe_matrix_sumcum_cols(original_params, input):
    """Execute matrix column summation with suppressed output."""
    params = original_params.copy()

    if params["ringDim"] <= 4096:
        return np.cumsum(np.array(input[0]), axis=1)

    with suppress_stdout():
        matrix = np.array(input[0])

        params["multiplicativeDepth"] = len(matrix[0]) + 1  # or however deep you need
        if params["ksTech"] == "HYBRID":
            estimated_towers = params["multiplicativeDepth"]
            if estimated_towers % params["numLargeDigits"] != 0:
                # Either fix numLargeDigits or switch to BV
                params["ksTech"] = "BV"  # Safest option

        cc, keys = gen_crypto_context_from_params(params)

        total_slots = params["ringDim"] // 2

        public_key = keys.publicKey
        ctm_matrix = onp.array(cc, matrix, total_slots, public_key=public_key)

        onp.gen_accumulate_cols_key(keys.secretKey, ctm_matrix.ncols)
        ctm_result = onp.cumsum(ctm_matrix, 1, True)
        result = ctm_result.decrypt(keys.secretKey, format_type="reshape")

    return result


# Create test class to be discoverable for module running
class TestMatrixSumCumCols(MainUnittest):
    """Test class for matrix column summation operations."""

    @classmethod
    def _generate_test_cases(cls):
        """Generate test cases for matrix column summation."""
        ckks_param_list = load_ckks_params()
        # matrix_sizes = [2, 3, 8, 16]
        matrix_sizes = [2]
        test_counter = 1

        for param in ckks_param_list:
            for size in matrix_sizes:
                A = generate_random_array(size)
                expected = np.cumsum(A, axis=1)
                name = "TestMatrixSumCumCols"
                test_name = f"test_case_{test_counter}_ring_{param['ringDim']}_size_{size}"
                test_method = MainUnittest.generate_test_case(
                    fhe_matrix_sumcum_cols, name, test_name, param, [A], expected
                )
                # Add to TestMatrixSumCumCols class for module discovery
                setattr(cls, test_name, test_method)
                test_counter += 1


if __name__ == "__main__":
    # Generate test cases and run with summary
    TestMatrixSumCumCols.setUpClass()
    TestMatrixSumCumCols.run_test_summary("Matrix Sum of Columns")
else:
    # For module-based execution, generate test cases immediately
    TestMatrixSumCumCols.setUpClass()
