"""
TestMatrixSumCumCols:
Column-wise cumulative sum FHE tests for OpenFHE-NumPy.
"""

# Standard library imports
import argparse
import os
import sys
from typing import Any, Dict, List

# Third-party imports
import numpy as np

# Local/OpenFHE imports
import openfhe_numpy as onp
from tests.main_unittest import (
    MainUnittest,
    generate_random_array,
    gen_crypto_context,
    load_ckks_params,
    suppress_stdout,
)

# Module-level logger is used internally by the framework


def fhe_matrix_sumcum_cols(
    original_params: Dict[str, Any],
    inputs: List[np.ndarray],
) -> np.ndarray:
    """Execute matrix column summation with minimal stdout noise."""
    params = original_params.copy()
    matrix = inputs[0]

    # Ensure sufficient multiplicative depth
    if params["multiplicativeDepth"] < matrix.shape[1]:
        params["multiplicativeDepth"] = matrix.shape[1] + 1

    # Suppress verbose context creation
    with suppress_stdout(True):
        cc, keys = gen_crypto_context(params)

    total_slots = params["ringDim"] // 2
    ctm_matrix = onp.array(cc, matrix, total_slots, public_key=keys.publicKey)
    onp.gen_accumulate_cols_key(keys.secretKey, ctm_matrix.ncols)

    # Perform the column-wise cumulative sum
    ctm_result = onp.cumsum(ctm_matrix, axis=1, inplace=True)
    return ctm_result.decrypt(keys.secretKey, format_type="reshape")


class TestMatrixSumCumCols(MainUnittest):
    """Test suite for homomorphic column-wise cumulative sum."""

    # Force test case generation at import time
    _test_cases_generated = False

    @classmethod
    def setUpClass(cls):
        """Ensure tests are generated before discovery."""
        super().setUpClass()
        if not cls._test_cases_generated:
            cls._generate_test_cases()
            cls._test_cases_generated = True

    @classmethod
    def _generate_test_cases(cls):
        ckks_params = load_ckks_params()
        matrix_sizes = [2, 3, 8, 16]
        test_counter = 1

        for params in ckks_params:
            for size in matrix_sizes:
                # Generate random test matrix and expected result
                A = generate_random_array(size)
                expected = np.cumsum(A, axis=1)

                name = "TestMatrixSumCumCols"
                test_name = f"test_sumcum_cols_{test_counter}_ring_{params['ringDim']}_size_{size}"

                # Create the test method dynamically
                test_method = cls.generate_test_case(
                    func=fhe_matrix_sumcum_cols,
                    name=name,
                    test_name=test_name,
                    params=params,
                    input_data=[A],
                    expected=expected,
                    debug=False,  # set True to see context creation logs
                )

                setattr(cls, test_name, test_method)
                test_counter += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TestMatrixSumCumCols only")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    args = parser.parse_args()

    name = os.path.basename(__file__)
    sys.exit(
        TestMatrixSumCumCols.run_test_summary("Matrix Cumulative Sum Columns", debug=args.debug)
    )
