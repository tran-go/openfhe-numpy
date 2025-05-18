# At the top of each test file
import numpy as np
import openfhe_numpy as onp

# Single import line replaces all the try/except logic
from tests.test_imports import *


def fhe_addition(params, input):
    """Execute vector addition with suppressed output."""
    total_slots = params["ringDim"] // 2

    with suppress_stdout():
        cc, keys = get_cached_crypto_context(params)
        tensor1 = onp.array(cc, np.array(input[0]), total_slots, public_key=keys.publicKey)
        tensor2 = onp.array(cc, np.array(input[1]), total_slots, public_key=keys.publicKey)
        tensor_sum = onp.add(tensor1, tensor2)
        result = tensor_sum.decrypt(keys.secretKey, True)

    return result


# Create test class to be discoverable for module running
class TestVectorAddition(MainUnittest):
    """Test class for vector addition operations."""

    @classmethod
    def _generate_test_cases(cls):
        """Generate test cases for vector addition."""
        ckks_param_list = load_ckks_params()
        vector_sizes = [2, 3, 8, 16]
        test_counter = 1

        for param in ckks_param_list:
            for size in vector_sizes:
                A = generate_random_array(rows=size, cols=1)
                B = generate_random_array(rows=size, cols=1)
                expected = np.array(A) + np.array(B)
                name = "TestVectorAddition"
                test_name = f"test_case_{test_counter}_ring_{param['ringDim']}_size_{size}"
                test_method = MainUnittest.generate_test_case(
                    fhe_addition, name, test_name, param, [A, B], expected
                )
                # Add to TestVectorAddition class for module discovery
                setattr(cls, test_name, test_method)
                test_counter += 1


if __name__ == "__main__":
    # Generate test cases and run with summary
    TestVectorAddition.setUpClass()
    TestVectorAddition.run_test_summary("Vector Addition")
else:
    # For module-based execution, generate test cases immediately
    TestVectorAddition.setUpClass()
