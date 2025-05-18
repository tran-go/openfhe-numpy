# At the top of each test file
import numpy as np
import openfhe_numpy as onp

# Single import line replaces all the try/except logic
from tests.test_imports import *


def fhe_vector_dot(params, input):
    """Execute vector dot product with suppressed output."""
    total_slots = params["ringDim"] // 2

    with suppress_stdout():
        cc, keys = get_cached_crypto_context(params)
        public_key = keys.publicKey
        input_a = np.array(input[0])
        input_b = np.array(input[1])
        if input_a.ndim == 1:
            ctm_input_a = onp.array(cc, input_a, total_slots, public_key=keys.publicKey)
            ctm_input_b = onp.array(cc, input_b, total_slots, public_key=keys.publicKey)
        else:
            ctm_input_a = onp.array(cc, input_a, total_slots, public_key=keys.publicKey)
            ctm_input_b = onp.array(cc, input_b, total_slots, public_key=keys.publicKey)

        ctm_dot = onp.dot(ctm_input_a, ctm_input_b)
        result = ctm_dot.decrypt(keys.secretKey, format_type="reshape", new_shape=(1,))

    return result


# Create test class to be discoverable for module running
class TestVectorInnerProduct(MainUnittest):
    """Test class for vector inner product operations."""

    @classmethod
    def _generate_test_cases(cls):
        """Generate test cases for vector inner product."""
        ckks_param_list = load_ckks_params()
        vector_sizes = [2, 3, 8, 16]
        dims = [1, 2]  # Fixed variable name
        test_counter = 1

        for param in ckks_param_list:
            for size in vector_sizes:
                input_a = generate_random_array(size, 1)
                input_b = generate_random_array(size, 1)
                expected = np.dot(input_a, input_b)
                name = "TestVectorInnerProduct"
                test_name = f"test_case_{test_counter}_ring_{param['ringDim']}_size_{size}"
                test_method = MainUnittest.generate_test_case(
                    fhe_vector_dot, name, test_name, param, [input_a, input_b], [expected]
                )
                # Add to TestVectorInnerProduct class for module discovery
                setattr(cls, test_name, test_method)
                test_counter += 1


if __name__ == "__main__":
    # Generate test cases and run with summary
    TestVectorInnerProduct.setUpClass()
    TestVectorInnerProduct.run_test_summary("Vector Inner Product")
else:
    # For module-based execution, generate test cases immediately
    TestVectorInnerProduct.setUpClass()
