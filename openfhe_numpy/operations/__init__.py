"""Operations for homomorphic encryption tensors."""

# Import arithmetic operations
from .arithmetic import (
    add,
    multiply,
    dot,
    matmul,
    transpose,
    power,
    sum,
    reduce,
)

# Import crypto context utilities
from .crypto_context import (
    gen_sum_row_keys,
    gen_sum_col_keys,
    gen_rotation_keys,
    gen_lintrans_keys,
    gen_transpose_keys,
    gen_square_matrix_product,
)

# Define public API
__all__ = [
    # Arithmetic and matrix operations
    "add",
    "multiply",
    "dot",
    "matmul",
    "transpose",
    "power",
    "sum",
    "reduce",
    # Key generation utilities
    "gen_sum_row_keys",
    "gen_sum_col_keys",
    "gen_rotation_keys",
    "gen_lintrans_keys",
    "gen_transpose_keys",
    "gen_square_matrix_product",
]
