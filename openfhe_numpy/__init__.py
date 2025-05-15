"""
OpenFHE-NumPy: A NumPy-inspired framework for homomorphic encryption operations.

This module provides a familiar NumPy-like interface for performing homomorphic
encryption operations using the OpenFHE library.
"""

# Version information
try:
    from .version import __version__
except ImportError:
    __version__ = "unknown"

# Core imports that are always needed
from ._openfhe_numpy import (
    LinTransType,
    MatVecEncoding,
)

# Import tensor classes
from .tensor import (
    BaseTensor,
    FHETensor,
    PTArray,
    CTArray,
    BlockFHETensor,
    BlockCTArray,
    array,
)

# Import operations API
from .operations.api import add, multiply, dot, matmul, transpose, power, cumsum, cumreduce

# Import crypto context utilities
from .operations.crypto_context import (
    gen_sum_row_keys,
    gen_sum_col_keys,
    gen_rotation_keys,
    gen_lintrans_keys,
    gen_transpose_keys,
    gen_square_matrix_product,
    gen_accumulate_rows_key,
    gen_accumulate_cols_key,  # <-- add this line
)

# Import utility functions
from .utils.utils import is_power_of_two, next_power_of_two, check_equality_matrix

# Define complete public API
__all__ = [
    # Tensor classes
    "BaseTensor",
    "FHETensor",
    "PTArray",
    "CTArray",
    "BlockFHETensor",
    "BlockCTArray",
    # Constructors
    "array",
    # Operations
    "add",
    "multiply",
    "dot",
    "matmul",
    "transpose",
    "power",
    "cumsum",
    "cumreduce",
    # Utilities
    "is_power_of_two",
    "next_power_of_two",
    "check_equality_matrix",
    # Core OpenFHE types
    "LinTransType",
    "MatVecEncoding",
    # Key generation utilities
    "gen_sum_row_keys",
    "gen_sum_col_keys",
    "gen_rotation_keys",
    "gen_lintrans_keys",
    "gen_transpose_keys",
    "gen_square_matrix_product",
    "gen_accumulate_rows_key",
    "gen_accumulate_cols_key",
]
