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

# Import from C++ extension module (explicit imports instead of wildcard)
from ._openfhe_numpy import (
    # Enums
    LinTransType,
    MatVecEncoding,
    ArrayEncodingType,
    # Key generation operations
    EvalLinTransKeyGen,
    EvalSumCumRowsKeyGen,
    EvalSumCumColsKeyGen,
    EvalSquareMatMultRotateKeyGen,
    # Matrix transformation operations
    EvalLinTransSigma,
    EvalLinTransTau,
    EvalLinTransPhi,
    EvalLinTransPsi,
    EvalTranspose,
    EvalMatMulSquare,
    EvalMultMatVec,
    # Reduction operations
    EvalSumCumRows,
    EvalSumCumCols,
    EvalReduceCumRows,
    EvalReduceCumCols,
)

# Import tensor classes
from .tensor.tensor import BaseTensor, FHETensor, PTArray, CTArray
from .tensor.constructors import array

# Import configuration
from .config import MatrixOrder, DataType

# Import operations
from .operations.crypto_context import (
    gen_sum_row_keys,
    gen_sum_col_keys,
    gen_rotation_keys,
    gen_lintrans_keys,
    gen_transpose_keys,
    gen_square_matrix_product,
)

from .operations.arithmetic import (
    add,
    multiply,
    dot,
    matmul,
    transpose,
    power,
    sum,
    reduce,
)

# Import specific utility functions rather than the module
from .utils.utils import (
    is_power_of_two,
    next_power_of_two,
    check_equality_matrix,
)

# Define public API
__all__ = [
    # Core tensor classes
    "BaseTensor",
    "FHETensor",
    "PTArray",
    "CTArray",
    # Constructor functions
    "array",
    # Enumerations
    "LinTransType",
    "MatVecEncoding",
    "ArrayEncodingType",
    "MatrixOrder",
    "DataType",
    # Matrix operations from C++ extension
    "EvalLinTransSigma",
    "EvalLinTransTau",
    "EvalLinTransPhi",
    "EvalLinTransPsi",
    "EvalTranspose",
    "EvalMatMulSquare",
    "EvalMultMatVec",
    # High-level operations
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
    # Utility functions
    "is_power_of_two",
    "next_power_of_two",
    "check_equality_matrix",
    # Version information
    "__version__",
]
