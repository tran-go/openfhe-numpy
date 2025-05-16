"""
OpenFHE-NumPy: A NumPy-inspired framework for homomorphic encryption operations.

This module provides a familiar NumPy-like interface for performing homomorphic
encryption operations using the OpenFHE library.
"""

# openfhe_numpy / __init__.py

import importlib
import sys

# Version information
try:
    from .version import __version__
except ImportError:
    __version__ = "unknown"

# Core imports that are always needed
from ._openfhe_numpy import (
    LinTransType,
    MatVecEncoding,
    MulDepthAccumulation,
    EvalLinTransKeyGen,
    EvalSquareMatMultRotateKeyGen,
    EvalSumCumRowsKeyGen,
    EvalSumCumColsKeyGen,
    EvalMultMatVec,
    EvalMatMulSquare,
    EvalTranspose,
    EvalSumCumRows,
    EvalSumCumCols,
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
from .operations.matrix_api import add, multiply, dot, matmul, transpose, power, cumsum, cumreduce

# Import crypto context utilities
from .operations.crypto_context import (
    sum_row_keys,
    gen_sum_col_keys,
    gen_rotation_keys,
    gen_lintrans_keys,
    gen_transpose_keys,
    gen_square_matmult_key,
    gen_accumulate_rows_key,
    gen_accumulate_cols_key,  # <-- add this line
)

# Import utility functions
from .utils.utils import is_power_of_two, next_power_of_two, check_equality_matrix

# Import log functions
from .utils.log import ONP_WARNING, ONP_DEBUG, ONP_ERROR, ONPNotImplementedError

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
    "MulDepthAccumulation",
    "EvalLinTransKeyGen",
    "EvalSquareMatMultRotateKeyGen",
    "EvalSumCumRowsKeyGen",
    "EvalSumCumColsKeyGen",
    "EvalMultMatVec",
    "EvalMatMulSquare",
    "EvalTranspose",
    "EvalSumCumRows",
    "EvalSumCumCols",
    # Key generation utilities
    "sum_row_keys",
    "gen_sum_col_keys",
    "gen_rotation_keys",
    "gen_lintrans_keys",
    "gen_transpose_keys",
    "gen_square_matmult_key",
    "gen_accumulate_rows_key",
    "gen_accumulate_cols_key",
    # log
    "ONP_WARNING",
    "ONP_DEBUG",
    "ONP_ERROR",
    "ONPNotImplementedError",
]


def __getattr__(name):
    """
    Lazy-loads submodules on first attribute access.
    """
    if name in ("tensor", "operations", "utils", "logs"):
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """
    Ensures dir() and IDE completions include both submodules and re-exported names.
    """
    return sorted(list(globals().keys()) + __all__)
