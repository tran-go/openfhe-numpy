"""
OpenFHE-NumPy: A NumPy-inspired framework for homomorphic encryption operations.

This module provides a NumPy-style API for homomorphic encryption using OpenFHE.

Example:
    ```python
    from openfhe_numpy import array, add
    ct = array([1, 2, 3])
    result = add(ct, ct)
    ```
"""

import importlib
import sys
from typing import Any, Dict, List

# Version handling
try:
    from .version import __version__  # CMake-generated version
except ImportError:
    try:
        from ._version import __version__  # Development version
    except ImportError:
        __version__ = "0.0.1"  # Fallback version

_MODULE_EXPORTS = {
    "tensor": [
        "BaseTensor",
        "FHETensor",
        "PTArray",
        "CTArray",
        "BlockFHETensor",
        "BlockCTArray",
        "array",
    ],
    "operations.matrix_api": [
        "add",
        "multiply",
        "dot",
        "matmul",
        "transpose",
        "power",
        "cumsum",
        "cumreduce",
        "sum",
    ],
    "operations.crypto_context": [
        "sum_row_keys",
        "sum_col_keys",
        "gen_rotation_keys",
        "gen_lintrans_keys",
        "gen_transpose_keys",
        "gen_square_matmult_key",
        "gen_accumulate_rows_key",
        "gen_accumulate_cols_key",
    ],
    "_onp_cpp": [
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
    ],
    "utils.utils": [
        "is_power_of_two",
        "next_power_of_two",
        "check_equality_matrix",
        "pack_vector_row_wise",
    ],
    "config": [
        "MatrixOrder",
        "DataType",
        "EPSILON",
        "EPSILON_HIGH",
        "FormatType",
    ],
    "utils.log": [
        "ONP_WARNING",
        "ONP_DEBUG",
        "ONP_ERROR",
        "ONPNotImplementedError",
    ],
}

# Build export map and __all__ from module groups
_EXPORT_MAP: Dict[str, str] = {
    name: module for module, names in _MODULE_EXPORTS.items() for name in names
}

__all__ = list(_EXPORT_MAP.keys())

# Module cache
_MODULE_CACHE: Dict[str, Any] = {}


def __getattr__(name: str) -> Any:
    """Lazy-load symbols from submodules on first access."""
    if name not in _EXPORT_MAP:
        raise AttributeError(
            f"'{__name__}' has no attribute '{name}'. Valid exports: {', '.join(sorted(__all__))}"
        )

    module_path = _EXPORT_MAP[name]
    if module_path not in _MODULE_CACHE:
        _MODULE_CACHE[module_path] = importlib.import_module(
            f"{__name__}.{module_path}"
        )

    return getattr(_MODULE_CACHE[module_path], name)


def __dir__() -> List[str]:
    """Return list of public attributes."""
    return sorted(
        __all__ + [name for name in globals() if name.startswith("__")]
    )
