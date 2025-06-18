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
from typing import Any, Dict, List

# Version handling
try:
    from .version import __version__
except ImportError:
    try:
        from ._version import __version__
    except ImportError:
        __version__ = "0.0.1"

from openfhe_numpy._onp_cpp import ArrayEncodingType

# Direct imports for common constants
ROW_MAJOR = ArrayEncodingType.ROW_MAJOR
COL_MAJOR = ArrayEncodingType.COL_MAJOR


# === OPTION: Choose Import Style ===
USE_LAZY_IMPORTS = True  # Set to True for lazy loading, False for eager/explicit

# --- API Declarations ---
_MODULE_EXPORTS = {
    "tensor": ["BaseTensor", "FHETensor", "PTArray", "CTArray", "BlockFHETensor", "BlockCTArray", "array"],
    "operations.matrix_api": ["add", "multiply", "dot", "matmul", "transpose", "power", "cumsum", "cumreduce", "sum"],
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
        "ArrayEncodingType",
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
    "utils.matlib": ["is_power_of_two", "next_power_of_two", "check_equality_matrix", "check_equality_vector"],
    "utils.log": ["ONP_WARNING", "ONP_DEBUG", "ONP_ERROR", "ONPNotImplementedError"],
    "utils.constants": ["DataType", "EPSILON", "EPSILON_HIGH", "UnpackType"],
}
_EXPORT_MAP: Dict[str, str] = {name: module for module, names in _MODULE_EXPORTS.items() for name in names}
__all__ = list(_EXPORT_MAP.keys()) + ["ROW_MAJOR", "COL_MAJOR"]

# === EXPLICIT IMPORTS  ===
if not USE_LAZY_IMPORTS:
    from openfhe_numpy.tensor import BaseTensor, FHETensor, PTArray, CTArray, BlockFHETensor, BlockCTArray, array
    from openfhe_numpy.operations.matrix_api import add, multiply, dot, matmul, transpose, power, cumsum, cumreduce, sum
    from openfhe_numpy.operations.crypto_context import (
        sum_row_keys,
        sum_col_keys,
        gen_rotation_keys,
        gen_lintrans_keys,
        gen_transpose_keys,
        gen_square_matmult_key,
        gen_accumulate_rows_key,
        gen_accumulate_cols_key,
    )
    from openfhe_numpy._onp_cpp import (
        LinTransType,
        MatVecEncoding,
        ArrayEncodingType,
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
    from openfhe_numpy.utils.matlib import (
        is_power_of_two,
        next_power_of_two,
        check_equality_matrix,
        check_equality_vector,
    )
    from openfhe_numpy.utils.constants import DataType, EPSILON, EPSILON_HIGH, UnpackType
    from openfhe_numpy.utils.log import ONP_WARNING, ONP_DEBUG, ONP_ERROR, ONPNotImplementedError

# === LAZY IMPORTS ===
if USE_LAZY_IMPORTS:
    _MODULE_CACHE: Dict[str, Any] = {}

    def __getattr__(name: str) -> Any:
        """Lazy-load symbols from submodules on first access.

        Example:
            >>> from openfhe_numpy import add
            >>> add  # triggers import of operations.matrix_api.add
        """
        if name not in _EXPORT_MAP:
            # Help users find what they misspelled
            import difflib

            suggestions = difflib.get_close_matches(name, __all__)
            hint = f" Did you mean: {suggestions}?" if suggestions else ""
            raise AttributeError(
                f"'{__name__}' has no attribute '{name}'. Valid exports: {', '.join(sorted(__all__))}.{hint}"
            )
        module_path = _EXPORT_MAP[name]
        if module_path not in _MODULE_CACHE:
            try:
                _MODULE_CACHE[module_path] = importlib.import_module(f"{__name__}.{module_path}")
            except ImportError as e:
                raise ImportError(f"Could not import submodule '{module_path}' for '{name}': {e}") from e
        try:
            return getattr(_MODULE_CACHE[module_path], name)
        except AttributeError:
            raise AttributeError(f"Module '{module_path}' does not export '{name}'")

    def __dir__() -> List[str]:
        """Return list of public attributes for tab-completion, etc."""
        # Combine lazy and already-imported symbols for best developer UX
        return sorted(__all__ + [name for name in globals() if name.startswith("__")])
