"""OpenFHE-NumPy: A NumPy-inspired framework for homomorphic encryption operations."""

import importlib
import sys

# --- Version Handling ---
try:
    from .version import __version__
except ImportError:
    from ._version import __version__  # type: ignore

# --- Core Constants ---
from openfhe_numpy._onp_cpp import ArrayEncodingType

ROW_MAJOR = ArrayEncodingType.ROW_MAJOR
COL_MAJOR = ArrayEncodingType.COL_MAJOR

# --- Export Configuration ---
_EXPORTS = {
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
    # Internal bindings (optional for public use)
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
    "utils.matlib": [
        "is_power_of_two",
        "next_power_of_two",
        "check_equality_matrix",
        "check_equality_vector",
        "check_single_equality",
    ],
    "utils.log": ["ONP_WARNING", "ONP_DEBUG", "ONP_ERROR", "ONPNotImplementedError"],
    "utils.constants": ["DataType", "EPSILON", "EPSILON_HIGH", "UnpackType"],
}

# --- Public API Declaration ---
_PUBLIC_MODULES = {"tensor", "operations.matrix_api", "operations.crypto_context"}
__all__ = [name for mod, names in _EXPORTS.items() if mod in _PUBLIC_MODULES for name in names] + [
    "ROW_MAJOR",
    "COL_MAJOR",
]

# --- Internal Symbol Map for Lazy Loading ---
_NAME_TO_MODULE = {name: module for module, names in _EXPORTS.items() for name in names}

# --- Import Style Configuration ---
USE_LAZY_IMPORTS = True


# Support interactive/script use where __package__ may be None
_ROOT_PKG = __package__ or "openfhe_numpy"
_MODULE_CACHE = {}

if not USE_LAZY_IMPORTS:
    for module, names in _EXPORTS.items():
        mod = importlib.import_module(f"{_ROOT_PKG}.{module}")
        for name in names:
            globals()[name] = getattr(mod, name)
else:

    def __getattr__(name):
        """Lazily import symbols on first access."""
        if name not in _NAME_TO_MODULE:
            import difflib

            suggestions = difflib.get_close_matches(name, __all__, n=3)
            hint = f". Did you mean: {suggestions[0]}?" if suggestions else ""
            raise AttributeError(f"'{__name__}' has no attribute '{name}'{hint}")

        module_path = _NAME_TO_MODULE[name]
        if module_path not in _MODULE_CACHE:
            _MODULE_CACHE[module_path] = importlib.import_module(f"{_ROOT_PKG}.{module_path}")
        return getattr(_MODULE_CACHE[module_path], name)

    def __dir__():
        """Return list of attributes for tab-completion."""
        return sorted(__all__)
