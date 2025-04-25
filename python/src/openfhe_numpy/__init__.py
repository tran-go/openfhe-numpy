"""
openfhe_numpy
=============

A NumPy-inspired encrypted linear algebra framework built upon OpenFHE.
"""

from .tensor import ctarray, ptarray
from .constructors import array
from .algebra import (
    add,
    multiply,
    dot,
    matvec,
    matmul_square,
    matrix_power,
    transpose,
    add_reduce,
)
from .utils import pack_vec_row_wise
from .crypto_context import (
    gen_sum_row_keys,
    gen_sum_col_keys,
    gen_rotation_keys,
    gen_lintrans_keys,
    gen_transpose_keys,
    gen_square_matrix_product,
)
from .matlib import *
from . import config
from . import utils

__all__ = [
    "ctarray",
    "ptarray",
    "array",
    "add",
    "multiply",
    "dot",
    "matmul_square",
    "matrix_power",
    "config",
    "utils",
]
