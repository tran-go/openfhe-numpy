"""
openfhe_numpy
=============

A NumPy-inspired encrypted linear algebra framework built upon OpenFHE.
"""

from .tensor import ctarray, ptarray
from .constructors import array, ptarray as ptarray_factory
from .algebra import add, multiply, dot, matmul_square, matrix_power
from . import config
from . import utils

__all__ = [
    "ctarray",
    "ptarray",
    "array",
    "ptarray_factory",
    "add",
    "multiply",
    "dot",
    "matmul_square",
    "matrix_power",
    "encoding",
    "config",
    "utils",
]
