import os
import numpy as np
from typing import Tuple

# Import OpenFHE-related libraries
import openfhe
import openfhe_matrix

# Import OpenFHE NumPy modules
from openfhe_numpy.config import *
from openfhe_numpy.matlib import *
import openfhe_numpy.utils as utils


class BaseTensor:
    """Common base structure for encrypted/encoded arrays."""

    def __init__(
        self,
        original_shape: Tuple[int, int],
        ndim: int,
        size: int,
        ncols: int = 1,
        encoding_order: int = MatrixEncoding.ROW_MAJOR,
    ):
        self.original_shape = original_shape
        self.ndim = ndim
        self.ncols = ncols
        self.nrows = size // ncols
        self.batch_size = size
        self.encoding_order = encoding_order

    def info(self):
        return [
            None,
            self.original_shape,
            self.ndim,
            self.batch_size,
            self.ncols,
            self.encoding_order,
        ]

    def __repr__(self):
        return f"{self.__class__.__name__}(shape=({self.nrows}, {self.ncols}), original={self.original_shape}, encoding={self.encoding_order})"

    @property
    def shape(self):
        return (self.nrows, self.ncols)


class ptarray(BaseTensor):
    """Plaintext array representation that includes shape and encoding metadata."""

    def __init__(
        self,
        data: openfhe.Plaintext,
        original_shape: Tuple[int, int],
        ndim: int,
        size: int,
        ncols: int = 1,
        encoding_order: int = MatrixEncoding.ROW_MAJOR,
    ):
        super().__init__(original_shape, ndim, size, ncols, encoding_order)
        self.data = data

    def copy(self, is_deep_copy: bool = True):
        return ptarray(
            self.data,
            self.original_shape,
            self.ndim,
            self.batch_size,
            self.ncols,
            self.encoding_order,
        )


class ctarray(BaseTensor):
    """Ciphertext array representation with shape and encoding metadata."""

    def __init__(
        self,
        data: openfhe.Ciphertext,
        original_shape: Tuple[int, int],
        ndim: int,
        size: int,
        ncols: int = 1,
        encoding_order: int = MatrixEncoding.ROW_MAJOR,
    ):
        super().__init__(original_shape, ndim, size, ncols, encoding_order)
        self.data = data

    def decrypt(self, cc, sk, isFormat=True, precision=None):
        result = cc.Decrypt(self.data, sk)
        result.SetLength(self.batch_size)
        # if precision is not None:
        #     result.GetFormattedValues(precision)
        result = result.GetRealPackedValue()
        if isFormat:
            result = utils.format(result, self.ndim, self.original_shape, self.shape)
        return result

    def copy(self):
        return ctarray(
            self.data,
            self.original_shape,
            self.ndim,
            self.batch_size,
            self.ncols,
            self.encoding_order,
        )
