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
        matrix_enconding: int = MatrixEncoding.ROW_MAJOR,
    ):
        self.original_shape = original_shape
        self.ndim = ndim
        self.ncols = ncols
        self.nrows = size // ncols
        self.batch_size = size
        self.matrix_enconding = matrix_enconding

    def info(self):
        return [
            None,
            self.original_shape,
            self.ndim,
            self.batch_size,
            self.ncols,
            self.matrix_enconding,
        ]

    def __repr__(self):
        return f"{self.__class__.__name__}(shape=({self.nrows}, {self.ncols}), original={self.original_shape}, encoding={self.matrix_enconding})"

    @property
    def shape(self):
        return (self.nrows, self.ncols)


class PTarray(BaseTensor):
    """Plaintext array representation that includes shape and encoding metadata."""

    def __init__(
        self,
        data: openfhe.Plaintext,
        original_shape: Tuple[int, int],
        ndim: int,
        size: int,
        ncols: int = 1,
        matrix_enconding: int = MatrixEncoding.ROW_MAJOR,
    ):
        super().__init__(original_shape, ndim, size, ncols, matrix_enconding)
        self.data = data

    def copy(self, is_deep_copy: bool = True):
        return PTarray(
            self.data,
            self.original_shape,
            self.ndim,
            self.batch_size,
            self.ncols,
            self.matrix_enconding,
        )


class CTarray(BaseTensor):
    """Ciphertext array representation with shape and encoding metadata."""

    def __init__(
        self,
        data: openfhe.Ciphertext | None,
        original_shape: Tuple[int, int],
        ndim: int,
        size: int,
        ncols: int = 1,
        matrix_enconding: int = MatrixEncoding.ROW_MAJOR,
    ):
        super().__init__(original_shape, ndim, size, ncols, matrix_enconding)
        self.data = data

    def decrypt(self, cc, sk, isFormat=True, precision=None):
        result = cc.Decrypt(self.data, sk)
        result.SetLength(self.batch_size)
        result = result.GetRealPackedValue()
        if isFormat:
            result = utils.format(result, self.ndim, self.original_shape, self.shape)
        return result

    def copy(self):
        return CTarray(
            self.data,
            self.original_shape,
            self.ndim,
            self.batch_size,
            self.ncols,
            self.matrix_enconding,
        )
