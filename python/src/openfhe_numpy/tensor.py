import os
import numpy as np
from typing import Union, Tuple
from abc import ABC, abstractmethod

# Import OpenFHE-related libraries
import openfhe
import openfhe_matrix

# Import OpenFHE NumPy modules
from openfhe_numpy.config import *
from openfhe_numpy.matlib import *
import openfhe_numpy.utils as utils
from openfhe_numpy.log import *


class BaseTensor(ABC):
    """
    Abstract Class: The common base structure to represent encrypted/encoded arrays.
    """

    @property
    @abstractmethod
    def shape(self): ...

    @property
    @abstractmethod
    def original_shape(self): ...

    @property
    @abstractmethod
    def ndim(self): ...

    @property
    @abstractmethod
    def batch_size(self): ...

    @property
    @abstractmethod
    def order(self): ...

    @property
    @abstractmethod
    def tensor_type(self) -> str: ...

    # @abstractmethod
    # def clone(self): ...

    # @abstractmethod
    # def add(self, other): ...

    # @abstractmethod
    # def sum(self): ...

    # @abstractmethod
    # def square_matmul(self, a, b): ...

    # @abstractmethod
    # def multiply(self, a, b): ...

    # @abstractmethod
    # def sub(self, a, b): ...

    # @abstractmethod
    # def dot(self, a, b): ...

    # @abstractmethod
    # def matvec(self, a, b, sumkey, rowsize): ...

    # @abstractmethod
    # def matrix_power(self, a, exponent): ...

    # @abstractmethod
    # def add_reduce(self, sumkey, a, rowsize): ...

    # @abstractmethod
    # def add_accumulate(self, sumkey, a, rowsize): ...

    # @abstractmethod
    # def sub_reduce(self, sumkey, a, rowsize): ...

    # @abstractmethod
    # def sub_accumulate(self, sumkey, a, rowsize): ...

    # @abstractmethod
    # def transpose(self, a): ...

    # # Special methods
    # def __add__(self, other: "BaseTensor") -> "BaseTensor":
    #     return self.add(other)

    # def __sub__(self, other):
    #     return self.sub(other)

    # def __mul__(self, other):
    #     return self.multiply(self, other)


class FHETensor(BaseTensor):
    """
    Concrete base class for OpenFHE tensors, storing shape and packing metadata.
    """

    def __init__(
        self,
        original_shape: Tuple[int, int],
        batch_size: int,
        rowsize: int = 0,
        order: int = MatrixOrder.ROW_MAJOR,
    ):
        self._original_shape = original_shape
        self._batch_size = batch_size
        self._rowsize = rowsize
        self._ndim = len(original_shape)
        self._order = order

    def info(self):
        return [
            None,
            self.original_shape,
            self.batch_size,
            self.rowsize,
            self.order,
        ]

    @property
    def rowsize(self):
        return self._rowsize

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def shape(self):
        return (self._batch_size // self._rowsize, self._rowsize)

    @property
    def original_shape(self):
        return self._original_shape

    @property
    def ndim(self):
        return self._ndim

    @property
    def order(self):
        return self._order

    def set_batch_size(self, value):
        self._batch_size = value

    def set_rowsize(self, value):
        self._rowsize = value


class CTArray(FHETensor):
    """Ciphertext array representation with shape and encoding metadata."""

    def __init__(
        self,
        data: openfhe.Ciphertext,
        original_shape: Tuple[int, int],
        batch_size: int,
        rowsize: int = 1,
        order: int = MatrixOrder.ROW_MAJOR,
    ):
        super().__init__(original_shape, batch_size, rowsize, order)
        self._data = data

    def __repr__(self):
        return (
            f"CTArray(original_shape={self.original_shape}, ndim={self.ndim}, "
            f"size={self.batch_size}, rowsize={self.rowsize}, order={self.order})"
        )

    def decrypt(self, sk, isFormat=True, precision=None):
        cc = self.data.GetCryptoContext()
        result = cc.Decrypt(self._data, sk)
        result.SetLength(self.batch_size)
        result = result.GetRealPackedValue()
        print(self.original_shape, self.shape)
        if isFormat:
            result = utils.format(result, self.ndim, self.original_shape, self.shape)
        return result

    def clone(self):
        return CTArray(
            self._data,
            self.original_shape,
            self.ndim,
            self.batch_size,
            self.rowsize,
            self.order,
        )

    @property
    def tensor_type(self) -> str:
        return "CTArray"

    @property
    def data(self) -> openfhe.Ciphertext:
        return self._data


class PTArray(FHETensor):
    """Plaintext array representation with shape and encoding metadata."""

    def __init__(
        self,
        data: openfhe.Plaintext,
        original_shape: Tuple[int, int],
        batch_size: int,
        rowsize: int = 1,
        order: int = MatrixOrder.ROW_MAJOR,
    ):
        super().__init__(original_shape, batch_size, rowsize, order)
        self._data = data

    def __repr__(self):
        return (
            f"PTArray(original_shape={self.original_shape}, ndim={self.ndim}, "
            f"size={self.batch_size}, rowsize={self.rowsize}, order={self.order})"
        )

    def clone(self):
        return PTArray(
            self._data,
            self.original_shape,
            self.ndim,
            self.batch_size,
            self.rowsize,
            self.order,
        )

    @property
    def tensor_type(self) -> str:
        return "PTArray"

    @property
    def data(self) -> openfhe.Plaintext:
        return self._data
