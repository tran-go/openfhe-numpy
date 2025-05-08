import os
import numpy as np
from typing import Union, Tuple

# Import OpenFHE-related libraries
import openfhe
import openfhe_matrix

# Import OpenFHE NumPy modules
from openfhe_numpy.config import *
from openfhe_numpy.matlib import *
import openfhe_numpy.utils as utils
from openfhe_numpy.log import *


class BaseCrypto:
    """
    Abstract Class: The common base structure to represent encrypted/encoded arrays.
    """

    def add(self, other):
        raise NotImplementedError

    def sum(self, axis=None):
        raise NotImplementedError

    def square_matmul(self, other):
        raise NotImplementedError

    def multiply(self, other):
        raise NotImplementedError

    def sub(self, other):
        raise NotImplementedError

    def dot(self, other):
        raise NotImplementedError

    def matvec(self, sumkey, other, rowsize):
        raise NotImplementedError

    def matrix_power(self, exponent):
        raise NotImplementedError

    def add_reduce(self, sumkey):
        raise NotImplementedError

    def add_accumulate(self, sumkey):
        raise NotImplementedError

    def sub_reduce(self, sumkey, a):
        raise NotImplementedError

    def sub_accumulate(self, sumkey, a):
        raise NotImplementedError

    def transpose(self):
        raise NotImplementedError


class FHETensor:
    """
    Base Tensor: The common base structure to represent encrypted/encoded arrays.

    Parameters
    -------
    original_shape: the original shape of the tensor
    ndim : the number of dimensions (axes) of the array
    rowsize: the number of columns of the array
    nrows: the number of rows of the array
    size : the total number of elements in the array.
    order: the tensor-to-vector packing scheme

    """

    def __init__(
        self,
        data: Union[openfhe.Ciphertext, openfhe.Plaintext],
        original_shape: Tuple[int, int],
        size: int,
        rowsize: int = 0,
        order: int = MatrixOrder.ROW_MAJOR,
    ):
        self.__original_shape = original_shape
        self.__batch_size = size
        self.__order = order
        self.__data = data
        if data is None:
            self.__is_encrypted = None
        elif isinstance(data, openfhe.Ciphertext):
            self.__is_encrypted = True
        elif isinstance(data, openfhe.Plaintext):
            self.__is_encrypted = False
        else:
            raise NotImplementedError

    ###
    # Getter/Setter Operations
    ###
    @property
    def shape(self):
        return (self.__nrows, self.__batch_size // self.__nrows)

    @property
    def original_shape(self):
        return self.__original_shape

    @property
    def ndim(self):
        return len(self.__original_shape)

    @property
    def data(self):
        return self.__data

    @property
    def info(self):
        return [
            self.__data,
            self.__original_shape,
            self.__batch_size,
            self.__rowsize,
            self.__order,
            self.__is_encrypted,
        ]

    ###
    # Dunder methods
    ###

    def __repr__(self):
        return f"Tensor(original_shape={self.__original_shape}, ndim=\n{self.ndim}), size=\n{self.__batch_size}), nrows=\n{self.__nrows}), order=\n{self.__order})"

    def __add__(self, other):
        # if isinstance(other, Tensor):
        #     return Tensor(self.__data + other.data)
        # return BaseTensor(self.__data + other)
        raise NotImplementedError

    def __mul__(self, other):
        # if isinstance(other, Tensor):
        #     return Tensor(self.__data * other.data)
        # return BaseTensor(self.__data * other)
        raise NotImplementedError

    def sum(self, axis=None):
        # return Tensor(np.sum(self.__data, axis=axis))
        raise NotImplementedError

    def transpose(self, *axes):
        # return BaseTensor(self.__data.transpose(*axes))
        raise NotImplementedError

    ###
    # Get infomation methods
    ###

    def clone(self):
        return FHETensor(
            self.__data, self.__original_shape, self.__batch_size, self.__rowsize, self.__order
        )

    def decrypt(self, cc, sk, isFormat=True, precision=None):
        if not self.__is_encrypted:
            FP_NOTIFY(f"Tensor is not encrpted: tensor type is {type(self.data)}")
        result = cc.Decrypt(self.__data, sk)
        result.SetLength(self.size)
        result = result.GetRealPackedValue()
        if isFormat:
            result = utils.format(result, self.ndim, self.__original_shape, self.shape)
        return result
