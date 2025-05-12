# Standard library imports
import os
from abc import ABC, abstractmethod
from typing import Union, Tuple, Generic, TypeVar

# Third-party imports
import numpy as np

# OpenFHE-related imports
import openfhe
import openfhe_matrix

# OpenFHE NumPy module imports
from openfhe_numpy.config import *
from openfhe_numpy.matlib import *

# from openfhe_numpy.log import *
from openfhe_numpy.log import FP_ERROR, FP_DEBUG
import openfhe_numpy.utils as utils

T = TypeVar("T")


class BaseTensor(ABC, Generic[T]):
    """
    Abstract Class: The common base structure to represent encrypted/encoded arrays.
    """

    @property
    @abstractmethod
    def shape(self) -> Tuple[int, ...]:
        pass

    @property
    @abstractmethod
    def original_shape(self) -> Tuple[int, ...]:
        pass

    @property
    @abstractmethod
    def ndim(self) -> int:
        pass

    @property
    @abstractmethod
    def batch_size(self): ...

    @property
    @abstractmethod
    def order(self): ...

    @property
    @abstractmethod
    def tensor_type(self) -> str: ...

    @abstractmethod
    def clone(self, data: T = None) -> "BaseTensor[T]": ...

    # @abstractmethod
    # def add(self, other): ...

    # a.add(b)

    # add(a, b):
    #     a.add(b)

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
    # def matvec(self, a, b, sumkey, ncols): ...

    # @abstractmethod
    # def matrix_power(self, a, exponent): ...

    # @abstractmethod
    # def add_reduce(self, sumkey, a, ncols): ...

    # @abstractmethod
    # def add_accumulate(self, sumkey, a, ncols): ...

    # @abstractmethod
    # def sub_reduce(self, sumkey, a, ncols): ...

    # @abstractmethod
    # def sub_accumulate(self, sumkey, a, ncols): ...

    # @abstractmethod
    # def transpose(self, a): ...

    # # Special methods
    # def __add__(self, other: "BaseTensor") -> "BaseTensor":
    #     return self.add(other)

    # def __sub__(self, other):
    #     return self.sub(other)

    # def __mul__(self, other):
    #     return self.multiply(self, other)


class FHETensor(BaseTensor, Generic[T]):
    """
    The concrete base class for OpenFHE tensors, storing shape and packing metadata.

    Parameters
    -------
    original_shape  : the original shape of the tensor
    ndim            : the number of dimensions (axes) of the array
    ncols           : the number of columns of the array after padding
    batch_size      : the total number of elements in the array.
    order           : the tensor-to-vector packing scheme

    """

    def __init__(
        self,
        data: T,
        original_shape: Tuple[int, int],
        batch_size: int,
        ncols: int = 0,
        order: int = MatrixOrder.ROW_MAJOR,
    ):
        self._data = data
        self._original_shape = original_shape
        self._batch_size = batch_size
        self._ncols = ncols
        self._ndim = len(original_shape)
        self._order = order

    def info(self):
        return [
            None,
            self._original_shape,
            self._batch_size,
            self._ncols,
            self._order,
        ]

    # -----------------------------------------------------------
    # Properties
    # -----------------------------------------------------------
    @property
    def data(self) -> T:
        return self._data

    @property
    def ncols(self):
        return self._ncols

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def shape(self):
        return (self._batch_size // self._ncols, self._ncols)

    @property
    def original_shape(self):
        return self._original_shape

    @property
    def ndim(self):
        return self._ndim

    @property
    def order(self):
        return self._order

    @property
    def meta(self) -> dict:
        return {
            "shape": self._shape,
            "batch_size": self._batch_size,
            "ncols": self._ncols,
            "order": self._order,
        }

    # -----------------------------------------------------------
    # Protected Functions
    # -----------------------------------------------------------
    # Todo [trango]: add later
    def _convert(self, data):
        return data

    # -----------------------------------------------------------
    # Public Functions
    # -----------------------------------------------------------

    def set_batch_size(self, value):
        self._batch_size = value

    def set_ncols(self, value):
        self._ncols = value

    def serialize_meta(self) -> dict:
        return {
            "tensor_type": self.tensor_type,
            "original_shape": self.original_shape,
            "batch_size": self.batch_size,
            "ncols": self.ncols,
            "order": self.order,
        }

    # -----------------------------------------------------------
    # Operators
    # -----------------------------------------------------------

    def __add__(self, other):
        other = self._convert(other)
        if self.shape != other.shape:
            FP_ERROR("Shape does not match for element-wise addition")

        crypto_context = self.data.GetCryptoContext()
        ct_sum = crypto_context.EvalAdd(self.data, other.data)
        return self.clone(ct_sum)

    def __sub__(self, other):
        other = self._convert(other)
        if self.shape != other.shape:
            FP_ERROR("Shape does not match for element-wise subtraction")

        crypto_context = self.data.GetCryptoContext()
        ct_sum = crypto_context.EvalSub(self.data, other.data)
        return self.clone(ct_sum)

    def __mul__(self, other):
        other = self._convert(other)
        if other.shape != other.shape:
            FP_ERROR("Shape does not match for element-wise addition")

        crypto_context = self.data.GetCryptoContext()
        ct_sum = crypto_context.EvalMult(self.data, other.data)
        return self.clone(ct_sum)


class CTArray(FHETensor[openfhe.Ciphertext]):
    """
    The concrete base class for OpenFHE Ciphertext Matrix, storing shape and packing metadata.

    Parameters
    -------
    data            : the encryption of the encoded matrix
    original_shape  : the original shape of the tensor
    ndim            : the number of dimensions (axes) of the array
    ncols           : the number of columns of the array after padding
    batch_size      : the total number of elements in the array.
    order           : the tensor-to-vector packing scheme
    """

    # Constructors
    def __init__(
        self,
        data: openfhe.Ciphertext,
        original_shape: Tuple[int, int],
        batch_size: int,
        ncols: int = 1,
        order: int = MatrixOrder.ROW_MAJOR,
    ):
        super().__init__(data, original_shape, batch_size, ncols, order)

    def clone(self, data=None):
        return CTArray(
            data or self.data,
            self.original_shape,
            self.batch_size,
            self.ncols,
            self.order,
        )

    def __repr__(self):
        return (
            f"CTArray(original_shape={self.original_shape}, ndim={self.ndim}, "
            f"size={self.batch_size}, ncols={self.ncols}, order={self.order})"
        )

    # -----------------------------------------------------------
    # Properties
    # -----------------------------------------------------------

    ###
    @property
    def tensor_type(self) -> str:
        return "CTArray"

    # -----------------------------------------------------------
    # Protected Functions
    # -----------------------------------------------------------

    ###
    def _convert(self, data):
        return data

    # -----------------------------------------------------------
    # Operator Functions
    # -----------------------------------------------------------

    ###

    # -----------------------------------------------------------
    # Public Functions
    # -----------------------------------------------------------

    def decrypt(self, secret_key, is_format=True, precision=None):
        cc = self.data.GetCryptoContext()
        result = cc.Decrypt(self.data, secret_key)
        result.SetLength(self.batch_size)
        result = result.GetRealPackedValue()
        print(self.original_shape, self.shape)
        if is_format:
            result = utils.format(result, self.ndim, self.original_shape, self.shape)
        return result

    def serialize(self) -> dict:
        import io

        stream = io.BytesIO()
        if not openfhe.Serialize(self.data, stream):
            raise RuntimeError("Failed to serialize Ciphertext")
        return {
            "type": self.tensor_type,
            "original_shape": self.original_shape,
            "batch_size": self.batch_size,
            "ncols": self.ncols,
            "order": self.order,
            "ciphertext": stream.getvalue().hex(),
        }

    @classmethod
    def deserialize(cls, obj: dict) -> "CTArray":
        import io

        stream = io.BytesIO(bytes.fromhex(obj["ciphertext"]))
        ciphetext = openfhe.Ciphertext()
        openfhe.Deserialize(ciphetext, stream)
        return cls(
            ciphetext, tuple(obj["original_shape"]), obj["batch_size"], obj["ncols"], obj["order"]
        )


class PTArray(FHETensor[openfhe.Plaintext]):
    """
    The concrete base class for OpenFHE Plaintext Matrix, storing shape and packing metadata.

    Parameters
    -------
    data            : the encoded matrix
    original_shape  : the original shape of the tensor
    ndim            : the number of dimensions (axes) of the array
    ncols           : the number of columns of the array after padding
    batch_size      : the total number of elements in the array.
    order           : the tensor-to-vector packing scheme
    """

    def __init__(
        self,
        data: openfhe.Plaintext,
        original_shape: Tuple[int, int],
        batch_size: int,
        ncols: int = 1,
        order: int = MatrixOrder.ROW_MAJOR,
    ):
        super().__init__(data, original_shape, batch_size, ncols, order)

    def __repr__(self):
        return (
            f"PTArray(original_shape={self.original_shape}, ndim={self.ndim}, "
            f"size={self.batch_size}, ncols={self.ncols}, order={self.order})"
        )

    def clone(self, data=None):
        return PTArray(
            data or self.data,
            self.original_shape,
            self.batch_size,
            self.ncols,
            self.order,
        )

    @property
    def tensor_type(self) -> str:
        return "PTArray"

    def decrypt(self, *args, **kwargs):
        raise NotImplementedError("Decrypt not implemented for PTArray")
