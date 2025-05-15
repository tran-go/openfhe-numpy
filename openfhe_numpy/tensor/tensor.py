# -----------------------------------------------------------
# Standard Library Imports
# -----------------------------------------------------------
from abc import ABC, abstractmethod
from typing import (
    Any,
    Dict,
    Generic,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
)
import io
import sys
import logging


# -----------------------------------------------------------
# Third-Party Imports
# -----------------------------------------------------------
import numpy as np
import openfhe
from openfhe import *

from .. import _openfhe_numpy  # Import from parent package
from openfhe_numpy.utils.log import ONP_ERROR
from openfhe_numpy.utils.utils import is_power_of_two, next_power_of_two, MatrixOrder

# -----------------------------------------------------------
# Ultilities Imports
# -----------------------------------------------------------
T = TypeVar("T")


# -----------------------------------------------------------
# BaseTensor - Abstract Interface
# -----------------------------------------------------------
class BaseTensor(ABC, Generic[T]):
    @property
    @abstractmethod
    def shape(self) -> Tuple[int, ...]: ...

    @property
    @abstractmethod
    def original_shape(self) -> Tuple[int, ...]: ...

    @property
    @abstractmethod
    def ndim(self) -> int: ...

    @property
    @abstractmethod
    def batch_size(self) -> int: ...

    @property
    @abstractmethod
    def ncols(self) -> int: ...

    @property
    @abstractmethod
    def order(self) -> int: ...

    @property
    @abstractmethod
    def dtype(self) -> str: ...

    @property
    @abstractmethod
    def meta(self) -> dict: ...

    @abstractmethod
    def clone(self, data: T = None) -> "BaseTensor[T]": ...

    @abstractmethod
    def decrypt(self, *args, **kwargs): ...

    @abstractmethod
    def _add(self, other: "BaseTensor[T]") -> "BaseTensor[T]": ...

    @abstractmethod
    def _sub(self, other: "BaseTensor[T]") -> "BaseTensor[T]": ...

    @abstractmethod
    def _multiply(self, other: "BaseTensor[T]") -> "BaseTensor[T]": ...

    @abstractmethod
    def _matmul(self, other: "BaseTensor[T]") -> "BaseTensor[T]": ...

    @abstractmethod
    def _pow(self, exp: int) -> "BaseTensor[T]": ...

    @abstractmethod
    def _sum(self, axis: int = 0) -> "BaseTensor[T]": ...

    @abstractmethod
    def _reduce(self, axis: int = 0) -> "BaseTensor[T]": ...

    @abstractmethod
    def _transpose(self) -> "BaseTensor[T]": ...

    @abstractmethod
    def _trace(self) -> "BaseTensor[T]": ...


# -----------------------------------------------------------
# FHETensor - Generic Tensor with Metadata
# -----------------------------------------------------------
class FHETensor(BaseTensor[T], Generic[T]):
    """
    Concrete base class for tensors in FHE computation.

    Parameters
    ----------
    data : T
        Underlying encrypted or encoded data.
    original_shape : Tuple[int, int]
        Shape before any padding.
    batch_size : int
        Total number of packed slots.
    ncols : int
        Number of logical columns after padding.
    order : int
        Packing order, e.g., row- or column-major.
    """

    __slots__ = ("_data", "_original_shape", "_batch_size", "_ncols", "_ndim", "_order")

    def __init__(
        self,
        data: T,
        original_shape: Tuple[int, int],
        batch_size: int,
        ncols: int = 1,
        order: int = MatrixOrder.ROW_MAJOR,
    ):
        self._data = data
        self._original_shape = original_shape
        self._batch_size = batch_size
        self._ncols = ncols if is_power_of_two(ncols) else next_power_of_two(ncols)
        self._ndim = len(original_shape)
        self._order = order
        self._dtype = self.__class__.__name__  # e.g., "CTArray", "BlockCTArray"

    ###
    ### Properties
    ###

    @property
    def dtype(self):
        return self._dtype

    @property
    def data(self) -> T:
        """Underlying encrypted/plaintext payload."""
        return self._data

    @property
    def shape(self) -> Tuple[int, int]:
        """Logical 2-D shape after packing."""
        rows = self._batch_size // self._ncols
        return (rows, self._ncols)

    @property
    def original_shape(self) -> Tuple[int, int]:
        """Shape before any padding was applied."""
        return self._original_shape

    @property
    def ndim(self) -> int:
        """Dimensionality of the original tensor."""
        return self._ndim

    @property
    def batch_size(self) -> int:
        """Total number of packed slots."""
        return self._batch_size

    @property
    def ncols(self) -> int:
        """Number of columns in the packed representation."""
        return self._ncols

    @property
    def order(self) -> int:
        """Packing order constant (row- or column-major)."""
        return self._order

    @property
    def meta(self) -> Dict[str, Any]:
        """Metadata dict for serialization or inspection."""
        return {
            "type": self.dtype,
            "shape": self.shape,
            "original_shape": self.original_shape,
            "ncols": self.ncols,
            "batch_size": self.batch_size,
            "order": self.order,
        }

    @property
    def info(self) -> Tuple:
        """
        Tuple of shape and encoding metadata.

        Returns
        -------
        Tuple
            Contains [None, original_shape, batch_size, ncols, order]
        """
        return [None, self.original_shape, self.batch_size, self.ncols, self.order]

    ###
    ### Setter
    ###

    def set_batch_size(self, value: int):
        """Set batch size with validation."""
        if not isinstance(value, int):
            raise TypeError(f"Batch size must be integer, got {type(value)}")
        if value <= 0:
            raise ValueError(f"Batch size must be positive, got {value}")
        self._batch_size = value

    def set_ncols(self, value: int):
        """
        Set the number of columns in the packed representation.

        Parameters
        ----------
        value : int
            New number of columns value. Should be a power of two.
        """
        if not isinstance(value, int):
            raise TypeError(f"Batch size must be integer, got {type(value)}")
        if value <= 0:
            raise ValueError(f"Batch size must be positive, got {value}")
        self._ncols = value

    def clone(self, data: Optional[T] = None) -> "BaseTensor[T]":
        """
        Copy the tensor, optionally replacing the data payload.
        """
        return type(self)(
            data or self.data,
            self.original_shape,
            self.batch_size,
            self.ncols,
            self.order,
        )

    def __eq__(self, other) -> bool:
        """
        Structural comparison of shape and layout.

        Parameters
        ----------
        other : object
            Object to compare with

        Returns
        -------
        bool
            True if other is the same type and has identical metadata
        """
        return (
            isinstance(other, type(self))
            and self.original_shape == other.original_shape
            and self.batch_size == other.batch_size
            and self.ncols == other.ncols
            and self.order == other.order
        )

    ###
    ### Operators
    ###

    def __add__(self, other):
        if isinstance(other, FHETensor):
            if self.dtype == "CTArray":
                return self._add(other)
            elif self.dtype == "PTArray":
                return other._add(self)
        elif isinstance(other, (int, float)):
            # Handle scalar addition
            if self.dtype == "CTArray":
                return self._add_scalar(other)
        elif isinstance(other, np.ndarray):
            # Handle numpy array addition
            if self.dtype == "CTArray":
                return self._add_list(other)

        return NotImplemented

    def __sub__(self, other):
        if self.dtype == "CTArray":
            return self._sub(other)
        elif self.dtype == "PTArray":
            return other._sub(self)
        else:
            ONP_ERROR("NOT_IMPLEMENTED")

    def __mul__(self, other):
        if self.dtype == "CTArray":
            return self._multiply(other)
        elif self.dtype == "PTArray":
            return other._multiply(self)
        else:
            ONP_ERROR("NOT_IMPLEMENTED")

    def __matmul__(self, other):
        if self.dtype == "CTArray":
            return self._matmul(other)
        elif self.dtype == "PTArray":
            return other._matmul(self)
        else:
            ONP_ERROR("NOT_IMPLEMENTED")

    def __pow__(self, exp):
        if self.dtype == "CTArray":
            return self._pow(exp)
        else:
            ONP_ERROR("NOT_IMPLEMENTED")

    def __del__(self):
        """Release any critical resources when object is destroyed."""
        # The OpenFHE library might have resource cleanup needs
        pass

    def __getitem__(self, key):
        """
        Extract a slice from the encrypted tensor.

        Parameters
        ----------
        key : int, tuple, or slice
            Indices to extract

        Returns
        -------
        CTArray
        """
        pass

    def sum(self, axis=0) -> "CTArray":
        if axis < 0 or axis >= self.ndim:
            raise ValueError(f"Invalid axis {axis} for tensor with {self.ndim} dimensions.")
        if self.dtype == "CTArray":
            return self._sum(axis)
        raise NotImplementedError()

    def reduce(self, axis=0) -> "CTArray":
        if self.dtype == "CTArray":
            return self._reduce(axis)
        else:
            ONP_ERROR("NOT_IMPLEMENTED")

    def transpose(self) -> "FHETensor":
        try:
            return self._transpose()
        except NotImplementedError:
            ONP_ERROR("Transpose not implemented for {self.__class__.__name__}")

    def ensure_compatible_packing(self, other):
        """
        Ensure tensors have compatible packing for operations.

        Returns a version of 'other' with matching packing order.
        """
        if not isinstance(other, FHETensor):
            return other

        if self.order == other.order:
            return other

        return other.convert_packing_order(self.order)

    def convert_packing_order(self, target_order):
        """
        Convert tensor to a different packing order.

        Parameters
        ----------
        target_order : int
            Desired packing order (ROW_MAJOR or COL_MAJOR)

        Returns
        -------
        FHETensor
            New tensor with converted packing order
        """
        if self.order == target_order:
            return self.clone()

        # Perform conversion
        if self.dtype == "CTArray":
            # For ciphertexts, use transpose operation
            transposed = self._transpose()
            # Update order flag
            transposed._order = target_order
            return transposed
        else:
            pass


# -----------------------------------------------------------
# CTArray - Ciphertext Tensor
# -----------------------------------------------------------
class CTArray(FHETensor[openfhe.Ciphertext]):
    """
    Encrypted tensor class for OpenFHE ciphertexts.

    This class represents encrypted tensors that can be manipulated
    using homomorphic operations. It supports standard operations
    like addition, multiplication, and matrix operations.

    Examples
    --------
    >>> # Create and encrypt a matrix
    >>> cc, keys = gen_crypto_context(4)
    >>> matrix = np.array([[1, 2], [3, 4]])
    >>> encrypted = onp.array(cc, matrix, slots, keys.publicKey)
    >>> result = encrypted + encrypted  # Homomorphic addition
    >>> decrypted = result.decrypt(keys.secretKey)
    """

    # @property
    # def dtype(self) -> Literal["CTArray"]:
    #     return "CTArray"

    def decrypt(self, secret_key: openfhe.PrivateKey) -> np.ndarray:
        """Decrypt ciphertext using given secret key."""
        if secret_key is None:
            ONP_ERROR("Secret Key is missing!!!")
        cc = self.data.GetCryptoContext()
        plaintext = cc.Decrypt(self.data, secret_key)
        plaintext.SetLength(self.batch_size)
        return plaintext.GetRealPackedValue()

    def serialize(self) -> dict:
        """Serialize ciphertext and metadata to dictionary."""
        stream = io.BytesIO()
        if not openfhe.Serialize(self.data, stream):
            raise RuntimeError("Failed to serialize Ciphertext")
        return {
            "type": self.type,
            "original_shape": self.original_shape,
            "batch_size": self.batch_size,
            "ncols": self.ncols,
            "order": self.order,
            "ciphertext": stream.getvalue().hex(),
        }

    @classmethod
    def deserialize(cls, obj: dict) -> "CTArray":
        """Deserialize from dictionary back to CTArray."""
        stream = io.BytesIO(bytes.fromhex(obj["ciphertext"]))
        ciphertext = openfhe.Ciphertext()
        openfhe.Deserialize(ciphertext, stream)
        return cls(
            ciphertext, tuple(obj["original_shape"]), obj["batch_size"], obj["ncols"], obj["order"]
        )

    def __repr__(self) -> str:
        return f"CTArray(meta={self.meta})"

    def _add(self, other) -> "CTArray":
        """Element-wise addition with packing compatibility."""
        other = self.ensure_compatible_packing(other)

        if self.shape != other.shape:
            ONP_ERROR("Shape does not match for element-wise addition")
        crypto_context = self.data.GetCryptoContext()
        ciphertext = crypto_context.EvalAdd(self.data, other.data)
        return self.clone(ciphertext)

    def _add_scalar(self, scalar) -> "CTArray":
        """Add scalar value to all elements."""
        crypto_context = self.data.GetCryptoContext()

        # Create plaintext encoding of scalar
        plaintext = crypto_context.MakeCKKSPackedPlaintext([scalar] * self.batch_size)

        # Perform addition
        result = crypto_context.EvalAdd(self.data, plaintext)
        return self.clone(result)

    # Add these methods for consistency with _add_scalar:

    def _sub_scalar(self, scalar) -> "CTArray":
        """Subtract scalar value from all elements."""
        # Implementation

    def _sub(self, other) -> "CTArray":
        if self.shape != other.shape:
            ONP_ERROR("Shape does not match for element-wise subtraction")
        crypto_context = self.data.GetCryptoContext()
        ciphertext = crypto_context.EvalSub(self.data, other.data)
        return self.clone(ciphertext)

    def _multiply(self, other) -> "CTArray":
        if self.shape != other.shape:
            ONP_ERROR(f"Shape mismatch for multiplication: {self.shape} vs {other.shape}")

        crypto_context = self.data.GetCryptoContext()
        ciphertext = crypto_context.EvalMul(self.data, other.data)
        return self.clone(ciphertext)

    def _multiply_scalar(self, scalar) -> "CTArray":
        """Multiply all elements by scalar."""
        # Implementation

    def _matvec(self, other, sumkey) -> "CTArray":
        crypto_context = self.data.GetCryptoContext()

        if not isinstance(other, FHETensor):
            ONP_ERROR("NOT_IMPLEMENTED")

        if self.ndim == 1 and other.ndim == 2:
            tensor_matrix = other.clone()
            tensor_vector = self.clone()
        elif self.ndim == 2 and other.ndim == 1:
            tensor_matrix = self.clone()
            tensor_vector = other.clone()
        else:
            ONP_ERROR("NOT_IMPLEMENTED")
        if tensor_matrix.original_shape[1] != tensor_vector.original_shape[0]:
            ONP_ERROR(
                f"Matrix dimension [{tensor_matrix.original_shape}] mismatch with vector dimension [{tensor_vector.shape}]"
            )
        if (
            tensor_matrix.order == MatrixOrder.ROW_MAJOR
            and tensor_vector.order == MatrixOrder.COL_MAJOR
        ):
            ciphertext = EvalMultMatVec(
                sumkey,
                MatVecEncoding.MM_CRC,
                tensor_matrix.ncols,
                tensor_vector.data,
                tensor_matrix.data,
            )
            return CTArray(
                ciphertext,
                (tensor_matrix.original_shape[0], 1),
                tensor_matrix.batch_size,
                tensor_matrix.ncols,
                MatrixOrder.COL_MAJOR,
            )

        elif (
            tensor_matrix.order == MatrixOrder.COL_MAJOR
            and tensor_vector.order == MatrixOrder.ROW_MAJOR
        ):
            ct_product = EvalMultMatVec(
                crypto_context,
                sumkey,
                MatVecEncoding.MM_RCR,
                tensor_matrix.ncols,
                tensor_vector.data,
                tensor_matrix.data,
            )
            return CTArray(
                ct_product,
                (tensor_matrix.original_shape[0], 1),
                tensor_matrix.batch_size,
                tensor_matrix.ncols,
                MatrixOrder.ROW_MAJOR,
            )
        else:
            ONP_ERROR(
                "Encoding styles of matrix and vector must be complementary (ROW_MAJOR/COL_MAJOR or vice versa)."
            )

    def _matmul(self, other: "CTArray") -> "CTArray":
        if self.shape != other.shape:
            if isinstance(other, FHETensor):
                if other.ndim == 1:
                    return self._matvec(other, self.ncols)
                return self.clone(EvalMatMulSquare(self.data, other.data, self.ncols))

        else:
            ONP_ERROR(
                f"Matrix dimension mismatch for multiplication: {self.shape} and {other.shape}"
            )

    def _dot(self, other: "CTArray") -> "CTArray":
        if self.ndim == 1 and other.ndim == 1:
            crypto_context = self.data.GetCryptoContext()
            ciphertext = crypto_context.EvalInnerProduct(self.data, other.data, self.ncols)
            return self.clone(ciphertext)
        else:
            return self._matmul(other)

    def _pow(self, exp: int) -> "CTArray":
        """Exponentiate a matrix to power k using homomorphic multiplication."""
        if not isinstance(exp, int):
            ONP_ERROR(f"Exponent must be integer, got {type(exp).__name__}")

        if exp < 0:
            ONP_ERROR("Negative exponent not supported in homomorphic encryption")

        if exp == 0:
            # return algebra.eye(self))
            pass

        if exp == 1:
            return self.clone()

        # Binary exponentiation implementation
        base = self.clone()
        result = None

        while exp:
            if exp & 1:
                result = base if result is None else base @ result
            base = base @ base
            exp >>= 1
        return result

    def _sum(self, axis=0) -> "CTArray":
        """
        Compute the cumulative sum of tensor elements along a given axis.

        Parameters
        ----------
        axis : int, optional
            Axis along which the cumulative sum is computed. Default is 0.

        Returns
        -------
        CTArray
            A new tensor with cumulative sums along the specified axis.
        """
        if axis == 0:
            ciphertext = _openfhe_numpy.EvalSumCumRows(
                self.data, self.ncols, self.original_shape[1]
            )
        else:
            ciphertext = _openfhe_numpy.EvalSumCumCols(self.data, self.ncols)
        return self.clone(ciphertext)

    def _reduce(self, axis=0) -> "CTArray":
        if axis == 0:
            ciphertext = _openfhe_numpy.EvalReduceCumRows(
                self.data, self.ncols, self.original_shape[1]
            )
        else:
            ciphertext = _openfhe_numpy.EvalReduceCumCols(self.data, self.ncols)
        return self.clone(ciphertext)

    def _transpose(self) -> "CTArray":
        ciphertext = _openfhe_numpy.EvalTranspose(self.data, self.ncols)
        shape = (self.original_shape[1], self.original_shape[0])
        ncols = next_power_of_two(shape[1])
        return CTArray(ciphertext, shape, self.batch_size, ncols, self.order)

    def _trace(self) -> "CTArray":
        """Sum along the main diagonal of a 2-D array:"""
        ONP_ERROR("Trace operation not implemented for CTArray.")


# -----------------------------------------------------------
# PTArray - Plaintext Tensor
# -----------------------------------------------------------
class PTArray(FHETensor[openfhe.Plaintext]):
    """Concrete tensor class for OpenFHE plaintexts."""

    def clone(self, data=None):
        return PTArray(
            data or self.data,
            self.original_shape,
            self.batch_size,
            self.ncols,
            self.order,
        )

    # @property
    # def dtype(self) -> Literal["PTArray"]:
    #     return "PTArray"

    def decrypt(self, *args, **kwargs):
        raise NotImplementedError("Decrypt not implemented for plaintext")

    def __repr__(self) -> str:
        return f"PTArray(meta={self.meta})"

    def serialize(self) -> dict:
        raise NotImplementedError("Serialize not implemented for plaintext")

    @classmethod
    def deserialize(cls, obj: dict) -> "PTArray":
        raise NotImplementedError("Deserialize not implemented for plaintext")


def copy_tensor(tensor: "FHETensor") -> "FHETensor":
    """
    Generic copy constructor for FHETensor and subclasses.

    Parameters
    ----------
    tensor : FHETensor
        Tensor to be copied.

    Returns
    -------
    FHETensor
        A new instance with the same metadata and (optionally deep-copied) data.
    """
    import copy

    return type(tensor)(
        data=copy.deepcopy(tensor.data),
        original_shape=tensor.original_shape,
        batch_size=tensor.batch_size,
        ncols=tensor.ncols,
        order=tensor.order,
    )
