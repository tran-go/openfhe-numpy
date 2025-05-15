from typing import Any
import numpy as np


from openfhe_numpy.tensor import FHETensor, CTArray, PTArray
from openfhe_numpy.config import MatrixOrder
from openfhe_numpy.utils.log import ONP_ERROR, ONP_DEBUG


def add(a: Any, b: Any) -> Any:
    """Functional sub: delegates to a + b"""
    if hasattr(a, "dtype") or hasattr(b, "dtype"):
        if a.dtype == "BlockCTArray" or b.dtype == "BlockCTArray":
            return a + b
        else:
            if a.dtype == CTArray or b.dtype == "CTArray":
                return a + b

    ONP_ERROR(f"Addition not implemented between {type(a).__name__} and {type(b).__name__}")


def sub(a: Any, b: Any) -> Any:
    """Functional sub: delegates to a - b"""
    if hasattr(a, "dtype") or hasattr(b, "dtype"):
        if a.dtype == "BlockCTArray" or b.dtype == "BlockCTArray":
            return a + b
        else:
            if a.dtype == CTArray or b.dtype == "CTArray":
                return a + b

    ONP_ERROR(f"Addition not implemented between {type(a).__name__} and {type(b).__name__}")


def multiply(a: Any, b: Any) -> Any:
    if hasattr(a, "dtype") or hasattr(b, "dtype"):
        if (
            a.dtype == "BlockCTArray"
            or b.dtype == "BlockCTArray"
            or a.dtype == "CTArray"
            or b.dtype == "CTArray"
        ):
            return a * b

    ONP_ERROR(f"Addition not implemented between {type(a).__name__} and {type(b).__name__}")


def dot(a: Any, b: Any) -> Any:
    return a * b


def matmul(a: Any, b: Any) -> Any:
    """Functional matmul: delegates to a @ b"""
    return a @ b


def power(a: Any, exp: int) -> Any:
    """Functional power: delegates to a ** 2"""
    return a**exp


def sum(tensor_a: "CTArray", axis=None, keepdims=False) -> "CTArray":
    """
    Reduce of array elements over specified axis.

    Parameters
    ----------
    tensor : CTArray
        Input tensor
    axis : int, optional
        Axis along which to sum
    keepdims : bool, optional
        Whether to keep dimensions with length 1

    Returns
    -------
    CTArray
        Reduce of tensor elements
    """
    return tensor_a.sum(axis)


def reduce(tensor_a: "CTArray", axis=None, keepdims=False) -> "CTArray":
    """
    Reduce of array elements over specified axis.

    Parameters
    ----------
    tensor : CTArray
        Input tensor
    axis : int, optional
        Axis along which to sum
    keepdims : bool, optional
        Whether to keep dimensions with length 1

    Returns
    -------
    CTArray
        Reduce of tensor elements
    """
    return tensor_a.reduce(axis)


def transpose(tensor_a: "CTArray") -> "CTArray":
    return tensor_a.transpose()


# Array Creation Functions:
def zeros(shape, crypto_context, key):
    """Create an encrypted array of zeros."""
    pass


def ones(shape, crypto_context, key):
    """Create an encrypted array of ones."""
    pass


def eye(n, crypto_context, key):
    """Create an encrypted identity matrix."""
    pass


# Broadcasting Support:
def _get_broadcast_shape(self, other):
    """Calculate broadcast shape between tensors."""
    # Implementation...


def _broadcast_tensor(self, target_shape):
    """Broadcast this tensor to target shape."""
    # Implementation...


# Array Manipulation Methods:
def reshape(self, new_shape):
    """Reshape tensor to new dimensions."""
    # Implementation...


def concat(tensors, axis=0):
    """Concatenate tensors along specified axis."""
    # Implementation...
