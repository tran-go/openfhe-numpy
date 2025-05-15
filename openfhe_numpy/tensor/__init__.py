"""Tensor implementations for homomorphic encryption operations."""

# Import tensor classes
from .tensor import (
    BaseTensor,
    FHETensor,
    PTArray,
    CTArray,
)

# Import tensor constructors
from .constructors import array

# Define public API
__all__ = [
    # Core tensor classes
    "BaseTensor",
    "FHETensor",
    "PTArray",
    "CTArray",
    # Constructor functions
    "array",
]
