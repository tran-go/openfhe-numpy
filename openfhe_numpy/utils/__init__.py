"""Utility functions for OpenFHE-NumPy."""

# Import utility functions from modules
from .utils import (
    is_power_of_two,
    next_power_of_two,
    check_equality_matrix,
)

"""Logging utilities for OpenFHE-NumPy."""
from .log import get_logger, ONPError, InvalidAxisError, ONPNotImplementedError

# Import enum types if present
try:
    from .config import MatrixOrder, DataType
except ImportError:
    pass

# Define public API
__all__ = [
    # Math utilities
    "is_power_of_two",
    "next_power_of_two",
    # Validation utilities
    "check_equality_matrix",
    # Enums 
    "MatrixOrder",
    "DataType",
    "get_logger",
    "ONPError",
    "InvalidAxisError",
    "ONPNotImplementedError",
]

