"""Utility functions for OpenFHE-NumPy.

This module provides math utilities, validation functions, and logging tools
for the OpenFHE-NumPy package.
"""

from openfhe_numpy.utils.matlib import is_power_of_two, next_power_of_two, check_equality_matrix
from openfhe_numpy.utils.log import ONPError


__all__ = [
    "is_power_of_two",
    "next_power_of_two",
    "check_equality_matrix",
    "ONPError",
]
