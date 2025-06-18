import openfhe_numpy._onp_cpp as backend
from enum import Enum

# Numerical constants
EPSILON = 1e-8
EPSILON_HIGH = 1e-4


# Types of data representation
class DataType:
    PLAINTEXT = "P"
    CIPHERTEXT = "C"


# Default format result
class UnpackType(Enum):
    RAW = "raw"
    ORIGINAL = "original"
    RESHAPE = "reshape"
