# import openfhe
from enum import Enum


# Types of data representation
class DataType:
    PLAINTEXT = "P"
    CIPHERTEXT = "C"


# Numerical constants
EPSILON = 1e-8
EPSILON_HIGH = 1e-4


# default format result
class UnpackType(Enum):
    RAW = "raw"
    ORIGINAL = "original"
    RESHAPE = "reshape"
