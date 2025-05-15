import openfhe
from enum import Enum


# Type Aliases for OpenFHE Types
Plaintext = openfhe.Plaintext
Ciphertext = openfhe.Ciphertext
CryptoContext = openfhe.CryptoContext
KeyPair = openfhe.KeyPair


# Encoding strategy for matrix packing
class MatrixOrder:
    ROW_MAJOR = "R"  # Encode data row-wise (default)
    COL_MAJOR = "C"  # Encode data column-wise
    DIAG_MAJOR = "D"  # Optional: encode data diagonally (future use)


# Types of data representation
class DataType:
    PLAINTEXT = "P"
    CIPHERTEXT = "C"
