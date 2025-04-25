import openfhe
from enum import Enum

# Type Aliases for OpenFHE Types
Plaintext = openfhe.Plaintext
Ciphertext = openfhe.Ciphertext
CryptoContext = openfhe.CryptoContext
KeyPair = openfhe.KeyPair


# Encoding strategy for matrix packing
class MatrixEncoding:
    ROW_MAJOR = "R"  # Encode data row-wise (default)
    COL_MAJOR = "C"  # Encode data column-wise
    DIAG_MAJOR = "D"  # Optional: encode data diagonally (future use)


# Types of data representation
class DataType:
    PLAINTEXT = "P"
    CIPHERTEXT = "C"


# Defined in C++
# # Packing strategies used in EvalMultMatVec and similar operations
# class PackStyle:
#     """
#     Defines homomorphic multiplication strategies for matrix-vector products.

#     Attributes
#     ----------
#     MM_CRC : int
#         Matrix: Row-wise, Vector: Column-wise -> Result: Column-wise
#     MM_RCR : int
#         Matrix: Column-wise, Vector: Row-wise -> Result: Row-wise
#     MM_DIAG : int
#         Diagonal packing (not yet implemented)
#     """

#     MM_CRC = 0
#     MM_RCR = 1
#     MM_DIAG = 2  # Placeholder for diagonal style (experimental)
