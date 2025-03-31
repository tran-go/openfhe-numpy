import openfhe
from enum import Enum

PT = openfhe.Plaintext
CT = openfhe.Ciphertext
CC = openfhe.CryptoContext
KP = openfhe.KeyPair


class MatrixEncoding:
    ROW_MAJOR = "R"
    COL_MAJOR = "C"
    DIAG_MAJOR = "D"


class PackStyles:
    """
    Attributes:
    -----------
    MM_CRC : int
        Pack the matrix row-wise and the vector column-wise, resulting in a column-wise order.

    MM_RCR : int
        Pack the matrix column-wise and the vector row-wise, resulting in a row-wise order.

    MM_DIAG : int
        Pack the matrix diagonally.
    """

    MM_CRC = 0
    MM_RCR = 1
    MM_DIAG = 2


PRECISION_DEFAULT = 1
