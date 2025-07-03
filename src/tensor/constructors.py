# Third-party imports
from typing import Optional, Literal

# from pydantic import validate_call
import numpy as np
import openfhe

# Package-level imports (consider using relative imports here)
from openfhe_numpy._onp_cpp import ArrayEncodingType

# Utils imports
from openfhe_numpy.utils.errors import ONP_ERROR
from openfhe_numpy.utils.matlib import is_power_of_two
from openfhe_numpy.utils.packing import (
    _pack_matrix_col_wise,
    _pack_matrix_row_wise,
    _pack_vector_row_wise,
    _pack_vector_col_wise,
)
from openfhe_numpy.utils.typecheck import (
    is_numeric_scalar,
    is_numeric_arraylike,
    Number,
)

# Tensor imports
from .ctarray import CTArray
from .ptarray import PTArray
from .tensor import FHETensor


# TODO: constructor for block matrix
def _get_block_dimensions(data, slots) -> tuple[int, int]:
    pass


def block_array(
    cc: openfhe.CryptoContext,
    data: np.ndarray | Number | list,
    batch_size: Optional[int] = None,
    order: int = ArrayEncodingType.ROW_MAJOR,
    type: str = "C",
    mode: str = "repeat",
    package: Optional[dict] = None,
    public_key: openfhe.PublicKey = None,
    **kwargs,
):
    pass


def _pack_array(
    data: np.ndarray | Number | list,
    batch_size: int,
    order: int = ArrayEncodingType.ROW_MAJOR,
    mode: str = "repeat",
    **kwargs,
):
    """
    Helper function to flatten a scalar, vector, or matrix into a 1D array and fill all slots by either padding with zeros or duplicating the array contents.

    Parameters
    ----------
    data : np.ndarray | Number | list
    batch_size : int
        The number of available plaintext slots
    order : int, optional
        The packing/encoding style, by default ArrayEncodingType.ROW_MAJOR
    repeats : int, optional
        The number of repetitions.
        When repeats = -1, the the array content is duplicated all avaialble slots. Ortherwise, it will repeats n time. when repeats = 1, means only repeat 1 times,
        Example: batch_size = 32, encoding type is row-major, repeats
        Case 1: repeats  = -1
        [[1 2 3], [4 5 6]] -> 1 2 3 0 4 5 6 0 | 1 2 3 0 4 5 6 0 | 1 2 3 0 4 5 6 0 | 1 2 3 0 4 5 6 0
        [[1], [2], [3]] -> 1 1 1 1 | 2 2 2 2 | 3 3 3 3 | 0 0 0 0
        [1 2 3] -> 1 2 3 0 1 2 3 0 | 1 2 3 0 1 2 3 0 | 1 2 3 0 1 2 3 0 | 1 2 3 0 1 2 3 0
        [1] -> 1 0 0 0 | 1 0 0 0 | 1 0 0 0 | 1 0 0 0 | 1 0 0 0 | 1 0 0 0 | 1 0 0 0 | 1 0 0 0

    Returns
    -------
    _type_
        _description_
    """

    if batch_size < 0:
        ONP_ERROR("The batch size cannot be negative.")
    if not is_power_of_two(batch_size):
        ONP_ERROR(f"Batch size [{batch_size}] must be a power of two")

    # We convert the data to numpy array
    data = np.array(data)

    if is_numeric_scalar(data):
        if mode == "zero":
            packed_data = np.zeros(batch_size, dtype=type(data))
            packed_data[0] = data
        elif mode == "repeat":
            packed_data = np.full(batch_size, data)
        else:
            ONP_ERROR(f"Invalid padding mode: '{mode}'. Valid options are 'zero' or 'repeat'.")

    elif is_numeric_arraylike(data):
        if data.ndim == 2:
            packed_data, shape = _ravel_matrix(data, batch_size, order, True, mode, **kwargs)
        elif data.ndim == 1:
            packed_data, shape = _ravel_vector(data, batch_size, order, True, mode, **kwargs)
        else:
            ONP_ERROR(f"Not support dimension [{data.ndim}]")
    else:
        ONP_ERROR("Input is not numeric")

    return {
        "data": packed_data,
        "original_shape": data.shape,
        "ndim": data.ndim,
        "batch_size": batch_size,
        "shape": shape,
        "order": order,
    }


def array(
    cc: openfhe.CryptoContext,
    data: np.ndarray | Number | list,
    batch_size: Optional[int] = None,
    order: int = ArrayEncodingType.ROW_MAJOR,
    type: Literal["C", "P"] = "C",
    mode: str = "repeat",
    package: dict = {},
    public_key: openfhe.PublicKey = None,
    **kwargs,
) -> FHETensor:
    """
    Construct either a ciphertext or plaintext (CTArray/PTArray) from raw input data.

    Parameters
    ----------
    cc : CryptoContext
        The OpenFHE CryptoContext.
    data : matrix/vector/int
    batch_size : int
        Number of total plaintext batch_size.
    order : int
        Encoding order: ArrayEncodingType.ROW_MAJOR or ArrayEncodingType.COL_MAJOR.
    type : str
        DataType.CIPHERTEXT or PLAINTEXT.
    public_key : optional
        Public key needed for encryption if the output is encrypted

    Returns
        - A matrix has a dimension of m x n (m rows x n cols)
        - A vector is considered as n x 1 matrix
        - A number is considered as a vector with duplicated entries: [1 1 1 1] or [1 0 0 0]
    -------
    FHETensor Object
    """

    if cc is None:
        ONP_ERROR("CryptoContext cannot be None")

    if batch_size is not None and not isinstance(batch_size, int):
        ONP_ERROR(f"batch_size must be int or None, got {type(batch_size).__name__}")

    if package == {}:
        package = _pack_array(data, batch_size, order, mode, **kwargs)

    packed_data = package["data"]
    ndim = package["ndim"]
    batch_size = package["batch_size"]
    shape = package["shape"]
    order = package["order"]
    plaintext = cc.MakeCKKSPackedPlaintext(packed_data)

    if type == "P":
        result = PTArray(plaintext, package["original_shape"], batch_size, shape, order)
    else:
        if public_key is None:
            ONP_ERROR("Public key must be provided for ciphertext encoding.")

        ciphertext = cc.Encrypt(public_key, plaintext)
        result = CTArray(ciphertext, package["original_shape"], batch_size, shape, order)

    result.set_shape(shape)
    result.set_batch_size(batch_size)
    return result


def _ravel_matrix(data, batch_size, order=ArrayEncodingType.ROW_MAJOR, pad_to_pow2=True, mode="repeat", **kwargs):
    """
    Encode a 2D matrix into a packed array.

    Returns
    -------
    Plaintext
    """

    if order == ArrayEncodingType.ROW_MAJOR:
        packed_data, shape = _pack_matrix_row_wise(data, batch_size, pad_to_pow2, mode)
    elif order == ArrayEncodingType.COL_MAJOR:
        packed_data, shape = _pack_matrix_col_wise(data, batch_size, pad_to_pow2, mode)
    else:
        raise ValueError("Unsupported encoding order")

    return packed_data, shape


def _ravel_vector(data, batch_size, order=ArrayEncodingType.ROW_MAJOR, pad_to_pow2=True, tile="repeats", **kwargs):
    """
    Encode a 1D vector into a packed array.

    Parameters
    ----------
    data : list
    repeats : int
        Number of repeats.
    order : ArrayEncodingType

    Returns
    -------
    Plaintext
    """

    target_cols = kwargs.get("target_cols")
    if target_cols is not None:
        if not (isinstance(target_cols, int) and target_cols > 0):
            ONP_ERROR(f"target_cols must be positive int or None, got {target_cols!r}")

    pad_value = kwargs.get("pad_value", "repeat")
    expand = kwargs.get("expand", "repeat")

    if order == ArrayEncodingType.ROW_MAJOR:
        packed_data, shape = _pack_vector_row_wise(data, batch_size, target_cols, expand, tile, pad_to_pow2, pad_value)
    elif order == ArrayEncodingType.COL_MAJOR:
        packed_data, shape = _pack_vector_col_wise(data, batch_size, target_cols, expand, tile, pad_to_pow2, pad_value)
    else:
        raise ONP_ERROR("Unsupported encoding order")
    return packed_data, shape
