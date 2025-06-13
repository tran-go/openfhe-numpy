# Standard library imports
import numpy as np

# Third-party imports
import openfhe

# Local imports
from ..utils.constants import DataType, MatrixOrder
from ..utils.log import ONP_ERROR
from ..utils.packing import (
    _get_shape,
    _pack_matrix_col_wise,
    _pack_matrix_row_wise,
    _pack_vector_row_wise,
    _pack_vector_col_wise,
)
from ..utils.matlib import is_power_of_two, next_power_of_two
from .ctarray import CTArray
from .ptarray import PTArray
from .tensor import FHETensor


# TODO: constructor for block matrix
def _get_block_dimensions(data, slots) -> tuple[int, int]:
    pass


def _pack_array(
    data: list | tuple | np.ndarray | int,
    batch_size: int,
    order: int = MatrixOrder.ROW_MAJOR,
    repeats: int = 0,
):
    if batch_size < 0:
        ValueError("The batch size cannot be negative.")

    org_rows, org_cols, ndim = _get_shape(data)

    if isinstance(data, int):
        packed_data = [data] * batch_size
        shape = (batch_size,)
    else:
        if ndim == 2:
            packed_data, shape = _ravel_matrix(data, batch_size, order, True, repeats)
        elif ndim == 1:
            # if repeats == 0 and order == MatrixOrder.ROW_MAJOR:
            #     repeats = 1
            packed_data, shape = _ravel_vector(data, batch_size, order, True, repeats)
        else:
            ONP_ERROR("Not support dimension [{ndim}]")
    return {
        "data": packed_data,
        "original_shape": (org_rows, org_cols),
        "ndim": ndim,
        "batch_size": batch_size,
        "shape": shape,
        "order": order,
    }


def array(
    cc: openfhe.CryptoContext,
    data: list | tuple | np.ndarray | int,
    slots: int = -1,
    # ncols: int = 0,
    order: int = MatrixOrder.ROW_MAJOR,
    type: str = DataType.CIPHERTEXT,
    package=None,
    public_key=None,
) -> FHETensor:
    """
    Construct either a ciphertext (FHETensor) or plaintext (PTArray) from raw input data.

    Parameters
    ----------
    cc : CryptoContext
        The OpenFHE CryptoContext.
    data : list
        Input list or matrix data.
    slots : int
        Number of total CKKS slots.
    ncols : int
        Size of a row.
    order : int
        MatrixOrder.ROW_MAJOR or COL_MAJOR.
    type : str
        DataType.CIPHERTEXT or PLAINTEXT.
    public_key : optional
        Public key needed for encryption.

    Returns
    -------
    FHETensor Object
    """

    if package is None:
        package = _pack_array(data, slots, order, 0)

    packed_data = package["data"]
    org_rows = package["original_shape"][0]
    org_cols = package["original_shape"][1]
    ndim = package["ndim"]
    slots = package["batch_size"]
    shape = package["shape"]
    order = package["order"]

    plaintext = cc.MakeCKKSPackedPlaintext(packed_data)

    if type == DataType.PLAINTEXT:
        result = PTArray(plaintext, (org_rows, org_cols), ndim, slots, shape, order)
    else:
        if public_key is None:
            raise ValueError("Public key must be provided for ciphertext encoding.")

        ciphertext = cc.Encrypt(public_key, plaintext)
        result = CTArray(ciphertext, (org_rows, org_cols), slots, shape, order, True)

    result.set_shape(shape)
    result.set_batch_size(slots)
    return result


def _ravel_matrix(data, slots, order=MatrixOrder.ROW_MAJOR, pad_to_pow2=True, repeats=0):
    """
    Encode a 2D matrix into a packed array.

    Parameters
    ----------
    data : list of list
    ncols : int
        Block size per row.
    order : MatrixOrder
    repeats : int
        Number of repeats.
        0: full repeats
        1: packing only
        n: n

    Returns
    -------
    Plaintext
    """

    if order == MatrixOrder.ROW_MAJOR:
        packed_data, shape = _pack_matrix_row_wise(data, slots, pad_to_pow2, repeats)
    elif order == MatrixOrder.COL_MAJOR:
        packed_data, shape = _pack_matrix_col_wise(data, slots, pad_to_pow2, repeats)
    else:
        raise ValueError("Unsupported encoding order")

    return packed_data, shape


def _ravel_vector(data, slots, order=MatrixOrder.ROW_MAJOR, pad_to_pow2=True, repeats=1):
    # def _ravel_vector(data, slots, ncols=1, order=MatrixOrder.ROW_MAJOR):
    """
    Encode a 1D vector into a packed array.

    Parameters
    ----------
    data : list
    repeats : int
        Number of repeats.
    order : MatrixOrder

    Returns
    -------
    Plaintext
    """

    if repeats < 0:
        raise ONP_ERROR("Number of repetitions must be ≥ 0. Full repeats: 0, partial repeats: n.")

    # It will depend to user
    # if not is_power_of_two(repeats):
    #     raise ONP_ERROR("Repetition count must be a power of two")

    if order == MatrixOrder.ROW_MAJOR:
        packed_data, shape = _pack_vector_row_wise(data, slots, repeats, pad_to_pow2, "default")
    elif order == MatrixOrder.COL_MAJOR:
        packed_data, shape = _pack_vector_col_wise(data, slots, repeats, pad_to_pow2, "default")
    else:
        raise ONP_ERROR("Unsupported encoding order")

    return packed_data, shape
