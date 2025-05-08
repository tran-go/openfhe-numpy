from openfhe_numpy.tensor import *
from openfhe_numpy.config import MatrixOrder, DataType
from openfhe_numpy import utils
import numpy as np
import openfhe


def array(
    cc: openfhe.CryptoContext,
    data: list | tuple | np.ndarray | int,
    slots: int = -1,
    rowsize: int = 0,
    order: int = MatrixOrder.ROW_MAJOR,
    type: str = DataType.CIPHERTEXT,
    public_key=None,
) -> BaseTensor:
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
    rowsize : int
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

    org_rows, org_cols, ndim = utils.get_shape(data)

    if slots == -1:
        slots = cc.GetBatchSize()

    if isinstance(data, int):
        packed_data = np.zeros(slots)
    else:
        if ndim == 2:
            if rowsize == 0:
                rowsize = utils.next_power_of_two(org_cols)
            packed_data = ravel_matrix(data, slots, rowsize, order)
        else:
            if rowsize == 0:
                rowsize = 1
            packed_data = ravel_vector(data, slots, rowsize, order)
    plaintext = cc.MakeCKKSPackedPlaintext(packed_data)

    if type == DataType.PLAINTEXT:
        result = PTArray(plaintext, (org_rows, org_cols), ndim, slots, rowsize, order)
    else:
        if public_key is None:
            raise ValueError("Public key must be provided for ciphertext encoding.")

        ciphertext = cc.Encrypt(public_key, plaintext)
        result = CTArray(ciphertext, (org_rows, org_cols), slots, rowsize, order)

    result.set_rowsize(rowsize)
    result.set_batch_size(slots)
    return result


def ravel_matrix(data, slots, rowsize=1, order=MatrixOrder.ROW_MAJOR, reps=1):
    """
    Encode a 2D matrix into a CKKS plaintext.

    Parameters
    ----------
    data : list of list
    rowsize : int
        Block size per row.
    order : MatrixOrder
    reps : int
        Number of repetitions

    Returns
    -------
    Plaintext
    """

    if order == MatrixOrder.ROW_MAJOR:
        packed_data = utils.pack_mat_row_wise(data, rowsize, slots, reps)
    elif order == MatrixOrder.COL_MAJOR:
        packed_data = utils.pack_mat_col_wise(data, rowsize, slots, reps)
    else:
        raise ValueError("Unsupported encoding order")

    return packed_data


def ravel_vector(data, slots, rowsize=1, order=MatrixOrder.ROW_MAJOR):
    """
    Encode a 1D vector into a CKKS plaintext.

    Parameters
    ----------
    data : list
    rowsize : int
        Number of repetitions.
    order : MatrixOrder

    Returns
    -------
    Plaintext
    """

    if rowsize < 1:
        raise ValueError("Number of repetitions must be > 0")

    if rowsize == 1 and order == MatrixOrder.ROW_MAJOR:
        raise ValueError("Can't encode row-wise with rowsize = 1")

    if not utils.is_power_of_two(rowsize):
        raise ValueError("Repetition count must be a power of two")

    if order == MatrixOrder.ROW_MAJOR:
        packed_data = utils.pack_vec_row_wise(data, rowsize, slots)
    elif order == MatrixOrder.COL_MAJOR:
        packed_data = utils.pack_vec_col_wise(data, rowsize, slots)
    else:
        raise ValueError("Unsupported encoding order")

    return packed_data
