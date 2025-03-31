import os
import numpy as np
import math
from typing import Tuple


# import openfhe related libraries
import openfhe
import openfhe_matrix

# # import config and auxilarries files
from openfhe_numpy.config import *
from openfhe_numpy.matlib import *
import openfhe_numpy.utils as utils
from openfhe_numpy.ptarray import ravel_mat, ravel_vec

"""
todo openfhenumpy <<<<---- CHOOSE IT
todo change name class ctarray:
"""


# a  = [1,2,3,4]
# ndarray
# np.array(a)


class ctarray:
    """
    Simulate numpy.ndarray.
    Case 1. 1 matrix = 1 ct
    """

    def __init__(
        self,
        data: openfhe.Ciphertext,
        original_shape: Tuple[int, int],
        ndim: int,
        batch_size: int,  # available slots that you can use )
        ncols: int = 1,
        encoding_order: int = MatrixEncoding.ROW_MAJOR,
    ):
        """
        Initializes the class with the given parameters.

        Parameters
        ----------
        data : openfhe.Ciphertext
            The encrypted data (ciphertext).
        original_shape : Tuple[int, int]
            The original dimensions of the data before padding, in the form (nrows, ncols).
        ndim : int
            Dimension of an array
        batch_size : int
            The maximum available slots
        ncols : int, optional
            The number of columns (default is 1, representing the block size).
        encoding_order : int, optional
            The encoding order, default is `MatrixEncoding.ROW_MAJOR`.
        """

        self.data = data
        self.original_shape = original_shape
        self.ndim = ndim
        self.ncols = ncols
        self.nrows = batch_size // ncols
        self.batch_size = batch_size
        self.encoding_order = encoding_order

    @property
    def shape(self):
        """Return the shape of the array"""
        return (self.nrows, self.ncols)

    @property
    def info(self):
        return [
            None,
            self.original_shape,
            self.ndim,
            self.batch_size,
            self.ncols,
            self.encoding_order,
        ]

    def copy(self):
        return ctarray(
            self.data,
            self.original_shape,
            self.ndim,
            self.batch_size,
            self.ncols,
            self.encoding_order,
        )

    def decrypt(self, cc, sk, format=1, precision=PRECISION_DEFAULT):
        result = cc.Decrypt(self.data, sk)
        result.SetLength(self.batch_size)
        result.GetFormattedValues(precision)
        result = result.GetRealPackedValue()
        if format:
            result = utils.format(result, self.ndim, self.original_shape, self.shape)
        return result

    def ravel(self, cc, keys, rot_keys, encoding_order):
        # change the original data
        # todo this function use to perform linear transformation. I will do it later
        if self.encoding_order == "R":
            if encoding_order == "C":
                print("...change order from RW to CW")

        if self.encoding_order == "C":
            if encoding_order == "R":
                print("...change order from CW to RW")

        return

    def flatten(self, cc, keys, rot_keys, encoding_order):
        # Return a copy of the array collapsed into one dimension.
        return

    def transpose(self, cc, keys, rot_keys):
        return

    # CALCULATION

    def sum(self, cc, keys, rot_keys, axis):
        """
        Return the sum of the array elements over the given axis.
        """
        return

    def cumsum(self, cc, keys, rot_keys, axis):
        # Return the cumulative sum of the elements along the given axis.
        return


#########################################
# Public Methods
#########################################


def array(
    cc,
    data: list,
    size: int,
    block_size: int = 1,
    encoding_type: int = MatrixEncoding.ROW_MAJOR,
    type: str = "C",
    pub_key: KP = None,
):
    """
    block_size = row_size, number of repetitions, number of columns
    block_size is important for packing vectors
    """
    org_rows, org_cols, ndim = utils.get_shape(data)

    print(org_rows, org_cols, ndim)
    if ndim == 2:
        ncols = next_power2(org_cols)
        plaintext = ravel_mat(cc, data, size, ncols, encoding_type)

    else:
        ncols = block_size
        plaintext = ravel_vec(cc, data, size, ncols, encoding_type)

    if type == "P":
        return ptarray(
            plaintext,
            (org_rows, org_cols),
            ndim,
            size,
            ncols,
            encoding_type,
        )
    else:
        ciphertext = cc.Encrypt(pub_key, plaintext)
        return ctarray(
            ciphertext,
            (org_rows, org_cols),
            ndim,
            size,
            ncols,
            encoding_type,
        )


def ptarray(
    cc,
    data: list,
    size: int,
    block_size: int = 1,
    encoding_type: int = MatrixEncoding.ROW_MAJOR,
    type: str = "C",
):
    """
    block_size = row_size, number of repetitions, number of columns
    block_size is important for packing vectors
    """
    org_rows, org_cols, ndim = utils.get_shape(data)

    print(org_rows, org_cols, ndim)
    if ndim == 2:
        ncols = next_power2(org_cols)
        plaintext = ravel_mat(cc, data, size, ncols, encoding_type)

    else:
        ncols = block_size
        plaintext = ravel_vec(cc, data, size, ncols, encoding_type)

    return ptarray(
        plaintext,
        (org_rows, org_cols),
        ndim,
        size,
        ncols,
        encoding_type,
    )


def decrypt(cc, sk, data, size, precision=3):
    result = cc.Decrypt(data, sk)
    result.SetLength(size)
    result.GetFormattedValues(precision)
    result = result.GetRealPackedValue()
    result = [round(result[i], precision) for i in range(size)]
    return result


def gen_sum_row_keys(cc, sk, block_size):
    return cc.EvalSumRowsKeyGen(sk, None, block_size)


def gen_sum_col_keys(cc, sk, block_size):
    return cc.EvalSumColsKeyGen(sk)


def gen_rotation_keys(cc, sk, rotation_indices):
    cc.EvalRotateKeyGen(sk, rotation_indices)


#########################################
# Matrix Operations
#########################################
def matmul_square(cc: CC, keys: KP, ctm_A: ctarray, ctm_B: ctarray):
    """P
    Matrix product of two array

    Parameters
    ----------
    ctm_A: ctarray
    ctm_B: ctarray

    Returns
    -------
    ctarray
        Product of two square matrices
    """

    print("DEBUG: ctm_A.ncols= ", ctm_A.ncols)
    ct_prod = openfhe_matrix.EvalMatMulSquare(cc, keys, ctm_A.data, ctm_B.data, ctm_A.ncols)

    info = ctm_A.info
    info[0] = ct_prod
    return ctarray(*info)


def matvec(cc, keys, sum_col_keys, ctm_mat, ctv_v, block_size):
    """Matrix-vector dot product of two arrays."""
    # print(ctm_mat.encoding_order, ctv_v.encoding_order)
    if ctm_mat.encoding_order == "R" and ctv_v.encoding_order == "C":
        # print("CRC")
        ct_prod = openfhe_matrix.EvalMultMatVec(
            cc,
            keys,
            sum_col_keys,
            PackStyles.MM_CRC,
            block_size,
            ctv_v.data,
            ctm_mat.data,
        )
        rows, cols = ctm_mat.original_shape
        info = [ct_prod, (rows, 1), False, ctm_mat.size, cols, "C"]
        return ctarray(*info)

    elif ctm_mat.encoding_order == "C" and ctv_v.encoding_order == "R":
        # print("RCR")
        ct_prod = openfhe_matrix.EvalMultMatVec(
            cc,
            keys,
            sum_col_keys,
            PackStyles.MM_RCR,
            block_size,
            ctv_v.data,
            ctm_mat.data,
        )
        rows, cols = ctm_mat.original_shape
        info = [ct_prod, (rows, 1), False, ctm_mat.size, cols, "R"]
        return ctarray(*info)
    else:
        print("ERROR [matvec] encoding styles are not matching!!!")
        return None


def matrix_power(cc: CC, keys: KP, k: int, ctm_A: ctarray):
    """Raise a square matrix to the (integer) power n."""
    # todo power and squaring
    for i in range(k):
        res = matmul_square(cc, keys, ctm_A, ctm_A)
    return res


def matrix_transpose(ctm_mat):
    """
    Transposes a matrix (or a stack of matrices) x.
    Encoding converting: row-wise becomes column-wise and vice versal
    """
    return None


def dot(cc, keys, sum_col_keys, ctm_A, ctm_B):
    """
    Dot product of two arrays.
    - If both a and b are 1-D arrays, it is inner product of vectors (without complex conjugation).
    - If both a and b are 2-D arrays, it is matrix multiplication, but using matmul or a @ b is preferred.

    Example:
        dot(A.B) = A@B
        dot (v,w) = <v,w>
    """
    if ctm_A.ndim == ctm_B.ndim:
        ct_mult = cc.EvalMult(ctm_A.data, ctm_B.data)
        ct_prod = cc.EvalSumCols(ct_mult, ctm_A.ncols, sum_col_keys)
        rows, cols = ctm_A.original_shape
        info = ctm_A.info
        info[0] = ct_prod
        return ctarray(*info)
    else:
        return multiply(cc, keys, ctm_A, ctm_B)


# Hadamard product: multiply arguments element-wise.
def multiply(cc, keys, ctm_A, ctm_B):
    info = ctm_A.info
    info[0] = cc.EvalMult(ctm_A.data, ctm_B.data)
    return ctarray(*info)


def add(cc, ctm_A, ctm_B):
    # Add arguments element-wise.
    info = ctm_A.info
    info[0] = cc.EvalAdd(ctm_A.data, ctm_B.data)
    return ctarray(*info)


def sub(cc, keys, ctm_A, ctm_B):
    # Subtracts arguments element-wise.
    info = ctm_A.info
    info[0] = cc.EvalSub(ctm_A.data, ctm_B.data)
    return ctarray(*info)


def sum(cc, sum_keys, cta, axis=None):
    """Sum of array elements over a given axis"""
    # axis = None: sum everything
    # axis = 0: sum all rows
    # axis = 1: sum all cols
    rows_key, cols_key = sum_keys
    info = cta.info
    if cta.encoding_order == "R":
        if axis == 1:
            info[0] = cc.EvalSumCols(cta, cta.ncols, cols_key)
            info.ncols = 1
        elif axis == 0:
            info[0] = cc.EvalSumRows(cta, cta.ncols, rows_key)
            info.nrows = 1
        else:
            info[0] = cc.EvalSumCols(cta, cta.ncols, cols_key)
            info[0] = cc.EvalSumRows(info[0], info[0].ncols, rows_key)
            info.nrows = 1
            info.ncols = 1
        return ctarray(*info)
    else:
        # remark: for different encoding, need to repack it or find a better way to do the sum
        return None

    return None


def mean(cc, sum_keys, cta, axis=None):
    """Compute the arithmetic mean along the specified axis."""
    total = sum(cc, sum_keys, cta, axis)
    if axis == 0:
        n = cta.nrows
    elif axis == 1:
        n = cta.ncols
    else:
        n = 1
    total.data = cc.EvalMul(total.data, n)
    return total


# def cumsum(cc, sum_keys, cta, axis=None):
#     return


# def reduce(cc, sum_keys, cta, axis=None):
#     return


# #####################################################
# # Helper functions
# #####################################################
