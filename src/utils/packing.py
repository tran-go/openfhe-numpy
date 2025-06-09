from typing import Union

import numpy as np

from .log import ONP_ERROR
from .matlib import is_power_of_two, next_power_of_two


# === Shape and Utility Functions ==


def _get_shape(data: Union[List, Tuple, np.ndarray]) -> Tuple[int, int, int]:
    """Determine the shape and dimension of a given array

    Parameters
    ----------
    data : list, tuple, or ndarray
        The input matrix or array.

    Returns
    -------
    tuple
        A tuple containing the number of rows, number of columns, and dimensionality.

    Raises
    ------
    ONP_ERROR
        If input is neither list, tuple, nor ndarray.
    """
    if isinstance(data, (list, tuple)):
        rows = len(data)
        cols = len(data[0]) if isinstance(data[0], (list, tuple)) else 1
        ndim = 2 if cols > 1 else 1
        return rows, cols, ndim

    if isinstance(data, np.ndarray):
        if data.ndim == 1:
            return data.shape[0], 0, 1
        return data.shape[0], data.shape[1], 2

    ONP_ERROR("Invalid data type provided. Must be list, tuple, or ndarray.")


def print_matrix(matrix, rows):
    """
    Print a matrix in a nicely formatted way.

    Parameters
    ----------
    matrix : array_like
        A 2D matrix (list of lists or ndarray) to print.
    rows : int
        Number of rows to print from the matrix.
    """
    for i in range(rows):
        row_str = "\t".join(f"{val:.2f}" for val in matrix[i])
        print(f"[{row_str}]")


# === Vector Packing Functions ==


def _pack_vector_row_wise(v, block_size, num_slots):
    """
    Clone a vector v to fill num_slots
    1 -> 1111 2222 3333
    2
    3
    """
    n = len(v)
    assert is_power_of_two(block_size)
    assert is_power_of_two(num_slots)
    if num_slots < n:
        ONP_ERROR(
            "ERROR ::: [row_wise_vector] vector is longer than total   slots"
        )
    if num_slots == n:
        if num_slots // block_size > 1:
            ONP_ERROR(
                "ERROR ::: [row_wise_vector] vector is too longer, can't duplicate"
            )
        return v

    # print data
    assert num_slots % block_size == 0
    total_blocks = num_slots // block_size
    free_slots = num_slots - n * block_size

    # compute padding
    packed = np.zeros(num_slots)
    k = 0
    for i in range(n):
        for j in range(block_size):
            packed[k] = v[i]
            k += 1
    return packed


def _pack_vector_col_wise(v, block_size, num_slots):
    """
    Clone a vector v to fill num_slots
    1 -> 1230 1230 1230
    2
    3
    """
    n = len(v)
    assert is_power_of_two(block_size)
    assert is_power_of_two(num_slots)
    if block_size < n:
        ONP_ERROR(
            f"ERROR ::: [col_wise_vector] vector of size ({n}) is longer than size of a slot ({block_size})"
        )
    if num_slots < n:
        ONP_ERROR(
            "ERROR ::: [col_wise_vector] vector is longer than total slots"
        )
    if num_slots == n:
        return v

    packed = np.zeros(num_slots)

    # print data
    assert num_slots % block_size == 0
    total_blocks = num_slots // block_size
    free_slots = num_slots - n * total_blocks

    k = 0  # index into vector to write
    for i in range(total_blocks):
        for j in range(n):
            packed[k] = v[j]
            k += 1
        k += block_size - n

    return packed


# === Matrix Packing Functions ===


def _pack_matrix_row_wise(matrix, ncols, total_slots, is_row_padded=False):
    """Pack a matrix into a flat array row-wise with zero padding.

    Parameters
    ----------
    matrix : array_like
        The input 2D matrix to be packed.
    ncols : int
        Target row size after padding; must be a power of two.
    total_slots : int
        Total number of slots available in the output array; must be a power of two and divisible by ncols.
    is_row_padded : bool, optional
        If True, pad the number of rows to the next power of two. Default is False.

    Returns
    -------
    flat_array : ndarray
        Flat array containing the packed and padded elements of the input matrix.

    Raises
    ------
    ONP_ERROR
        If ncols or total_slots are not powers of two, or total_slots is insufficient.
    """

    rows, cols = len(matrix), len(matrix[0])
    ncols = next_power_of_two(ncols)

    if not is_power_of_two(total_slots):
        ONP_ERROR(f"total_slots [{total_slots}] must be a power of two")
    if total_slots % ncols != 0:
        ONP_ERROR("total_slots must be divisible by ncols")

    nrows = next_power_of_two(rows) if is_row_padded else rows
    required_size = nrows * ncols
    shape = nrows, ncols

    if total_slots < required_size:
        ONP_ERROR("Total slots insufficient for the given matrix and padding.")

    flat_array = np.zeros(total_slots)

    index = 0
    repeats = total_slots // required_size

    for _ in range(repeats):
        for i in range(rows):
            flat_array[index : index + cols] = matrix[i]
            index += ncols

        index += (nrows - rows) * ncols

    return flat_array, shape


def _pack_matrix_col_wise(
    matrix, ncols, num_slots, is_padded_rows=None, verbose=0
):
    """Pack a matrix into a flat array column-wise with zero padding.
    [[1 2 3] -> [1 4 7 0 2 5 8 0 3 6 9 0]
     [4 5 6]
     [7 8 9]]
    """
    assert is_power_of_two(ncols)
    assert is_power_of_two(num_slots)
    assert num_slots % ncols == 0

    rows = len(matrix)
    cols = len(matrix[0])

    nrows = next_power_of_two(rows) if is_padded_rows else rows

    if num_slots < ncols * nrows:
        ONP_ERROR(
            f"encrypt_matrix ::: Matrix [{rows} x {cols}]is too big compared with num_slots [{num_slots}]"
        )

    shape = nrows, ncols
    flat_array = np.zeros(num_slots)
    index = 0  # index into vector to write

    for c in range(ncols):
        for r in range(nrows):
            if r < rows and c < cols:
                flat_array[index] = matrix[r][c]
            index += 1

    return flat_array, shape


# === Conversion Functions ===


def reoriginal_shape(vec, total_slots, ncols):
    # convert a vector of an packed_rw_mat to its original matrix
    n_slots = len(vec)
    row = []
    mat = []
    for k in range(n_slots):
        row.append(vec[k])
        if (k + 1) % ncols == 0 and k >= 1:
            mat.append(row)
            row = []
    return mat


def convert_cw_rw(v, block_size, num_slots):
    org_v = v[:block_size]
    vv = _pack_vector_row_wise(org_v, block_size, num_slots)

    if 0:
        wnice_org = [round(x, 3) for x in v[: 2 * block_size]]
        vv_b = vv[: 2 * block_size]
        wnice = [round(x, 3) for x in vv_b]
        print(f"convert \n  {wnice_org}\n->{wnice}")
        print(f"{wnice}")
    return vv


def convert_rw_cw(v, block_size, num_slots):
    org_v = []
    # print(len(v), block_size, num_slots)
    for k in range(block_size):
        org_v.append(v[k * block_size])

    vv = _pack_vector_col_wise(org_v, block_size, num_slots)

    if 0:
        wnice_org = [round(x, 3) for x in v[: 2 * block_size]]
        vv_b = vv[: 2 * block_size]
        wnice = [round(x, 3) for x in vv_b]
        print(f"convert {org_v} to {vv[:block_size]}")
        print(f"{wnice}")
    return vv
