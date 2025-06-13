"""Packing utilities for OpenFHE-NumPy.

This module provides functions to pack vectors and matrices into flat arrays
with specific layouts (row-wise or column-wise) and padding. These functions
are essential for homomorphic encryption operations that require specific
data layouts for efficient computation.

The module is organized into sections:
1. Shape and utility functions
2. Vector packing functions
3. Matrix packing functions
4. Conversion functions between different packing formats
"""

from typing import Union, List, Tuple

import numpy as np

from .log import ONP_ERROR
from .matlib import is_power_of_two, next_power_of_two


# === Shape and Utility Functions ==


def _get_shape(data: Union[List, Tuple, np.ndarray]) -> Tuple[int, int, int]:
    """Determine the shape and dimension of a given array.

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

    ONP_ERROR(f"Invalid data type ({type(data)}) provided. Must be list, tuple, or ndarray.")


# === Vector Packing Functions ==


def _pack_vector_row_wise(v, total_slots, repeats=0, pad_to_power_of_2=True, padding_value="default"):
    """Pack a vector by repeating each element multiple times in a row-wise pattern.

    This function repeats each element of the input vector a specified number of times,
    arranging them in a row-wise pattern.

    Parameters
    ----------
    v : array_like
        The input vector to pack.
    total_slots : int
        Total number of slots available in the output array; must be a power of two.
    repeats : int
        Number of times to repeat each element. If pad_to_power_of_2 is True,
        this will be rounded up to the next power of two.
    pad_to_power_of_2 : bool, optional
        If True, round the repeats to the next power of two. Default is True.
    padding_value : str, optional
        How to pad values. "default" repeats the value, "0" padding zero. Default is "default".

    Returns
    -------
    ndarray
        Packed array with repeated elements.

    Raises
    ------
    ONP_ERROR
        If total_slots is not a power of two or if the packed vector would be too large.

    Example
    -------
    >>> _pack_vector_row_wise([1,2,3], 12, 3, True, "default")
    [1,1,1,1,2,2,2,2,3,3,3,3]
    >>> _pack_vector_row_wise([1,2,3], 12, 3, True, "0")
    [1,1,1,0,2,2,2,0,3,3,3,0]
    """
    if not is_power_of_two(total_slots):
        ONP_ERROR(f"Total_slots [{total_slots}] must be a power of two")

    n = len(v)
    size = n

    if repeats == 0:
        size = next_power_of_two(n)
        repeats = total_slots // size

    if pad_to_power_of_2:
        repeats = next_power_of_two(repeats)

    if total_slots < (repeats * n):
        ONP_ERROR(f"Padded vector [{repeats * size}] is longer than the total slots [{total_slots}]")

    packed = np.zeros(total_slots)
    shape = (size, repeats)

    if padding_value == "default":
        for i in range(n):
            start = i * repeats
            packed[start : start + repeats] = v[i]
    elif padding_value == "0":
        for i in range(n):
            start = i * repeats
            packed[start : start + n] = v[i]
    else:
        ONP_ERROR(f"Padding value [{padding_value}] is invalid")
    return packed, shape


def _pack_vector_col_wise(v, total_slots, repeats=0, pad_to_power_of_2=True):
    """Pack a vector by repeating it multiple times in a column-wise pattern.

    This function repeats the entire vector multiple times, arranging them in blocks.
    For example, packing [1,2,3] into 12 slots would result in [1,2,3,0,1,2,3,0,1,2,3,0]
    where 0 is padding if size is padded to power of two.

    Parameters
    ----------
    v : array_like
        The input vector to pack.
    total_slots : int
        Total number of slots available in the output array; must be a power of two.
    repeats : int, optional
        Number of times to repeat the vector. If -1, calculated automatically. Default is -1.
    pad_to_power_of_2 : bool, optional
        If True, pad the vector size to the next power of two. Default is True.

    Returns
    -------
    ndarray
        Packed array with the vector repeated multiple times.

    Raises
    ------
    ONP_ERROR
        If total_slots is not a power of two, or if the padded size is smaller than
        the vector size, or if total_slots is not divisible by size.

    Example
    -------
    >>> _pack_vector_col_wise([1,2,3], 12)
    [1,2,3,0,1,2,3,0,1,2,3,0]  # assuming pad_to_power_of_2=True which makes size=4
    """
    if not is_power_of_two(total_slots):
        ONP_ERROR(f"Total_slots [{total_slots}] must be a power of two")

    n = len(v)
    size = n

    if pad_to_power_of_2:
        size = next_power_of_two(size)
    if repeats == 0:
        repeats = total_slots // size
    shape = (size, repeats)

    if size < n:
        ONP_ERROR(f"Padded size [{size}] is smaller than vector size [{n}]")
    if total_slots % size != 0:
        ONP_ERROR(f"Total_slots [{total_slots}] must be divisible by size [{size}]")
    if total_slots < (repeats * size):
        ONP_ERROR(f"Padded vector [{repeats * size}] is longer than the total slots [{total_slots}]")

    packed = np.zeros(total_slots)

    k = 0
    for i in range(repeats):
        for j in range(n):
            packed[k] = v[j]
            k += 1
        k += size - n

    return packed, shape


# === Matrix Packing Functions ===


def _pack_matrix_row_wise(matrix, total_slots, pad_to_power_of_2=True, repeats=0):
    """Pack a matrix into a flat array row-wise with zero padding.

    This function packs a matrix row by row, padding each row to a power of two length
    if specified. For example:
    [[1,2,3],  ->  [1,2,3,0,4,5,6,0,7,8,9,0]
     [4,5,6],
     [7,8,9]]

    Parameters
    ----------
    matrix : array_like
        The input 2D matrix to be packed.
    total_slots : int
        Total number of slots available in the output array; must be a power of two.
    pad_to_power_of_2 : bool, optional
        If True, pad the number of rows and columns to the next power of two. Default is True.
    repeats : int, optional
        Number of times to repeat the matrix. If 0, calculated automatically. Default is 0.

    Returns
    -------
    tuple
        A tuple containing:
        - flat_array: ndarray containing the packed and padded elements of the input matrix
        - shape: tuple (nrows, ncols) indicating the logical shape of the packed matrix

    Raises
    ------
    ONP_ERROR
        If total_slots is not a power of two, or if total_slots is not divisible by
        the required size, or if total_slots is insufficient for the padded matrix.
    """
    if not is_power_of_two(total_slots):
        ONP_ERROR(f"Total_slots [{total_slots}] must be a power of two")

    rows, cols = len(matrix), len(matrix[0])
    nrows, ncols = rows, cols
    if pad_to_power_of_2:
        ncols = next_power_of_two(cols)
        nrows = next_power_of_two(rows)

    required_size = nrows * ncols
    shape = nrows, ncols

    if repeats == 0:
        repeats = total_slots // required_size

    if total_slots % required_size != 0:
        ONP_ERROR(f"Total_slots [{total_slots}] must be divisible by required_size")

    if total_slots < required_size:
        ONP_ERROR(f"Total_slots [{total_slots}] is insufficient for padding  [{required_size * repeats}].")

    flat_array = np.zeros(total_slots)
    index = 0

    for _ in range(repeats):
        for i in range(rows):
            flat_array[index : index + cols] = matrix[i]
            index += ncols
        index += (nrows - rows) * ncols

    return flat_array, shape


def _pack_matrix_col_wise(matrix, total_slots, pad_to_power_of_2=True, repeats=0):
    """Pack a matrix into a flat array column-wise with zero padding.

    This function packs a matrix column by column, padding each column to a power of two length
    if specified. For example:
    [[1,2,3],  ->  [1,4,7,0,2,5,8,0,3,6,9,0]
     [4,5,6],
     [7,8,9]]

    Parameters
    ----------
    matrix : array_like
        The input 2D matrix to be packed.
    total_slots : int
        Total number of slots available in the output array; must be a power of two.
    pad_to_power_of_2 : bool, optional
        If True, pad the number of rows and columns to the next power of two. Default is True.
    repeats : int, optional
        Number of times to repeat the matrix. If 0, calculated automatically. Default is 0.

    Returns
    -------
    tuple
        A tuple containing:
        - flat_array: ndarray containing the packed and padded elements of the input matrix
        - shape: tuple (nrows, ncols) indicating the logical shape of the packed matrix

    Raises
    ------
    ONP_ERROR
        If total_slots is not a power of two, or if total_slots is not divisible by
        the required size, or if total_slots is insufficient for the padded matrix.
    """
    if not is_power_of_two(total_slots):
        ONP_ERROR(f"Total_slots [{total_slots}] must be a power of two")

    rows, cols = len(matrix), len(matrix[0])
    nrows, ncols = rows, cols

    if pad_to_power_of_2:
        ncols = next_power_of_two(cols)
        nrows = next_power_of_two(rows)

    required_size = ncols * nrows
    shape = nrows, ncols

    if repeats == 0:
        repeats = total_slots // required_size

    if total_slots % required_size != 0:
        ONP_ERROR(f"Total_slots [{total_slots}] must be divisible by required_size")

    if total_slots < required_size:
        ONP_ERROR(f"Total_slots [{total_slots}] is insufficient for padding  [{required_size * repeats}].")

    flat_array = np.zeros(total_slots)
    index = 0

    for c in range(ncols):
        for r in range(nrows):
            if r < rows and c < cols:
                flat_array[index] = matrix[r][c]
            index += 1

    return flat_array, shape


# === Conversion Functions ===


def reoriginal_shape(vec, total_slots, ncols):
    """Convert a packed vector back to its original matrix shape.

    Parameters
    ----------
    vec : array_like
        The packed vector to reshape into a matrix.
    total_slots : int
        Total number of slots in the packed vector.
    ncols : int
        Number of columns in the target matrix.

    Returns
    -------
    list
        A 2D list representing the reshaped matrix.
    """
    n_slots = len(vec)
    row = []
    mat = []
    for k in range(n_slots):
        row.append(vec[k])
        if (k + 1) % ncols == 0 and k >= 1:
            mat.append(row)
            row = []
    return mat


def vector_col_major_2_row_major(v, block_size, total_slots):
    """Convert a column-wise packed vector to row-wise packing.

    Parameters
    ----------
    v : array_like
        The column-wise packed vector to convert.
    block_size : int
        Size of each block in the packed vector.
    total_slots : int
        Total number of slots in the output vector.

    Returns
    -------
    ndarray
        The vector repacked in row-wise format.
    """
    org_v = v[:block_size]
    vv = _pack_vector_row_wise(org_v, block_size, total_slots)

    if 0:
        wnice_org = [round(x, 3) for x in v[: 2 * block_size]]
        vv_b = vv[: 2 * block_size]
        wnice = [round(x, 3) for x in vv_b]
        print(f"convert \n  {wnice_org}\n->{wnice}")
        print(f"{wnice}")
    return vv


def vector_row_major_2_col_major(v, block_size, total_slots):
    """Convert a row-wise packed vector to column-wise packing.

    Parameters
    ----------
    v : array_like
        The row-wise packed vector to convert.
    block_size : int
        Size of each block in the packed vector.
    total_slots : int
        Total number of slots in the output vector.

    Returns
    -------
    ndarray
        The vector repacked in column-wise format.
    """
    org_v = []
    # print(len(v), block_size, total_slots)
    for k in range(block_size):
        org_v.append(v[k * block_size])

    vv = _pack_vector_col_wise(org_v, block_size, total_slots)

    if 0:
        wnice_org = [round(x, 3) for x in v[: 2 * block_size]]
        vv_b = vv[: 2 * block_size]
        wnice = [round(x, 3) for x in vv_b]
        print(f"convert {org_v} to {vv[:block_size]}")
        print(f"{wnice}")
    return vv
