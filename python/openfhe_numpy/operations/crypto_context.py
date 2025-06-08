"""
Crypto context operations and key generation utilities for homomorphic operations.

This module provides functions for generating rotation, accumulation, and other specialized
keys needed for various homomorphic operations in OpenFHE-NumPy.
"""

from openfhe_numpy import _openfhe_numpy  # Import from parent package


def accumulation_depth(nrows, ncols, accumulate_by_rows):
    """
    Compute the CKKS multiplicative depth needed to sum over a matrix.

    Parameters
    ----------
    nrows : int
        Number of rows in the matrix
    ncols : int
        Number of columns in the matrix
    accumulate_by_rows : bool
        Whether to sum over rows or columns

    Returns
    -------
    int
        Required multiplicative depth
    """
    return _openfhe_numpy.MulDepthAccumulation(nrows, ncols, accumulate_by_rows)


def sum_row_keys(secret_key, ncols=0):
    """
    Generate keys for summing rows in a matrix.

    Parameters
    ----------
    context : CryptoContext
        The OpenFHE crypto context
    secret_key : PrivateKey
        The private key to use for key generation
    ncols : int, optional
        Number of cols for the matrix, by default 0

    Returns
    -------
    object
        Generated sum keys
    """
    context = secret_key.GetCryptoContext()
    return context.EvalSumRowsKeyGen(secret_key, None, ncols)


def sum_col_keys(secret_key, ncols=0):
    """
    Generate keys for summing columns in a matrix.

    Parameters
    ----------
    context : CryptoContext
        The OpenFHE crypto context
    secret_key : PrivateKey
        The private key to use for key generation
    ncols : int, optional
        Number of columns in the matrix, by default 0
    """
    context = secret_key.GetCryptoContext()
    return context.EvalSumColsKeyGen(secret_key)


def gen_accumulate_rows_key(secret_key, ncols):
    """
    Generate keys for cumulative sum of rows in a matrix.

    Parameters
    ----------
    secret_key : PrivateKey
        The private key to use for key generation
    ncols : int
        Number of columns in the matrix
    """
    _openfhe_numpy.EvalSumCumRowsKeyGen(secret_key, ncols)


def gen_accumulate_cols_key(secret_key, ncols):
    """
    Generate keys for cumulative sum of columns in a matrix.

    Parameters
    ----------
    secret_key : PrivateKey
        The private key to use for key generation
    ncols : int
        Number of columns in the matrix
    """
    _openfhe_numpy.EvalSumCumColsKeyGen(secret_key, ncols)


def gen_rotation_keys(secret_key, rotation_indices):
    """
    Generate rotation keys for the specified indices.

    Parameters
    ----------
    context : CryptoContext
        The OpenFHE crypto context
    secret_key : PrivateKey
        The private key to use for key generation
    rotation_indices : list
        List of rotation indices to generate keys for
    """
    context = secret_key.GetCryptoContext()
    context.EvalRotateKeyGen(secret_key, rotation_indices)


def gen_lintrans_keys(secret_key, block_size, linear_transform_type, repetitions=0):
    """
    Generate keys for linear transformations.

    Parameters
    ----------
    secret_key : PrivateKey
        The private key to use for key generation
    block_size : int
        linear_transform_type size for the matrix
    linear_transform_type : LinTransType
        Type of linear transformation
    repetitions : int, optional
        Number of repetitions, by default 0
    """
    _openfhe_numpy.EvalLinTransKeyGen(secret_key, block_size, linear_transform_type, repetitions)


def gen_square_matmult_key(secret_key, block_size):
    """
    Generate keys for square matrix multiplication.

    Parameters
    ----------
    secret_key : PrivateKey
        The private key to use for key generation
    block_size : int
        Block size for the matrix
    """
    _openfhe_numpy.EvalSquareMatMultRotateKeyGen(secret_key, block_size)


def gen_transpose_keys(secret_key, ctm_matrix):
    """
    Generate keys for matrix transposition.

    Parameters
    ----------
    secret_key : PrivateKey
        The private key to use for key generation
    ctm_matrix : CTArray
        The ciphertext matrix to transpose
    """
    _openfhe_numpy.EvalLinTransKeyGen(
        secret_key, ctm_matrix.ncols, _openfhe_numpy.LinTransType.TRANSPOSE
    )
