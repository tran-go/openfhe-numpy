"""
Arithmetic operations for homomorphic encryption tensors.

This module implements the core arithmetic operations for encrypted tensors
using the OpenFHE library. Operations include addition, subtraction, multiplication,
matrix multiplication, and other mathematical operations.
"""

# Standard library imports
from typing import Optional, Union, List, Tuple

# Third-party imports
from numpy.typing import ArrayLike

# Project imports
from openfhe_numpy.operations.dispatch import register_tensor_function
from openfhe_numpy.tensor.ctarray import CTArray
from openfhe_numpy.tensor.ptarray import PTArray
from openfhe_numpy.utils.log import ONP_ERROR


# Import specific functions from C++ module
from openfhe_numpy._onp_cpp import *

##############################################################################
# BASIC ARITHMETIC OPERATIONS
##############################################################################


# ------------------------------------------------------------------------------
# Addition Operations
# ------------------------------------------------------------------------------
def _eval_add(lhs, rhs):
    """Internal function to evaluate addition between encrypted tensors."""
    crypto_context = lhs.data.GetCryptoContext()

    if isinstance(rhs, (int, float)):
        rhs = crypto_context.MakeCKKSPackedPlaintext([rhs] * lhs.batch_size)

    result = crypto_context.EvalAdd(lhs.data, rhs)
    return CTArray(result, lhs.original_shape, lhs.batch_size, lhs.shape, lhs.order)


@register_tensor_function("add", [("CTArray", "CTArray"), ("CTArray", "PTArray")])
def add_ct(a, b):
    """Add two tensors."""
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    return _eval_add(a, b.data)


@register_tensor_function("add", ("CTArray", "scalar"))
def add_ct_scalar(a, scalar):
    """Add a scalar to a tensor."""
    return _eval_add(a, scalar)


@register_tensor_function("add", [("BlockCTArray", "BlockCTArray"), ("BlockCTArray", "BlockPTArray")])
def add_block_ct(a, b):
    """Add two block tensors."""
    raise NotImplementedError("BlockPTArray and BlockCTArray addition not implemented yet.")


@register_tensor_function("add", [("BlockCTArray", "scalar")])
def add_block_ct_scalar(a, scalar):
    """Add a scalar to a block tensor."""
    raise NotImplementedError("BlockPTArray and scalar addition not implemented yet.")


# ------------------------------------------------------------------------------
# Subtraction Operations
# ------------------------------------------------------------------------------
def _eval_sub(lhs, rhs):
    """Internal function to evaluate subtraction between encrypted tensors."""
    crypto_context = rhs.data.GetCryptoContext() if rhs.is_ciphertext else lhs.data.GetCryptoContext()

    if isinstance(rhs, (int, float)):
        rhs = crypto_context.MakeCKKSPackedPlaintext([rhs] * lhs.batch_size)
    else:
        rhs = rhs.data

    result = crypto_context.EvalSub(lhs.data, rhs)
    return lhs.clone(result)


@register_tensor_function(
    "subtract",
    [("CTArray", "CTArray"), ("CTArray", "PTArray"), ("PTArray", "CTArray")],
)
def subtract_ct(a, b):
    """Subtract two tensors."""
    if a.shape != b.shape:
        ONP_ERROR("Shape does not match for element-wise subtraction")
    return _eval_sub(a, b)


@register_tensor_function("subtract", [("CTArray", "scalar"), ("scalar", "CTArray")])
def subtract_ct_scalar(a, b):
    """Subtract a scalar from a tensor or vice versa."""
    return _eval_sub(a, b.data)


# ------------------------------------------------------------------------------
# Multiplication Operations
# ------------------------------------------------------------------------------
def _eval_multiply(lhs, rhs):
    """Internal function to evaluate element-wise multiplication."""
    crypto_context = lhs.data.GetCryptoContext()
    if isinstance(rhs, (int, float)):
        rhs = crypto_context.MakeCKKSPackedPlaintext([rhs] * lhs.batch_size)
    else:
        rhs = rhs.data

    result = crypto_context.EvalMul(lhs.data, rhs)
    return lhs.clone(result)


@register_tensor_function("multiply", [("CTArray", "CTArray"), ("CTArray", "PTArray")])
def multiply_ct(a, b):
    """Multiply two tensors element-wise."""
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    return _eval_multiply(a, b.data)


@register_tensor_function("multiply", ("CTArray", "scalar"))
def multiply_ct_scalar(a, scalar):
    """Multiply a tensor by a scalar."""
    return _eval_multiply(a, scalar)


@register_tensor_function(
    "multiply",
    [("BlockCTArray", "BlockCTArray"), ("BlockCTArray", "BlockPTArray")],
)
def multiply_block_ct(a, b):
    """Multiply two block tensors element-wise."""
    raise NotImplementedError("BlockPTArray multiplication not implemented yet.")


@register_tensor_function("multiply", [("BlockCTArray", "scalar")])
def multiply_block_ct_scalar(a, scalar):
    """Multiply a block tensor by a scalar."""
    raise NotImplementedError("BlockPTArray and scalar multiplication not implemented yet.")


##############################################################################
# MATRIX OPERATIONS
##############################################################################


# ------------------------------------------------------------------------------
# Matrix Multiplication Operations
# ------------------------------------------------------------------------------


def _eval_matvec_ct(lhs, rhs):
    """Internal function to evaluate matrix-vector multiplication."""
    if lhs.ndim == 2 and rhs.ndim == 1:
        if lhs.original_shape[1] != rhs.original_shape[0]:
            ONP_ERROR(f"Matrix dimension [{lhs.original_shape}] mismatch with vector dimension [{rhs.shape}]")
        if lhs.order == ArrayEncodingType.ROW_MAJOR and rhs.order == ArrayEncodingType.COL_MAJOR:
            # print("MM_CRC")
            sumkey = lhs.extra["colkey"]
            ciphertext = EvalMultMatVec(
                sumkey,
                MatVecEncoding.MM_CRC,
                lhs.ncols,
                rhs.data,
                lhs.data,
            )
            return CTArray(
                ciphertext, (lhs.original_shape[0],), lhs.batch_size, (lhs.shape[0],), ArrayEncodingType.ROW_MAJOR
            )

        elif lhs.order == ArrayEncodingType.COL_MAJOR and rhs.order == ArrayEncodingType.COL_MAJOR:
            # print("MM_RCR")
            sumkey = lhs.extra["rowkey"]
            ciphertext = EvalMultMatVec(
                sumkey,
                MatVecEncoding.MM_RCR,
                lhs.ncols,
                rhs.data,
                lhs.data,
            )
            return CTArray(
                ciphertext, (lhs.original_shape[0],), lhs.batch_size, (lhs.shape[0],), ArrayEncodingType.ROW_MAJOR
            )
        else:
            ONP_ERROR(
                f"Encoding styles of matrix ({lhs.order}) and vector ({rhs.order}) must be complementary (ROW_MAJOR/COL_MAJOR or vice versa)."
            )
    else:
        ONP_ERROR(f"Matrix dimension mismatch for multiplication: {lhs.original_shape} and {rhs.original_shape}")


def _matmul_ct(lhs, rhs):
    """Internal function to evaluate matrix multiplication."""
    if lhs.is_encrypted and rhs.is_encrypted:
        if lhs.shape == rhs.shape:
            if rhs.ndim == 1:
                return _eval_matvec_ct(lhs, rhs)
            return lhs.clone(EvalMatMulSquare(lhs.data, rhs.data, lhs.ncols))

        else:
            ONP_ERROR(f"Matrix dimension mismatch for multiplication: {lhs.shape} and {rhs.shape}")


@register_tensor_function("matmul", [("CTArray", "CTArray")])
def matmul_ct(a, b):
    """Perform matrix multiplication between two tensors."""
    return _matmul_ct(a, b)


# ------------------------------------------------------------------------------
# Dot Product Operations
# ------------------------------------------------------------------------------
def _dot(lhs, rhs):
    """Internal function to evaluate dot product."""
    if lhs.ndim == 1 and rhs.ndim == 1:
        crypto_context = lhs.data.GetCryptoContext()
        ciphertext = crypto_context.EvalInnerProduct(lhs.data, rhs.data, lhs.original_shape[0])
        return lhs.clone(ciphertext)
    else:
        return lhs._matmul(rhs)


@register_tensor_function("dot", [("CTArray", "CTArray")])
def dot_ct(a, b):
    """Compute dot product between two tensors."""
    return _dot(a, b)


# ------------------------------------------------------------------------------
# Transpose Operations
# ------------------------------------------------------------------------------
def _transpose_ct(ctarray: CTArray) -> "CTArray":
    """Internal function to evaluate transpose of a tensor."""
    ciphertext = EvalTranspose(ctarray.data, ctarray.ncols)
    pre_padded_shape = (ctarray.original_shape[1], ctarray.original_shape[0])
    padded_shape = (ctarray.shape[1], ctarray.shape[0])

    return CTArray(ciphertext, pre_padded_shape, ctarray.batch_size, padded_shape, ctarray.order)


@register_tensor_function("transpose", [("CTArray",)])
def transpose_ct(a):
    """Transpose a tensor."""
    return _transpose_ct(a)


##############################################################################
# ADVANCED OPERATIONS
##############################################################################


# ------------------------------------------------------------------------------
# Power Operations
# ------------------------------------------------------------------------------
def _pow(tensor, exp: int):
    """Exponentiate a matrix to power k using homomorphic multiplication."""
    if not isinstance(exp, int):
        ONP_ERROR(f"Exponent must be integer, got {type(exp).__name__}")

    if exp < 0:
        ONP_ERROR("Negative exponent not supported in homomorphic encryption")

    if exp == 0:
        # return algebra.eye(tensor))
        pass

    if exp == 1:
        return tensor.clone()

    # Binary exponentiation implementation
    base = tensor.clone()
    result = None

    while exp:
        if exp & 1:
            result = base if result is None else base @ result
        base = base @ base
        exp >>= 1
    return result


@register_tensor_function("pow", [("CTArray", "int")])
def pow_ct(a, exp):
    """Raise a tensor to an integer power."""
    return _pow(a, exp)


@register_tensor_function("pow", [("BlockCTArray", "int")])
def pow_block_ct(a, exp):
    """Raise a block tensor to an integer power."""
    raise NotImplementedError("BlockPTArray power not implemented yet.")


# ------------------------------------------------------------------------------
# Cumulative Sum Operations
# ------------------------------------------------------------------------------
def _cumsum_ct(tensor, axis=0, keepdims=True):
    """
    Compute the cumulative sum of tensor elements along a given axis.

    Parameters
    ----------
    tensor : CTArray
        Input encrypted tensor.
    axis : int, optional
        Axis along which the cumulative sum is computed. Default is 0.
    keepdims : bool, optional
        Whether to keep the dimensions of the original tensor. Default is True.

    Returns
    -------
    CTArray
        A new tensor with cumulative sums along the specified axis.
    """
    if axis not in (0, 1):
        ONP_ERROR("Axis must be 0 or 1 for cumulative sum operation")
    if axis == 0:
        ciphertext = EvalSumCumRows(tensor.data, tensor.ncols, tensor.original_shape[1])
    else:
        ciphertext = EvalSumCumCols(tensor.data, tensor.ncols)
    return tensor.clone(ciphertext)


@register_tensor_function("cumsum", [("CTArray",), ("CTArray", "int"), ("CTArray", "int", "bool")])
def cumsum_ct(a, axis=0, keepdims=True):
    """Compute cumulative sum of a tensor along specified axis."""
    return _cumsum_ct(a, axis, keepdims)


@register_tensor_function("cumsum", [("BlockCTArray", "int"), ("BlockCTArray", "int", "bool")])
def cumsum_block_ct(a, axis=0, keepdims=True):
    """Compute cumulative sum of a block tensor along specified axis."""
    raise NotImplementedError("BlockPTArray cumulative not implemented yet.")


# ------------------------------------------------------------------------------
# Cumulative Reduce Operations
# ------------------------------------------------------------------------------
def _reduce_ct(a, axis=0, keepdims=False):
    """
    Compute the cumulative reduce of tensor elements along a given axis.

    Parameters
    ----------
    a : CTArray
        Input encrypted tensor.
    axis : int, optional
        Axis along which the cumulative reduction is computed. Default is 0.
    keepdims : bool, optional
        Whether to keep the dimensions of the original tensor. Default is False.

    Returns
    -------
    CTArray
        A new tensor with cumulative reduction along the specified axis.
    """
    if axis not in (0, 1):
        ONP_ERROR("Axis must be 0 or 1 for cumulative sum operation")

    if axis == 0:
        ciphertext = EvalReduceCumRows(a.data, a.ncols, a.original_shape[1])
    else:
        ciphertext = EvalReduceCumCols(a.data, a.ncols)
    return a.clone(ciphertext)


@register_tensor_function("cumreduce", [("CTArray", "int", "bool")])
def cumreduce_ct(a, axis=0, keepdims=False):
    """Compute cumulative reduction of a tensor along specified axis."""
    return _reduce_ct(a, axis, keepdims)


@register_tensor_function("cumreduce", [("BlockCTArray", "int")])
def cumreduce_block_ct(a, axis=0, keepdims=False):
    """Compute cumulative reduction of a block tensor along specified axis."""
    raise NotImplementedError("BlockPTArray power not implemented yet.")


# ------------------------------------------------------------------------------
# Sum Operations
# ------------------------------------------------------------------------------


def _ct_sum_matrix(tensor: ArrayLike, axis: Optional[int] = None):
    """
    This function computes a sum of a padded matrix. It is similar to np.sum

    """
    crypto_context = tensor.data.GetCryptoContext()
    rows, cols = tensor.original_shape
    nrows, ncols = tensor.shape
    order = tensor.order

    if axis is None:
        rotated = tensor.data
        ct_sum = tensor.data
        for i in range(nrows * ncols - 1):
            rotated = crypto_context.EvalRotate(rotated, 1)
            ct_sum = crypto_context.EvalAdd(ct_sum, rotated)
        shape = ()
        padded_shape = ()

    elif axis == 0:  # sum over rows
        ct_sum = crypto_context.EvalSumRows(tensor.data, ncols, tensor.extra["rowkey"])
        shape = (cols,)
        padded_shape = (nrows, ncols)
        if tensor.order == ROW_MAJOR:
            order = COL_MAJOR
        elif tensor.order == COL_MAJOR:
            order = COL_MAJOR
        else:
            ONP_ERROR("Not supported!!!")

    elif axis == 1:  # sum over cols
        ct_sum = crypto_context.EvalSumCols(tensor.data, ncols, tensor.extra["colkey"])
        shape = (rows,)
        padded_shape = (nrows, ncols)

    else:
        ONP_ERROR(f"The dimension is invalid axis = {axis}")
    return CTArray(ct_sum, shape, tensor.batch_size, padded_shape, order, tensor.is_padded)


def _ct_sum_vector(tensor: ArrayLike, axis: Optional[int] = None):
    crypto_context = tensor.data.GetCryptoContext()
    rows, cols = tensor.original_shape
    nrows, ncols = tensor.shape

    if axis is not None:
        ONP_ERROR(f"The dimension is invalid axis = {axis}")
    rotated = tensor.data
    ct_sum = tensor.data
    for i in range(nrows * ncols):
        rotated = crypto_context.EvalRotate(rotated, 1)
        ct_sum = crypto_context.EvalAdd(ct_sum, rotated)
    shape, padded_shape = (), ()
    return CTArray(ct_sum, shape, tensor.batch_size, padded_shape, tensor.is_padded)


@register_tensor_function("sum", [("CTArray",), ("CTArray", "int"), ("CTArray", "int", "bool")])
def sum_ct(tensor: ArrayLike, axis: Optional[int] = None, keepdims: bool = False):
    if tensor.ndim == 2:
        return _ct_sum_matrix(tensor, axis)
    elif tensor.ndim == 1:
        return _ct_sum_vector(tensor, axis)


# ------------------------------------------------------------------------------
# Mean Operations
# ------------------------------------------------------------------------------


@register_tensor_function("mean", [("CTArray", "int", "bool")])
def mean_ct(tensor: ArrayLike, axis: Optional[int] = None, keepdims: bool = False):
    cc = tensor.data.GetCryptoContext()
    nrows, ncols = tensor.shape
    n = nrows * ncols
    if axis is None:
        ciphertext = cc.EvalMul(1.0 / n, cc.EvalSum(tensor.data))
        return CTArray(ciphertext, 1, tensor.batch_size)
    elif axis == 0:  # sum over rows
        ciphertext = cc.EvalMul(
            1.0 / nrows,
            cc.EvalSumRows(tensor.data, nrows, tensor.extra["rowkey"]),
        )
        return CTArray(ciphertext, (1, tensor.original_shape[1]), tensor.batch_size)

    elif axis == 1:  # sum over cols
        ciphertext = cc.EvalMul(
            1.0 / ncols,
            cc.EvalSumCols(tensor.data, ncols, tensor.extra["colkey"]),
        )
        return CTArray(ciphertext, (tensor.original_shape[0], 1), tensor.batch_size)
    else:
        ONP_ERROR(f"The dimension is invalid axis = {axis}")
