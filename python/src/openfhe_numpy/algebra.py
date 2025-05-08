# from multimethod import multimethod
import numpy as np

import openfhe_matrix
from openfhe_matrix import MatVecEncoding

from openfhe_numpy.tensor import BaseTensor, CTArray
from openfhe_numpy.config import MatrixOrder
from openfhe_numpy.log import FP_ERROR, FP_DEBUG


def square_matmul(matrix_A: CTArray, matrix_B: CTArray) -> BaseTensor:
    """Encrypted matrix multiplication for square matrices.

    Equivalent to: np.matmul(A, B) or A @ B

    Internally calls EvalMatMulSquare using homomorphic operations, assuming both
    matrices are packed with the same layout and shape.

    Numpy Example
    -------
    >>> np.matmul([[1, 2], [3, 4]], [[5, 6], [7, 8]])
    array([[19, 22], [43, 50]])
    """
    if matrix_A.ndim == matrix_B.ndim:
        if matrix_A.shape == matrix_B.shape:
            info = matrix_A.info()
            info[0] = openfhe_matrix.EvalMatMulSquare(
                matrix_A.data, matrix_B.data, matrix_A.rowsize
            )
            return CTArray(*info)
    else:
        FP_ERROR(
            f"Matrix dimension mismatch for multiplication: {matrix_A.shape} and {matrix_B.shape}"
        )


# @add.register
# def _(matrix: (CTArray | Plaintext), vector: CTArray | Plaintext)) -> BaseTensor:
#     return CiphertextBaseTensor(eval_add(a.data, Plaintext(b.data)), shape=a.shape)


def multiply(tensor_a: CTArray, tensor_b: CTArray) -> BaseTensor:
    """Element-wise multiplication: np.multiply(a, b)

    Performs homomorphic multiplication between ciphertext arrays A and B,
    slot-wise, returning a new encrypted array.

    Numpy Example
    -------
    >>> np.multiply([1, 2, 3], [4, 5, 6])
    array([4, 10, 18])
    """
    if tensor_a.shape != tensor_b.shape:
        FP_ERROR("Shape does not match for element-wise multiplication")
    crypto_context = tensor_a.data.GetCryptoContext()
    info = tensor_a.info()
    info[0] = crypto_context.EvalMult(tensor_a.data, tensor_b.data)
    return CTArray(*info)


def add(tensor_a: CTArray, tensor_b: CTArray):
    """Element-wise addition: np.add(a, b)

    Performs encrypted addition over the packed slots of A and B.

    Numpy Example
    -------
    >>> np.add([1, 2, 3], [4, 5, 6])
    array([5, 7, 9])
    """
    if tensor_a.shape != tensor_b.shape:
        FP_ERROR("Shape does not match for element-wise addition")

    crypto_context = tensor_a.data.GetCryptoContext()
    info = tensor_a.info()
    print(tensor_a)
    print(tensor_b)

    info[0] = crypto_context.EvalAdd(tensor_a.data, tensor_b.data)
    return CTArray(*info)


def sub(tensor_a: CTArray, tensor_b: CTArray):
    """Element-wise subtraction: np.subtract(a, b)

    Homomorphically subtracts B from A, slot-by-slot.

    Numpy Example
    -------
    >>> np.subtract([5, 7, 9], [1, 2, 3])
    array([4, 5, 6])
    """
    if tensor_a.shape != tensor_b.shape:
        raise ValueError("Shapes do not match for element-wise subtraction")
    crypto_context = tensor_a.data.GetCryptoContext()
    info = tensor_a.info()
    info[0] = crypto_context.EvalSub(tensor_a.data, tensor_b.data)
    return BaseTensor(*info)


def dot(tensor_a: CTArray, tensor_b: CTArray) -> CTArray:
    """Dot product over encrypted vectors or matrices.

    Equivalent to: np.dot(a, b)

    Homomorphically multiplies A and B, then reduces across columns using EvalSumCols.

    Numpy Example
    -------
    >>> np.dot([1, 2, 3], [4, 5, 6])
    32
    """
    if tensor_a.shape != tensor_b.shape:
        raise ValueError("Shapes do not match for dot product")

    crypto_context = tensor_a.data.GetCryptoContext()
    info = tensor_a.info()
    if tensor_a.ndim == 1:
        info[0] = crypto_context.EvalInnerProduct(tensor_a, tensor_b, tensor_a.rowsize)
    else:
        info[0] = square_matmul(tensor_a, tensor_b)

    return BaseCTArrayTensor(*info)


def matvec(
    crypto_context, keys, sumkey, tensor_matrix: CTArray, tensor_vector: CTArray, rowsize: int
) -> BaseTensor:
    """Matrix-vector multiplication over encrypted data.

    Equivalent to: np.dot(A, v)

    Depending on encoding order of inputs (row vs column), uses optimized homomorphic
    packing routines to multiply matrix with vector.

    Numpy Example
    -------
    >>> np.dot([[1, 2], [3, 4]], [5, 6])
    array([17, 39])
    """
    if (
        tensor_matrix.order == MatrixOrder.ROW_MAJOR
        and tensor_vector.order == MatrixOrder.COL_MAJOR
    ):
        ct_product = openfhe_matrix.EvalMultMatVec(
            sumkey,
            MatVecEncoding.MM_CRC,
            rowsize,
            tensor_vector.data,
            tensor_matrix.data,
        )
        rows, _ = tensor_matrix.original_shape

        return CTArray(
            ct_product,
            (rows, 1),
            tensor_matrix.batch_size,
            tensor_matrix.rowsize,
            MatrixOrder.COL_MAJOR,
        )

    elif (
        tensor_matrix.order == MatrixOrder.COL_MAJOR
        and tensor_vector.order == MatrixOrder.ROW_MAJOR
    ):
        ct_product = openfhe_matrix.EvalMultMatVec(
            crypto_context,
            sumkey,
            MatVecEncoding.MM_RCR,
            rowsize,
            tensor_vector.data,
            tensor_matrix.data,
        )
        rows, _ = tensor_matrix.original_shape
        return CTArray(
            ct_product,
            (rows, 1),
            tensor_matrix.batch_size,
            tensor_matrix.rowsize,
            MatrixOrder.ROW_MAJOR,
        )

    else:
        FP_ERROR(
            "Encoding styles of matrix and vector must be complementary (ROW_MAJOR/COL_MAJOR or vice versa)."
        )


def matrix_power(crypto_context, keys, k: int, ctarray: CTArray) -> BaseTensor:
    """Exponentiate a matrix to power k using homomorphic multiplication.

    Equivalent to: np.linalg.matrix_power(A, k)

    Applies repeated encrypted matrix multiplications.

    Numpy Example
    -------
    >>> np.linalg.matrix_power([[2, 0], [0, 2]], 3)
    array([[8, 0], [0, 8]])
    """
    result = ctarray
    for _ in range(k):
        result = square_matmul(result, ctarray)
    return result


def add_reduce(crypto_context, sumkey, ctarray: CTArray, axis=None):
    """
    - axis=0, the function sums over rows; it adds up values column-wise.
    - axis=1, the function sums over columns; it adds up values row-wise.
    this function is equivalently with np.sum(a, axis)
    Numpy Example
    -------
    >>> np.add.reduce([[1, 2], [3, 4]], axis=0)
    array([4, 6])
    >>> np.add.reduce([[1, 2], [3, 4]], axis=1)
    array([3, 7])
    """
    if axis == 0:
        ct_result = crypto_context.EvalSumRows(ctarray.data, ctarray.rowsize, sumkey)
    elif axis == 1:
        ct_result = crypto_context.EvalSumCols(ctarray.data, ctarray.rowsize, sumkey)
    else:
        FP_ERROR(f"Axis {axis} is out of bound for array of dimension {ctarray.ndim}")

    info = ctarray.info()
    info[0] = ct_result
    return CTArray(*info)


def add_accumulate(crypto_context, sumkey, ctarray: CTArray, axis=None):
    """Homomorphic prefix sum: equivalent to np.add.accumulate(cta, axis=axis).

    Would be implemented via encrypted rotations + additions.

    Numpy Example
    -------
    >>> np.add.accumulate([1, 2, 3, 4])
    Return: array([1, 3, 6, 10])
    """

    pass


def sub_reduce(crypto_context, sum_keys, ctarray: CTArray, axis=None):
    """Homomorphic reduction with subtraction: np.subtract.reduce

    Placeholder for sequential subtraction across axis.

    Numpy Example
    -------
    >>> np.subtract.reduce([10, 1, 2])
    7
    """
    pass


def sub_accumulate(crypto_context, keys, ctarray: CTArray, axis=None):
    """Homomorphic prefix subtraction: np.subtract.accumulate.

    Numpy Example
    -------
    >>> np.subtract.accumulat([10, 1, 2])
    array([10, 9, 7])
    """
    pass


def transpose(ctarray: CTArray) -> BaseTensor:
    """Transpose a matrix
    Equivalent to: np.transpose()
    """
    ct_data = openfhe_matrix.EvalTranspose(ctarray.data, ctarray.rowsize)
    info = ctarray.info()
    info[0] = ct_data
    return CTArray(*info)
