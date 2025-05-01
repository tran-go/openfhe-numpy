import openfhe_matrix
from openfhe_matrix import MatVecEncoding

from openfhe_numpy.tensor import ctArray
from openfhe_numpy.config import MatrixEncoding
from openfhe_numpy.log import FP_ERROR, FP_DEBUG


def matmul_square(context, pub_key, ctmat_A: ctArray, ctmat_B: ctArray) -> ctArray:
    """Encrypted matrix multiplication for square matrices.

    Equivalent to: np.matmul(A, B) or A @ B

    Internally calls EvalMatMulSquare using homomorphic operations, assuming both
    matrices are packed with the same layout and shape.

    Example
    -------
    >>> np.matmul([[1, 2], [3, 4]], [[5, 6], [7, 8]])
    array([[19, 22], [43, 50]])
    """
    ct_product = openfhe_matrix.EvalMatMulSquare(
        context, pub_key, ctmat_A.data, ctmat_B.data, ctmat_A.ncols
    )
    info = ctmat_A.info()
    info[0] = ct_product
    return ctArray(*info)


def multiply(context, ct_a: ctArray, ct_b: ctArray) -> ctArray:
    """Element-wise multiplication: np.multiply(a, b)

    Performs homomorphic multiplication between ciphertext arrays A and B,
    slot-wise, returning a new encrypted array.

    Example
    -------
    >>> np.multiply([1, 2, 3], [4, 5, 6])
    array([4, 10, 18])
    """
    if ct_a.shape != ct_b.shape:
        raise ValueError("Shapes do not match for element-wise multiplication")
    info = ct_a.info()
    info[0] = context.EvalMult(ct_a.data, ct_b.data)
    return ctArray(*info)


def add(context, ct_a, ct_b):
    """Element-wise addition: np.add(a, b)

    Performs encrypted addition over the packed slots of A and B.

    Example
    -------
    >>> np.add([1, 2, 3], [4, 5, 6])
    array([5, 7, 9])
    """
    if ct_a.shape != ct_b.shape:
        raise ValueError("Shapes do not match for element-wise addition")
    info = ct_a.info()
    info[0] = context.EvalAdd(ct_a.data, ct_b.data)
    return ct_a.__class__(*info)


def sub(context, ct_a, ct_b):
    """Element-wise subtraction: np.subtract(a, b)

    Homomorphically subtracts B from A, slot-by-slot.

    Example
    -------
    >>> np.subtract([5, 7, 9], [1, 2, 3])
    array([4, 5, 6])
    """
    if ct_a.shape != ct_b.shape:
        raise ValueError("Shapes do not match for element-wise subtraction")
    info = ct_a.info()
    info[0] = context.EvalSub(ct_a.data, ct_b.data)
    return ct_a.__class__(*info)


def dot(context, pubkey, tensor_a: ctArray, tensor_b: ctArray) -> ctArray:
    """Dot product over encrypted vectors or matrices.

    Equivalent to: np.dot(a, b)

    Homomorphically multiplies A and B, then reduces across columns using EvalSumCols.

    Example
    -------
    >>> np.dot([1, 2, 3], [4, 5, 6])
    32
    """
    if tensor_a.shape != tensor_b.shape:
        raise ValueError("Shapes do not match for dot product")

    info = tensor_a.info()

    if tensor_a.ndim == 1:
        info[0] = context.EvalInnerProduct(tensor_a, tensor_b, tensor_a.ncols)
    else:
        info[0] = matmul_square(context, pubkey, tensor_a, tensor_b)

    return ctArray(*info)


def matvec(context, keys, sum_col_keys, ctmat: ctArray, ctvec: ctArray, block_size: int) -> ctArray:
    """Matrix-vector multiplication over encrypted data.

    Equivalent to: np.dot(A, v)

    Depending on encoding order of inputs (row vs column), uses optimized homomorphic
    packing routines to multiply matrix with vector.

    Example
    -------
    >>> np.dot([[1, 2], [3, 4]], [5, 6])
    array([17, 39])
    """
    if (
        ctmat.encoding_order == MatrixEncoding.ROW_MAJOR
        and ctvec.encoding_order == MatrixEncoding.COL_MAJOR
    ):
        ct_product = openfhe_matrix.EvalMultMatVec(
            context, sum_col_keys, MatVecEncoding.MM_CRC, block_size, ctvec.data, ctmat.data
        )
        rows, _ = ctmat.original_shape
        return ctArray(
            ct_product,
            (rows, 1),
            False,
            ctmat.batch_size,
            ctmat.ncols,
            MatrixEncoding.COL_MAJOR,
        )

    elif (
        ctmat.encoding_order == MatrixEncoding.COL_MAJOR
        and ctvec.encoding_order == MatrixEncoding.ROW_MAJOR
    ):
        ct_product = openfhe_matrix.EvalMultMatVec(
            context, sum_col_keys, MatVecEncoding.MM_RCR, block_size, ctvec.data, ctmat.data
        )
        rows, _ = ctmat.original_shape
        return ctArray(
            ct_product,
            (rows, 1),
            False,
            ctmat.batch_size,
            ctmat.ncols,
            MatrixEncoding.ROW_MAJOR,
        )

    else:
        raise ValueError(
            "Encoding styles of matrix and vector must be complementary (ROW_MAJOR/COL_MAJOR or vice versa)."
        )


def matrix_power(context, keys, k: int, ctarray: ctArray) -> ctArray:
    """Exponentiate a matrix to power k using homomorphic multiplication.

    Equivalent to: np.linalg.matrix_power(A, k)

    Applies repeated encrypted matrix multiplications.

    Example
    -------
    >>> np.linalg.matrix_power([[2, 0], [0, 2]], 3)
    array([[8, 0], [0, 8]])
    """
    result = ctarray
    for _ in range(k):
        result = matmul_square(context, keys, result, ctarray)
    return result


def add_reduce(context, sumkey, ctarray: ctArray, axis=None):
    """
    - axis=0, the function sums over rows; it adds up values column-wise.
    - axis=1, the function sums over columns; it adds up values row-wise.
    this function is equivalently with np.sum(a, axis)
    Example
    -------
    >>> np.add.reduce([[1, 2], [3, 4]], axis=0)
    array([4, 6])
    >>> np.add.reduce([[1, 2], [3, 4]], axis=1)
    array([3, 7])
    """
    if axis == 0:
        ct_result = context.EvalSumRows(ctarray.data, ctarray.ncols, sumkey)
    elif axis == 1:
        ct_result = context.EvalSumCols(ctarray.data, ctarray.ncols, sumkey)
    else:
        FP_ERROR(f"Axis {axis} is out of bound for array of dimension {ctarray.ndim}")

    info = ctarray.info()
    info[0] = ct_result
    return ctArray(*info)


def add_accumulate(context, sumkey, ctarray: ctArray, axis=None):
    """Homomorphic prefix sum: equivalent to np.add.accumulate(cta, axis=axis).

    Would be implemented via encrypted rotations + additions.

    Example
    -------
    >>> np.add.accumulate([1, 2, 3, 4])
    Return: array([1, 3, 6, 10])
    """
    info = ctarray.info()    
    info[0] = ctarray.data
    for i in range(ctarray.ncols):
        rotated = context.EvalRotate(info[0].data, i)
        info[0] = context.EvalAdd(rotated 


def sub_reduce(context, sum_keys, ctarray: ctArray, axis=None):
    """Homomorphic reduction with subtraction: np.subtract.reduce

    Placeholder for sequential subtraction across axis.

    Example
    -------
    >>> np.subtract.reduce([10, 1, 2])
    7
    """
    pass


def sub_accumulate(context, keys, ctarray: ctArray, axis=None):
    """Homomorphic prefix subtraction: np.subtract.accumulate.

    Example
    -------
    >>> np.subtract.accumulat([10, 1, 2])
    array([10, 9, 7])
    """
    pass


def transpose(context, public_key, ctarray: ctArray) -> ctArray:
    """Transpose a matrix
    Equivalent to: np.transpose()
    """
    ct_data = openfhe_matrix.EvalMatrixTranspose(context, public_key, ctarray.data, ctarray.ncols)
    info = ctarray.info()
    info[0] = ct_data
    return ctArray(*info)
