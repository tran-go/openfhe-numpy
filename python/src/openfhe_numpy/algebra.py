from openfhe_numpy.tensor import ctarray
import openfhe_matrix
from openfhe_numpy.config import PackStyles, MatrixEncoding
from openfhe_matrix import MatVecEncoding


def matmul_square(cc, keys, ct_matrixA: ctarray, ct_matrixB: ctarray) -> ctarray:
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
        cc, keys, ct_matrixA.data, ct_matrixB.data, ct_matrixA.ncols
    )
    info = ct_matrixA.info()
    info[0] = ct_product
    return ctarray(*info)


def multiply(cc, a: ctarray, b: ctarray) -> ctarray:
    """Element-wise multiplication: np.multiply(a, b)

    Performs homomorphic multiplication between ciphertext arrays A and B,
    slot-wise, returning a new encrypted array.

    Example
    -------
    >>> np.multiply([1, 2, 3], [4, 5, 6])
    array([4, 10, 18])
    """
    if a.shape != b.shape:
        raise ValueError("Shapes do not match for element-wise multiplication")
    info = a.info()
    info[0] = cc.EvalMult(a.data, b.data)
    return ctarray(*info)


def add(cc, a, b):
    """Element-wise addition: np.add(a, b)

    Performs encrypted addition over the packed slots of A and B.

    Example
    -------
    >>> np.add([1, 2, 3], [4, 5, 6])
    array([5, 7, 9])
    """
    if a.shape != b.shape:
        raise ValueError("Shapes do not match for element-wise addition")
    info = a.info()
    info[0] = cc.EvalAdd(a.data, b.data)
    return a.__class__(*info)


def sub(cc, a, b):
    """Element-wise subtraction: np.subtract(a, b)

    Homomorphically subtracts B from A, slot-by-slot.

    Example
    -------
    >>> np.subtract([5, 7, 9], [1, 2, 3])
    array([4, 5, 6])
    """
    if a.shape != b.shape:
        raise ValueError("Shapes do not match for element-wise subtraction")
    info = a.info()
    info[0] = cc.EvalSub(a.data, b.data)
    return a.__class__(*info)


def dot(cc, sum_col_keys, a: ctarray, b: ctarray) -> ctarray:
    """Dot product over encrypted vectors or matrices.

    Equivalent to: np.dot(a, b)

    Homomorphically multiplies A and B, then reduces across columns using EvalSumCols.

    Example
    -------
    >>> np.dot([1, 2, 3], [4, 5, 6])
    32
    """
    if a.shape != b.shape:
        raise ValueError("Shapes do not match for dot product")
    ct_mult = cc.EvalMult(a.data, b.data)
    ct_dot = cc.EvalSumCols(ct_mult, a.ncols, sum_col_keys)
    info = a.info()
    info[0] = ct_dot
    return ctarray(*info)


def matvec(
    cc, keys, sum_col_keys, ct_matrix: ctarray, ct_vector: ctarray, block_size: int
) -> ctarray:
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
        ct_matrix.encoding_order == MatrixEncoding.ROW_MAJOR
        and ct_vector.encoding_order == MatrixEncoding.COL_MAJOR
    ):
        ct_product = openfhe_matrix.EvalMultMatVec(
            cc, sum_col_keys, MatVecEncoding.MM_CRC, block_size, ct_vector.data, ct_matrix.data
        )
        rows, _ = ct_matrix.original_shape
        return ctarray(
            ct_product,
            (rows, 1),
            False,
            ct_matrix.batch_size,
            ct_matrix.ncols,
            MatrixEncoding.COL_MAJOR,
        )

    elif (
        ct_matrix.encoding_order == MatrixEncoding.COL_MAJOR
        and ct_vector.encoding_order == MatrixEncoding.ROW_MAJOR
    ):
        ct_product = openfhe_matrix.EvalMultMatVec(
            cc, sum_col_keys, MatVecEncoding.MM_RCR, block_size, ct_vector.data, ct_matrix.data
        )
        rows, _ = ct_matrix.original_shape
        return ctarray(
            ct_product,
            (rows, 1),
            False,
            ct_matrix.batch_size,
            ct_matrix.ncols,
            MatrixEncoding.ROW_MAJOR,
        )

    else:
        raise ValueError(
            "Encoding styles of matrix and vector must be complementary (ROW_MAJOR/COL_MAJOR or vice versa)."
        )


def matrix_power(cc, keys, k: int, ct_matrixA: ctarray) -> ctarray:
    """Exponentiate a matrix to power k using homomorphic multiplication.

    Equivalent to: np.linalg.matrix_power(A, k)

    Applies repeated encrypted matrix multiplications.

    Example
    -------
    >>> np.linalg.matrix_power([[2, 0], [0, 2]], 3)
    array([[8, 0], [0, 8]])
    """
    result = ct_matrixA
    for _ in range(k):
        result = matmul_square(cc, keys, result, ct_matrixA)
    return result


# Reduction/accumulation prototypes (placeholders for now)


def add_reduce(cc, sum_keys, cta: ctarray, axis=None):
    """Homomorphic sum: equivalent to np.add.reduce(cta, axis=axis).

    Placeholder for column/row reduction by homomorphic summation.

    Example
    -------
    >>> np.add.reduce([[1, 2], [3, 4]], axis=0)
    array([4, 6])
    """
    pass


def add_accumulate(cc, keys, cta: ctarray, axis=None):
    """Homomorphic prefix sum: equivalent to np.add.accumulate(cta, axis=axis).

    Would be implemented via encrypted rotations + additions.

    Example
    -------
    >>> np.add.accumulate([1, 2, 3])
    array([1, 3, 6])
    """
    pass


def sub_reduce(cc, sum_keys, cta: ctarray, axis=None):
    """Homomorphic reduction with subtraction: np.subtract.reduce.

    Placeholder for sequential subtraction across axis.

    Example
    -------
    >>> np.subtract.reduce([10, 1, 2])
    7
    """
    pass


def sub_accumulate(cc, keys, cta: ctarray, axis=None):
    """Homomorphic prefix subtraction: np.subtract.accumulate.

    Example
    -------
    >>> np.subtract.accumulate([10, 1, 2])
    array([10, 9, 7])
    """
    pass


def transpose(cc, public_key, ct_matrix) -> ctarray:
    """Transpose a matrix
    Equivalent to: np.transpose()
    """
    ct_data = openfhe_matrix.EvalMatrixTranspose(cc, public_key, ct_matrix.data, ct_matrix.ncols)
    info = ct_matrix.info()
    info[0] = ct_data
    return ctarray(*info)
