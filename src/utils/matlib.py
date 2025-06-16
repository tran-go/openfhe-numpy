import math
import numpy as np
from .constants import EPSILON


def next_power_of_two(x):
    """
    Compute the smallest power of two greater than or equal to x.

    Parameters
    ----------
    x : int
        The input number.

    Returns
    -------
    int
        The next power of two ≥ x.
    """
    return 2 ** math.ceil(math.log2(x))


def is_power_of_two(x):
    """
    Check if a number is a power of two.

    Parameters
    ----------
    x : int
        Number to check.

    Returns
    -------
    bool
        True if x is a power of two, False otherwise.
    """
    return (x & (x - 1) == 0) and x != 0


def check_single_equality(a, b, eps=EPSILON):
    """
    Compare two floating-point values within a tolerance.

    Parameters
    ----------
    a, b : float
        Values to compare.
    eps : float, optional
        Tolerance threshold (default: EPSILON).

    Returns
    -------
    tuple
        (is_equal: bool, error: float)
    """
    error = abs(a - b)
    return error <= eps, error


def check_equality_vector(a, b, vector_size=None, eps=EPSILON):
    """
    Compare two vectors element-wise with tolerance.

    Parameters
    ----------
    a, b : list
        Input vectors.
    vector_size : int, optional
        Number of entries to compare (default: full length).
    eps : float
        Comparison tolerance.

    Returns
    -------
    tuple
        (is_equal: bool, total_error: float)
    """
    if vector_size is None:
        vector_size = len(a)
    error = 0
    is_equal = True
    for i in range(vector_size):
        f, e = check_single_equality(a[i], b[i], eps)
        error += e
        if not f:
            is_equal = False
    return is_equal, error


def check_equality_matrix(a, b, eps=EPSILON):
    """
    Compare two matrices element-wise with tolerance.

    Parameters
    ----------
    a, b : list of list
        Matrices to compare.
    eps : float
        Tolerance threshold.

    Returns
    -------
    tuple
        (is_equal: bool, total_error: float)
    """
    a = np.array(a)
    b = np.array(b)

    error, is_equal = 0, True

    if a.ndim == 1:
        for i in range(len(a)):
            f, e = check_single_equality(a[i], b[i], eps)
            error += 2
            if not f:
                is_equal = False
        return is_equal, error

    rows, cols = len(a), len(a[0])

    for i in range(rows):
        for j in range(cols):
            f, e = check_single_equality(a[i][j], b[i][j], eps)
            error += e
            if not f:
                is_equal = False
    return is_equal, error


def _gen_comm_mat(m, n, opt=1):
    """
    Generate a commutation matrix https://en.wikipedia.org/wiki/Commutation_matrix
    """
    d = m * n
    vec_commutation = [0] * (d**2)
    matrix = np.zeros((m * n, m * n), dtype=int)
    # matrix = [[0] * d for _ in range(d)]
    for i in range(m):
        for j in range(n):
            vec_commutation[(i * n + j) * d + (j * m + i)] = 1
            matrix[i * n + j, j * m + i] = 1
    if opt == 0:
        return matrix
    return vec_commutation


# Function to generate a random square matrix of size n x n
def _generate_random_matrix(n):
    import random

    return [[random.randint(0, 9) for _ in range(n)] for _ in range(n)]


# Function to multiply two matrices A and B in Plain
def _matrix_multiply(A, B, precision=2):
    """
    Multiply two square matrices A and B.

    Parameters
    ----------
    A : list of list of float
        The left-hand matrix.
    B : list of list of float
        The right-hand matrix.
    precision : int, optional
        Number of decimal places to round the result to. Default is 2.

    Returns
    -------
    result : list of list of float
        The resulting matrix after multiplication.
    """
    n = len(A)
    result = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                result[i][j] += A[i][k] * B[k][j]
    return [[round(result[i][j], precision) for j in range(n)] for i in range(n)]


def _rotate_vector(vec, k):
    """Rotate a vector by k positions.

    Parameters
    ----------
    vec : list or ndarray
        The input vector to rotate.
    k : int
        Number of positions to rotate the vector.

    Returns
    -------
    list
        Rotated vector.
    """
    n = len(vec)
    new_vec = vec[:]
    return [new_vec[(i + k) % n] for i in range(n)]
