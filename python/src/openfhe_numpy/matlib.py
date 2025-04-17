import math

# Tolerances for numerical comparison
EPSILON = 1e-8
EPSILON_HIGH = 1e-4


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


class ErrorCodes:
    """Standard error identifiers for FHE matrix operations."""

    ERROR_MATRIX_SHAPE = "Matrix shapes do not match"
    ERROR_KEYGEN = "Key generation failed"
    ERROR_MATCHING = "Values do not match"


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
    rows, cols = len(a), len(a[0])
    error = 0
    is_equal = True
    for i in range(rows):
        for j in range(cols):
            f, e = check_single_equality(a[i][j], b[i][j], eps)
            error += e
            if not f:
                is_equal = False
    return is_equal, error
