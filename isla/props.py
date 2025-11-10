"""
Properties of interval matrices.
"""

import numpy as np
import isla as ia

# helpful things to compute when analyzing properties of interval matrices
def comparison_matrix(A):
    C = np.zeros((A.shape[0], A.shape[0]))

    for i in range(A.shape[0]):
        for j in range(A.shape[0]):
            if i == j:
                C[i, j] = mig(A[i, j]) # mignitude of diagonal element
            else:
                C[i, j] = -mag(A[i, j]) # magnitude of off-diagonal element
    return C

def mig(a):
    """
    Mignitude of an interval is equal to the minimum of the absolute values of the lower and upper bounds.
    """
    assert a.shape == ()
    return min(abs(a.lower), abs(a.upper))

def mag(a):
    """
    Magnitude of an interval is equal to the maximum of the absolute values of the lower and upper bounds.
    """
    assert a.shape == ()
    return max(abs(a.lower), abs(a.upper))

def _is_Z_matrix_real(A: np.ndarray):
    """
    Helper function to check if a real matrix is a Z-matrix.
    A real matrix is a Z-matrix if all its off-diagonal elements are non-positive.
    """
    for i in range(A.shape[0]):
        for j in range(A.shape[0]):
            if i != j and A[i, j] > 0: # any positive off-diagonal element makes it not a Z-matrix
                return False
    return True

def is_Z_matrix(A):
    """
    An interval matrix is a Z-matrix if all A in the interval matrix are Z-matrices.
    This is equivalent to all the upper bounds of off-diagonal elements being non-positive.
    Or, the upper bound matrix being a Z-matrix.
    """

    return _is_Z_matrix_real(A.upper)

def _is_M_matrix_real(A: np.ndarray):
    """
    Helper function to check if a real matrix is an M-matrix.

    A real matrix is an M-matrix if and only if it is a Z-matrix,
    and it is non-singular, and the inverse is non-negative.
    """

    return is_Z_matrix(A) and np.linalg.det(A) != 0 and np.all(np.linalg.inv(A) >= 0)

def is_M_matrix(A):
    """
    An interval matrix is an M-matrix if it is Z-matrix
    and there exists 0 < u < R^n s.t. Au > 0 (componentwise).

    See Theorem 4.9 in Horacek's thesis for some equivalent conditions.

    I will use the third statement in that Theorem, which is that being an M-matrix
    is equivalent to the upper and lower bounds being (real) M-matrices.

    """

    if not is_Z_matrix(A): return False

    return _is_M_matrix_real(A.lower) and _is_M_matrix_real(A.upper)

def is_H_matrix(A):
    """
    An interval matrix is an H-matrix if its comparison matrix (real) is an M-matrix.

    """
    return _is_M_matrix_real(comparison_matrix(A))

def is_strictly_diagonally_dominant(A):
    """
    An interval matrix is strictly diagonally dominant if
    mig(A[i, i]) > sum_{j!=i} mag(A[i, j]) for all rows i.
    """
    for i in range(A.shape[0]):
        row_sum = 0
        for j in range(A.shape[0]):
            row_sum += mag(A[i, j])
        if mig(A[i, i]) <= row_sum:
            return False
    return True

def is_strongly_regular(A):
    """
    An interval matrix is strongly regular if its product with its midpoint inverse (real)
    is an H-matrix.
    """

    midpoint_inverse = np.linalg.inv(A.midpoint)
    return _is_H_matrix(A @midpoint_inverse)