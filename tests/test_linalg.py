from hypothesis import given, strategies as st, settings
from hypothesis.extra import numpy as hnp
import numpy as np
import isla as ia

# reference for Gaussian elimination
# taken verbatim from stackoverflow
# https://math.stackexchange.com/questions/3073083/how-to-reduce-matrix-into-row-echelon-form-in-numpy
# numpy does not have a built-in function for row echelon form nor does scipy
# and I don't want to import sympy just for this
def row_echelon(A):
    """ Return Row Echelon Form of matrix A """

    # if matrix A has no columns or rows,
    # it is already in REF, so we return itself
    r, c = A.shape
    if r == 0 or c == 0:
        return A

    # we search for non-zero element in the first column
    for i in range(len(A)):
        if A[i,0] != 0:
            break
    else:
        # if all elements in the first column is zero,
        # we perform REF on matrix from second column
        B = row_echelon(A[:,1:])
        # and then add the first zero-column back
        return np.hstack([A[:,:1], B])

    # if non-zero element happens not in the first row,
    # we switch rows
    if i > 0:
        ith_row = A[i].copy()
        A[i] = A[0]
        A[0] = ith_row

    # we divide first row by first element in it
    A[0] = A[0] / A[0,0]
    # we subtract all subsequent rows with first row (it has 1 now as first element)
    # multiplied by the corresponding element in the first column
    A[1:] -= A[0] * A[1:,0:1]

    # we perform REF on matrix from second row, from second column
    B = row_echelon(A[1:,1:])

    # we add first row and first (zero) column, and return
    return np.vstack([A[:1], np.hstack([A[1:,:1], B]) ])

@given(st.integers(min_value=2, max_value=5).flatmap(
    lambda n: hnp.arrays(
        dtype=np.float64,
        shape=(n, n), # square matrix
        elements=st.floats(min_value=-5.0, max_value=5.0,
                           allow_nan=False, allow_infinity=False, allow_subnormal=False)
    ).filter(lambda A: np.linalg.cond(A) < 1e3) # well-conditioned matrix
))
@settings(max_examples=1000)
def test_gaussian_elimination_point_intervals(A):
    scipy_ref = row_echelon(A)

    isla_A = ia.array(A, intervals=False) # treat as point intervals
    isla_ref = ia.linalg.gaussian_elimination(isla_A) # row echelon form
    assert np.allclose(isla_ref.lower, isla_ref.upper)
    assert np.allclose(isla_ref.lower, scipy_ref)

