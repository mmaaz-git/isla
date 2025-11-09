from hypothesis import given, strategies as st, settings
from hypothesis.extra import numpy as hnp
import numpy as np
import isla as ia

@given(hnp.arrays(
    dtype = np.float64,
    shape = st.integers(min_value=2, max_value=5).map(lambda n: (n, n)),
    elements = st.floats(min_value=-5.0, max_value=5.0,
                         allow_nan=False, allow_infinity=False, allow_subnormal=False)
    ).filter(lambda A: np.linalg.cond(A) < 1e3) # well-conditioned matrix
)
@settings(max_examples=1000)
def test_gaussian_elimination_point_intervals_is_ref(A):
    """
    Test that applying Gaussian elimination to a point interval matrix results in a row echelon form matrix.
    """

    isla_A = ia.array(A, intervals=False) # treat as point intervals
    isla_ref = ia.linalg.gaussian_elimination(isla_A) # row echelon form

    # should still be point intervals
    assert np.all(isla_ref.width == 0)

    # should be upper triangular matrix
    for i in range(isla_ref.shape[0]):
        for j in range(i):
            assert isla_ref[i, j].lower == 0
            assert isla_ref[i, j].upper == 0

    # diagonal elements should be non-zero
    for i in range(isla_ref.shape[0]):
        assert isla_ref[i, i].lower != 0
        assert isla_ref[i, i].upper != 0


@given(st.integers(min_value=2, max_value=5).flatmap(
    # make upper triangular matrix
    lambda n: st.tuples(
        hnp.arrays(
            dtype=np.float64,
            shape=(n, n),
            elements=st.floats(min_value=-5.0, max_value=5.0,
                               allow_nan=False, allow_infinity=False, allow_subnormal=False)
        ).map(lambda A: np.triu(A))
         .filter(lambda U: np.linalg.cond(U) < 1e3),
        hnp.arrays(
            dtype=np.float64,
            shape=(n,),
            elements=st.floats(min_value=-5.0, max_value=5.0,
                               allow_nan=False, allow_infinity=False, allow_subnormal=False)
        )
    )
))
@settings(max_examples=1000)
def test_back_substitution_point_intervals(Ub):
    """
    Test that back substitution produces x such that Ux = b.
    """
    U, b = Ub

    isla_U = ia.array(U, intervals=False)
    isla_b = ia.array(b, intervals=False)

    # Solve Ux = b using back substitution
    x = ia.linalg.back_substitution(isla_U, isla_b)

    # Check: U @ x should equal b
    result = isla_U @ x
    assert np.allclose(result.lower, b, rtol=1e-8, atol=1e-8)
    assert np.allclose(result.upper, b, rtol=1e-8, atol=1e-8)
