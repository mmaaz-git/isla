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


@given(st.integers(min_value=2, max_value=5).flatmap(
    lambda n: st.tuples(
        hnp.arrays(
            dtype=np.float64,
            shape=(n, n),
            elements=st.floats(min_value=-5.0, max_value=5.0,
                               allow_nan=False, allow_infinity=False, allow_subnormal=False)
        ).filter(lambda A: np.linalg.cond(A) < 1e3),
        hnp.arrays(
            dtype=np.float64,
            shape=(n,),
            elements=st.floats(min_value=-5.0, max_value=5.0,
                               allow_nan=False, allow_infinity=False, allow_subnormal=False)
        )
    )
))
@settings(max_examples=1000)
def test_solve_gaussian_elimination_point_intervals(Ab):
    """
    Test that solving a point interval system with Gaussian elimination produces a point solution that satisfies the system.
    """
    A, b = Ab

    isla_A = ia.array(A, intervals=False)
    isla_b = ia.array(b, intervals=False)

    # Solve Ax = b
    x = ia.linalg.solve(isla_A, isla_b)

    # Check: A @ x should equal b
    result = isla_A @ x
    assert np.allclose(result.lower, b, rtol=1e-7, atol=1e-7)
    assert np.allclose(result.upper, b, rtol=1e-7, atol=1e-7)


@given(st.integers(min_value=2, max_value=5).flatmap(
    lambda n: st.tuples(
        hnp.arrays(
            dtype=np.float64,
            shape=(n, n),
            elements=st.floats(min_value=1.0, max_value=5.0,
                               allow_nan=False, allow_infinity=False, allow_subnormal=False)
        ).filter(lambda A: np.linalg.cond(A) < 1e3 and np.all(np.abs(A) > 1e-5)),  # intervals should not contain near-zero elements
        hnp.arrays(
            dtype=np.float64,
            shape=(n,),
            elements=st.floats(min_value=-5.0, max_value=5.0,
                               allow_nan=False, allow_infinity=False, allow_subnormal=False)
        ),
    )),
    # interval radius
    # we will use this to make the interval matrix
    st.floats(min_value=0.01, max_value=0.5)
)
@settings(max_examples=100)
def test_solve_gaussian_elimination_containment(Ab, r):
    """
    Take an interval matrix A and interval vector b.
    Take a scalar matrix A' and b', contained in A and b respectively.
    Then, the solution to A'x=b' should be contained in the (interval) solution to Ax=b.
    """
    A, b = Ab

    # Create interval matrix and vector with some width
    isla_A = ia.array(lower = A - r, upper = A + r)
    isla_b = ia.array(lower = b - r, upper = b + r)

    # Solve interval system
    try:
        isla_x = ia.linalg.solve(isla_A, isla_b)
    except ValueError:
        # intervals are too ill-conditioned
        # 0 in pivot, because intervals can blow up during gaussian elimination
        return

    # note that A and b are contained in isla_A and isla_b
    x = np.linalg.solve(A, b)

    # Check containment: x should be inside isla_x
    assert np.all(isla_x.contains(x))
