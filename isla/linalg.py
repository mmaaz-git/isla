"""
Linear algebra routines for interval arithmetic.
"""

import numpy as np
from .core import ndarray, array, zeros


def gaussian_elimination(A, b=None):
    """
    Interval Gaussian elimination.

    Implements Algorithm 5.9 in Horacek's thesis.

    Takes an interval matrix A and interval vector b and eliminates
    the matrix (A | b) into row echelon form.

    Algorithm:
    1. For rows i = 1, ..., (n-1) do the following steps
    2. For j = i, ..., n find row with 0 \\notin a_ji
    3. If such row cannot be found, notify that A is possibly singular
    4. For every row j > i set:
       - a_ji := [0, 0]
       - a_(j,i+1:n) := a_(j,i+1:n) - (a_ji/a_ii) * a_(i,i+1:n)
       - b_j := b_j - (a_ji/a_ii) * b_i

    Parameters
    ----------
    A : isla.ndarray
        Coefficient matrix of shape (n, n).
    b : isla.ndarray, optional
        Right-hand side vector of shape (n,). If provided, creates augmented matrix.

    Returns
    -------
    isla.ndarray
        Matrix in row echelon form. If b was provided, returns augmented matrix.

    Raises
    ------
    ValueError
        If matrix is possibly singular (no valid pivot found).

    Examples
    --------
    >>> import isla as ia
    >>> A = ia.array([[[2, 2], [1, 1]], [[1, 1], [3, 3]]])
    >>> b = ia.array([[5, 5], [7, 7]])
    >>> ia.linalg.gaussian_elimination(A, b)
    array([[[2. , 2. ],
            [1. , 1. ],
            [5. , 5. ]],
    <BLANKLINE>
           [[0. , 0. ],
            [2.5, 2.5],
            [4.5, 4.5]]])

    References
    ----------
    J. Horácek. Interval linear and nonlinear systems. PhD thesis. 2019.
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square")

    A_copy = array(A, copy=True)
    n_rows, n_cols = A_copy.shape

    # Create augmented matrix if b is provided
    if b is not None:
        # Check dimensions
        if n_rows != b.shape[0]:
            raise ValueError(f"Matrix and vector dimensions don't match: {n_rows} vs {b.shape[0]}")

        b_copy = array(b, copy=True)

    # Step 1: For rows i = 1, ..., (n-1) (in 0-based indexing: i = 0, ..., n-2)
    for i in range(n_rows - 1):
        # Step 2: For j \in 1...n, find row with 0 \notin a_ji using mignitude pivoting
        best_mignitude = 0
        pivot_row = None
        for j in range(i, n_rows):
            interval = A_copy[j, i]
            if not interval.contains(0):  # 0 \notin a_ji
                # Compute mignitude: min(|lower|, |upper|)
                mig = min(abs(interval.lower), abs(interval.upper))
                if mig > best_mignitude:
                    best_mignitude = mig
                    pivot_row = j

        # Step 3: If such row cannot be found, notify that A is possibly singular
        if pivot_row is None:
            raise ValueError(f"Matrix is possibly singular: no valid pivot found in column {i}")

        # Step 3.5: Swap rows if needed to bring pivot to diagonal
        if pivot_row != i:
            # Swap rows i and pivot_row in A_copy
            A_copy[[i, pivot_row]] = A_copy[[pivot_row, i]]
            # Also swap corresponding elements in b_copy if it exists
            if b is not None:
                b_copy[[i, pivot_row]] = b_copy[[pivot_row, i]]

        # Step 4: For every row j > i set the elimination formulas
        for j in range(i + 1, n_rows):
            # store this before we zero it out
            factor = A_copy[j, i] / A_copy[i, i]
            # a_ji := [0, 0]
            A_copy[j, i] = np.array([0.0, 0.0])
            # a_{j,i+1:n} := a_{j,i+1:n} - (a_ji/a_ii) * a_{i,i+1:n}
            A_copy[j, i + 1:] = A_copy[j, i + 1:] - factor * A_copy[i, i + 1:]
            # b_j := b_j - (a_ji/a_ii) * b_i
            if b is not None:
                b_copy[j] = b_copy[j] - factor * b_copy[i]

    if b is not None:
        # Create augmented matrix [A | b] by concatenating along column axis
        b_expanded = b_copy.data.reshape(b_copy.shape[0], 1, 2)
        augmented_data = np.concatenate([A_copy.data, b_expanded], axis=1)
        return ndarray(augmented_data)
    else:
        return A_copy


def back_substitution(A, b=None):
    """
    Interval back substitution for upper triangular systems.

    Solves Ux = b where U is upper triangular, using interval arithmetic.
    Implements Algorithm 5.10 in Horacek's thesis.

    Parameters
    ----------
    A : isla.ndarray
        Upper triangular matrix of shape (n, n) or augmented matrix (n, n+1).
    b : isla.ndarray, optional
        Right-hand side vector. If None, assumes U is augmented matrix.

    Returns
    -------
    isla.ndarray
        Solution vector x of shape (n,).

    Examples
    --------
    >>> import isla as ia
    >>> # Upper triangular system
    >>> U = ia.array([[[2, 2], [1, 1]], [[0, 0], [2.5, 2.5]]])
    >>> b = ia.array([[5, 5], [4.5, 4.5]])
    >>> ia.linalg.back_substitution(U, b)
    array([[1.6, 1.6],
           [1.8, 1.8]])

    References
    ----------
    J. Horácek. Interval linear and nonlinear systems. PhD thesis. 2019.
    """
    if not isinstance(A, ndarray):
        A = array(A)

    n_rows, n_cols = A.shape

    if b is not None:
        if not isinstance(b, ndarray):
            b = array(b)
        if A.shape[0] != b.shape[0]:
            raise ValueError(f"Matrix and vector dimensions don't match: {A.shape[0]} vs {b.shape[0]}")
        rhs = b
        n = n_rows
    else:
        # Assume augmented matrix
        if n_cols < n_rows + 1:
            raise ValueError("Matrix must be augmented (n x n+1) or provide separate b vector")
        rhs_data = A.data[:, -1, :]  # Last column
        rhs = ndarray(rhs_data)
        A_data = A.data[:, :-1, :]  # All but last column
        A = ndarray(A_data)
        n = n_rows

    # Initialize solution vector
    x_data = np.zeros((n, 2))
    x = ndarray(x_data)

    # Back substitution
    for i in range(n - 1, -1, -1):
        # Start with rhs[i]
        sum_val = rhs[i]

        # Subtract known terms
        for j in range(i + 1, n):
            sum_val = sum_val - A[i, j] * x[j]

        # Divide by diagonal element
        if A[i, i].contains(0):
            raise ValueError(f"Diagonal element at ({i}, {i}) contains zero")

        x[i] = sum_val / A[i, i]

    return x


def solve(A, b, method="gaussian_elimination"):
    """
    Solve interval linear system Ax = b.

    Parameters
    ----------
    A : array_like or isla.ndarray
        Coefficient matrix of shape (n, n).
    b : array_like or isla.ndarray
        Right-hand side vector of shape (n,).
    method : str
        Method to use for solving the system.
        - "gaussian_elimination": Gaussian elimination with back substitution

    Returns
    -------
    isla.ndarray
        Solution vector x of shape (n,).

    Examples
    --------
    >>> import isla as ia
    >>> A = ia.array([[[2, 2], [1, 1]], [[1, 1], [3, 3]]])
    >>> b = ia.array([[5, 5], [7, 7]])
    >>> x = ia.linalg.solve(A, b)
    >>> np.allclose((A @ x).data, b.data)
    True
    """
    if method == "gaussian_elimination":
        A_augmented = gaussian_elimination(A, b)
        x = back_substitution(A_augmented)
    else:
        raise ValueError(f"Unsupported method: {method}")

    return x
