"""
Interval Matrix Algebra implementation.

This module provides functions for performing interval linear algebra operations
on matrices where each element is represented as an interval [min, max].
"""

import numpy as np


class ndarray:
    """
    isla.ndarray - Interval array class for interval linear algebra.

    A pure Python class wrapping numpy arrays where the last dimension
    represents intervals [min, max]. Supports natural arithmetic operations
    with proper interval semantics.
    """

    def __init__(self, input_array, copy=True):
        """
        Create a new ndarray.

        Args:
            input_array: Array-like input where last dimension is [min, max]
            copy: Whether to copy the input array
        """
        # Convert input to numpy array
        self.data = np.array(input_array, copy=copy)

        # Ensure proper shape for intervals
        if self.data.ndim == 0 or self.data.shape[-1] != 2:
            raise ValueError("Last dimension must be 2 for [min, max] intervals")

        # Validate interval bounds
        if np.any(self.data[..., 0] > self.data[..., 1]):
            raise ValueError("Lower bounds must be <= upper bounds")

    def __add__(self, other):
        """Addition: A + B"""
        return add(self, other)

    def __radd__(self, other):
        """Right addition: B + A"""
        return add(other, self)

    def __sub__(self, other):
        """Subtraction: A - B"""
        return subtract(self, other)

    def __rsub__(self, other):
        """Right subtraction: B - A"""
        return subtract(other, self)

    def __mul__(self, other):
        """Multiplication: A * B"""
        return multiply(self, other)

    def __rmul__(self, other):
        """Right multiplication: B * A"""
        return multiply(other, self)

    def __neg__(self):
        """Negation: -A"""
        return negate(self)

    def __matmul__(self, other):
        """Matrix multiplication: A @ B"""
        return dot(self, other)

    def __rmatmul__(self, other):
        """Right matrix multiplication: B @ A"""
        return dot(other, self)

    def __truediv__(self, other):
        """Division: A / B"""
        return divide(self, other)

    def __rtruediv__(self, other):
        """Right division: B / A"""
        return divide(other, self)

    def __getitem__(self, key):
        """Indexing and slicing: A[0], A[1:3], etc."""
        return ndarray(self.data[key])

    def __setitem__(self, key, value):
        """Assignment: A[0] = value"""
        if not isinstance(value, ndarray):
            value = ndarray(value)
        self.data[key] = value.data

    @property
    def shape(self):
        """Shape of the interval array (excluding the last [min,max] dimension)."""
        return self.data.shape[:-1]

    @property
    def ndim(self):
        """Number of dimensions (excluding the last [min,max] dimension)."""
        return len(self.shape)

    @property
    def size(self):
        """Number of intervals in the array."""
        return np.prod(self.shape)

    @property
    def lower(self):
        """Get the lower bounds of all intervals."""
        return self.data[..., 0]

    @property
    def upper(self):
        """Get the upper bounds of all intervals."""
        return self.data[..., 1]

    @property
    def width(self):
        """Get the width of all intervals."""
        return self.upper - self.lower

    @property
    def midpoint(self):
        """Get the midpoint of all intervals."""
        return (self.lower + self.upper) / 2

    @property
    def as_np(self):
        """Get the underlying numpy array with shape (..., 2)."""
        return self.data

    @property
    def T(self):
        """Transpose of the array (like numpy's .T property)."""
        return transpose(self)

    def contains(self, value):
        """Check if intervals contain the given value(s)."""
        return (self.lower <= value) & (value <= self.upper)

    def is_empty(self):
        """Check if intervals are empty (represented as [nan, nan])."""
        return np.isnan(self.lower) & np.isnan(self.upper)

    def intersect(self, other):
        """Compute intersection with another interval array."""
        return intersect(self, other)

    def __repr__(self):
        """String representation showing intervals clearly."""
        return f"array({self.data.tolist()})"

    def __str__(self):
        """Readable string representation."""
        if self.ndim == 1:
            intervals = [f"[{self.data[i, 0]:.3f}, {self.data[i, 1]:.3f}]" for i in range(self.shape[0])]
            return "array([" + ", ".join(intervals) + "])"
        else:
            return repr(self)


def array(data=None, lower=None, upper=None, copy=True, intervals=True):
    """
    Create an isla.ndarray from various inputs.

    Args:
        data: Array-like data
        lower: Lower bounds (used with upper)
        upper: Upper bounds (used with lower)
        copy: Whether to copy the input data
        intervals: If True (default), treat data as intervals [..., 2].
                  If False, convert each element to point interval [x, x]

    Examples:
        >>> import isla
        >>> # From intervals directly
        >>> A = isla.array([[1, 2], [3, 4]])
        >>>
        >>> # From separate bounds
        >>> B = isla.array(lower=[1, 3], upper=[2, 4])
        >>>
        >>> # From numpy array (convert to point intervals)
        >>> C = isla.array([1, 2, 3], intervals=False)  # → [[1,1], [2,2], [3,3]]
        >>>
        >>> # From numpy array with uncertainty
        >>> D = isla.array([[1, 1.1], [2, 2.1], [3, 3.1]], intervals=True)
    """
    if lower is not None and upper is not None:
        # Construct from separate lower/upper bounds
        lower = np.asarray(lower)
        upper = np.asarray(upper)

        if lower.shape != upper.shape:
            raise ValueError("Lower and upper bounds must have the same shape")

        # Stack along last dimension: [..., 2]
        intervals_data = np.stack([lower, upper], axis=-1)
        return ndarray(intervals_data, copy=copy)

    elif data is not None:
        data = np.asarray(data)

        if intervals:
            # Standard case: data already formatted as [..., 2]
            if data.shape[-1] != 2:
                raise ValueError("When intervals=True, last dimension must be 2 for [min, max] intervals")
            return ndarray(data, copy=copy)
        else:
            # Convert each element to point interval [x, x]
            # Shape: (...) -> (..., 2)
            point_intervals = np.stack([data, data], axis=-1)
            return ndarray(point_intervals, copy=copy)

    else:
        raise ValueError("Must provide either 'data' or both 'lower' and 'upper'")


def full(shape, fill_value, copy=True):
    """
    Create an isla.ndarray filled with a given interval value (like np.full).

    Args:
        shape: Shape of the output array
        fill_value: Interval to fill with, as [lower, upper] or isla.ndarray
        copy: Whether to copy the fill_value

    Returns:
        isla.ndarray: Array filled with fill_value intervals

    Examples:
        >>> import isla
        >>> # Fill 2x3 array with interval [1, 2]
        >>> A = isla.full((2, 3), [1, 2])
        >>>
        >>> # Fill with existing interval
        >>> interval = isla.array([[3, 4]])
        >>> B = isla.full((2, 2), interval[0])
    """
    # Convert fill_value to interval if needed
    if isinstance(fill_value, ndarray):
        if fill_value.size != 1:
            raise ValueError("fill_value must be a single interval")
        interval_val = fill_value.data.flatten()
    else:
        fill_value = np.asarray(fill_value)
        if fill_value.shape != (2,):
            raise ValueError("fill_value must be [lower, upper] for interval")
        interval_val = fill_value

    # Create array with repeated intervals
    full_data = np.full(shape + (2,), 0.0)
    full_data[..., 0] = interval_val[0]  # Lower bounds
    full_data[..., 1] = interval_val[1]  # Upper bounds

    return ndarray(full_data, copy=copy)


def zeros(shape, copy=True):
    """
    Create an isla.ndarray filled with zero intervals [0, 0] (like np.zeros).

    Args:
        shape: Shape of the output array
        copy: Whether to copy the data

    Returns:
        isla.ndarray: Array filled with [0, 0] intervals
    """
    return full(shape, [0.0, 0.0], copy=copy)


def ones(shape, copy=True):
    """
    Create an isla.ndarray filled with one intervals [1, 1] (like np.ones).

    Args:
        shape: Shape of the output array
        copy: Whether to copy the data

    Returns:
        isla.ndarray: Array filled with [1, 1] intervals
    """
    return full(shape, [1.0, 1.0], copy=copy)


def negate(A):
    """
    Negate an interval array: -[a,b] = [-b,-a]

    Args:
        A: isla.ndarray

    Returns:
        isla.ndarray: Negated intervals
    """
    if not isinstance(A, ndarray):
        A = ndarray(A)

    result = np.stack([
        -A.data[..., 1],  # new min is negative of old max
        -A.data[..., 0]   # new max is negative of old min
    ], axis=-1)

    return ndarray(result)


def add(A, B):
    """
    Add two interval arrays: [a,b] + [c,d] = [a+c, b+d]

    Args:
        A: isla.ndarray
        B: isla.ndarray

    Returns:
        isla.ndarray: Sum of intervals
    """
    if not isinstance(A, ndarray):
        A = ndarray(A)
    if not isinstance(B, ndarray):
        B = ndarray(B)

    result = np.stack([
        A.data[..., 0] + B.data[..., 0],  # Add minimums
        A.data[..., 1] + B.data[..., 1]   # Add maximums
    ], axis=-1)

    return ndarray(result)


def subtract(A, B):
    """
    Subtract interval arrays: A - B

    Args:
        A: isla.ndarray
        B: isla.ndarray

    Returns:
        isla.ndarray: Difference of intervals
    """
    return add(A, negate(B))


def multiply(A, B):
    """
    Multiply interval arrays: products of all combinations, then take min/max

    Args:
        A: isla.ndarray
        B: isla.ndarray

    Returns:
        isla.ndarray: Product of intervals
    """
    if not isinstance(A, ndarray):
        A = ndarray(A)
    if not isinstance(B, ndarray):
        B = ndarray(B)

    # Multiply each combination of min/max
    products = np.array([
        A.data[..., 0] * B.data[..., 0],
        A.data[..., 0] * B.data[..., 1],
        A.data[..., 1] * B.data[..., 0],
        A.data[..., 1] * B.data[..., 1]
    ])

    result = np.stack([
        np.min(products, axis=0),
        np.max(products, axis=0)
    ], axis=-1)

    return ndarray(result)


def intersect(A, B):
    """
    Intersect two interval arrays

    Args:
        A: isla.ndarray
        B: isla.ndarray

    Returns:
        isla.ndarray: Intersection of intervals. Non-overlapping intervals
                     become [nan, nan] to indicate empty intersection.
    """
    if not isinstance(A, ndarray):
        A = ndarray(A)
    if not isinstance(B, ndarray):
        B = ndarray(B)

    # Compute intersection bounds
    lower_bounds = np.maximum(A.data[..., 0], B.data[..., 0])  # Max of lower bounds
    upper_bounds = np.minimum(A.data[..., 1], B.data[..., 1])  # Min of upper bounds

    # Check for empty intersections (lower > upper)
    empty_mask = lower_bounds > upper_bounds

    # Set empty intersections to [nan, nan]
    lower_bounds = np.where(empty_mask, np.nan, lower_bounds)
    upper_bounds = np.where(empty_mask, np.nan, upper_bounds)

    return array(lower=lower_bounds, upper=upper_bounds)


def eye(n):
    """Create an interval identity matrix of size n x n (like np.eye)"""
    return array(lower=np.eye(n), upper=np.eye(n))


def reciprocal(A):
    """Compute the reciprocal of an interval [a,b] = [1/b, 1/a] (like np.reciprocal)"""
    if ((A.lower <= 0) & (A.upper >= 0)).any():
        raise ValueError("Cannot compute reciprocal of interval containing zero")
    return array(lower=1/A.upper, upper=1/A.lower)


def _gauss_seidel(A, b, x0=None, max_iterations=100, tolerance=1e-6):
    """
    Solve Ax = b using interval Gauss-Seidel iteration.

    Args:
        A: isla.ndarray - Coefficient matrix
        b: isla.ndarray - Right-hand side vector
        x0: isla.ndarray - Initial guess, defaults to b if None
        max_iterations: Maximum number of iterations
        tolerance: Convergence tolerance

    Returns:
        isla.ndarray: Solution interval vector
    """
    if not isinstance(A, ndarray):
        A = ndarray(A)
    if not isinstance(b, ndarray):
        b = ndarray(b)

    if x0 is None:
        x = ndarray(b.data.copy())
    else:
        if not isinstance(x0, ndarray):
            x0 = ndarray(x0)
        x = ndarray(x0.data.copy())

    n = A.shape[0]

    for iteration in range(max_iterations):
        x_old = ndarray(x.data.copy())

        for i in range(n):
            # Compute sum of A[i,j] * x[j] for j != i using clean indexing
            sum_ax = zeros((1,))

            for j in range(n):
                if i != j:
                    a_ij = A[i:i+1, j:j+1]  # A[i,j] as 1x1 interval matrix
                    x_j = x[j:j+1]          # x[j] as 1D interval vector
                    prod = multiply(a_ij, x_j)
                    sum_ax = add(sum_ax, prod)

            # x[i] = (b[i] - sum_ax) / A[i,i] using clean operations
            b_i = b[i:i+1]              # b[i] as 1D interval
            a_ii = A[i:i+1, i:i+1]      # A[i,i] as 1x1 interval matrix

            numerator = subtract(b_i, sum_ax)
            x_i_new = divide(numerator, a_ii)

            # Update x[i] cleanly
            x[i:i+1] = x_i_new

        # Check convergence using clean API
        diff = subtract(x, x_old)
        max_width = np.max(diff.width)

        if max_width < tolerance:
            break

    return x


def solve(A, b, max_iterations=100, tolerance=1e-6):
    """
    Solve the interval linear system Ax = b (like np.linalg.solve).

    Args:
        A: isla.ndarray - Coefficient matrix
        b: isla.ndarray - Right-hand side vector
        max_iterations: Maximum iterations for Gauss-Seidel
        tolerance: Convergence tolerance

    Returns:
        isla.ndarray: Solution vector
    """
    return _gauss_seidel(A, b, max_iterations=max_iterations, tolerance=tolerance)


def inv(A, max_iterations=100, tolerance=1e-6):
    """
    Compute the inverse of an interval matrix using Gauss-Seidel iteration.

    Args:
        A: isla.ndarray - Square interval matrix
        max_iterations: Maximum iterations for Gauss-Seidel
        tolerance: Convergence tolerance

    Returns:
        isla.ndarray: Inverse interval matrix
    """
    if not isinstance(A, ndarray):
        A = ndarray(A)

    n = A.shape[0]
    identity = eye(n)

    # Create result array
    inverse_data = np.zeros((n, n, 2))

    # Solve A * X = I column by column
    for i in range(n):
        b_column = identity[:, i]  # i-th column of identity (isla.ndarray)
        x_column = _gauss_seidel(A, b_column, max_iterations=max_iterations, tolerance=tolerance)
        inverse_data[:, i] = x_column.data

    return ndarray(inverse_data)



def dot(a, b):
    """
    Compute interval dot product (like np.dot).

    Behavior matches numpy:
    - 1D x 1D: vector dot product (scalar result)
    - 2D x 1D: matrix-vector product
    - 1D x 2D: vector-matrix product
    - 2D x 2D: matrix-matrix product

    Args:
        a: isla.ndarray - First array
        b: isla.ndarray - Second array

    Returns:
        isla.ndarray: Dot product result
    """
    if not isinstance(a, ndarray):
        a = ndarray(a)
    if not isinstance(b, ndarray):
        b = ndarray(b)

    # Case 1: Both 1D (vector dot product)
    if a.ndim == 1 and b.ndim == 1:
        if a.shape[0] != b.shape[0]:
            raise ValueError("Vector dimensions must match for dot product")

        result = ndarray([[0.0, 0.0]])
        for i in range(a.shape[0]):
            prod = multiply(a[i:i+1], b[i:i+1])
            result = add(result, prod)
        return result[0]  # Return scalar interval

    # Case 2: 2D x 1D (matrix-vector)
    elif a.ndim == 2 and b.ndim == 1:
        if a.shape[1] != b.shape[0]:
            raise ValueError(f"Shapes {a.shape} and {b.shape} not aligned for matrix-vector product")

        result_data = np.zeros((a.shape[0], 2))
        for i in range(a.shape[0]):
            row_result = ndarray([[0.0, 0.0]])
            for j in range(a.shape[1]):
                prod = multiply(a[i:i+1, j:j+1], b[j:j+1])
                row_result = add(row_result, prod)
            result_data[i] = row_result.data[0]
        return ndarray(result_data)

    # Case 3: 1D x 2D (vector-matrix)
    elif a.ndim == 1 and b.ndim == 2:
        if a.shape[0] != b.shape[0]:
            raise ValueError(f"Shapes {a.shape} and {b.shape} not aligned for vector-matrix product")

        result_data = np.zeros((b.shape[1], 2))
        for j in range(b.shape[1]):
            col_result = ndarray([[0.0, 0.0]])
            for i in range(a.shape[0]):
                prod = multiply(a[i:i+1], b[i:i+1, j:j+1])
                col_result = add(col_result, prod)
            result_data[j] = col_result.data[0]
        return ndarray(result_data)

    # Case 4: 2D x 2D (matrix-matrix)
    elif a.ndim == 2 and b.ndim == 2:
        if a.shape[1] != b.shape[0]:
            raise ValueError(f"Shapes {a.shape} and {b.shape} not aligned for matrix multiplication")

        result_data = np.zeros((a.shape[0], b.shape[1], 2))
        for i in range(a.shape[0]):
            for j in range(b.shape[1]):
                element_result = ndarray([[0.0, 0.0]])
                for k in range(a.shape[1]):
                    prod = multiply(a[i:i+1, k:k+1], b[k:k+1, j:j+1])
                    element_result = add(element_result, prod)
                result_data[i, j] = element_result.data[0]
        return ndarray(result_data)

    else:
        raise ValueError(f"dot not implemented for {a.ndim}D x {b.ndim}D arrays")


def divide(a, b):
    """
    Divide interval arrays: a / b (like np.divide).

    Args:
        a: isla.ndarray - Numerator
        b: isla.ndarray - Denominator

    Returns:
        isla.ndarray: Division result
    """
    if not isinstance(a, ndarray):
        a = ndarray(a)
    if not isinstance(b, ndarray):
        b = ndarray(b)

    # a / b = a * (1/b)
    b_recip = reciprocal(b)
    return multiply(a, b_recip)



def transpose(A):
    """
    Transpose an interval array (like np.transpose).

    Args:
        A: isla.ndarray - Array to transpose

    Returns:
        isla.ndarray: Transposed array
    """
    if not isinstance(A, ndarray):
        A = ndarray(A)

    # Transpose the data array, keeping the interval dimension last
    # Original shape: (..., 2) -> Transposed: (...transposed, 2)
    transposed_data = np.transpose(A.data, axes=tuple(range(A.data.ndim-1)[::-1]) + (A.data.ndim-1,))

    return ndarray(transposed_data)