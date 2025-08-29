"""
Core functionality for isla.
"""

import numpy as np


class ndarray:
    """
    isla.ndarray - An interval matrix class.

    This class wraps numpy arrays where the last dimension represents intervals [min, max].
    It supports natural arithmetic operations with proper interval semantics.
    """

    def __init__(self, input_array : np.ndarray | list, copy : bool = True):
        """
        Constructor for isla.ndarray.

        You should probably use the `array` function instead as it is more flexible.

        Parameters
        ----------
        input_array : np.ndarray | list
            Array-like input where last dimension is [min, max].
        copy : bool, optional
            Whether to copy the input array.

        Returns
        -------
        isla.ndarray
            An interval matrix.

        Raises
        ------
        ValueError
            If the input array is not a numpy array or list.
            If the last dimension is not 2.
            If at least one lower bound is greater than an upper bound.

        Examples
        --------
        >>> import isla as ia
        >>> ia.ndarray([1,2]) # just one interval
        array([1, 2])

        >>> ia.ndarray([[1,2], [3,4]]) # a vector of intervals
        array([[1, 2],
               [3, 4]])

        >>> ia.ndarray([[[1,2], [3,4]], [[5,6], [7,8]]]) # a matrix of intervals
        array([[[1, 2],
                [3, 4]],
        <BLANKLINE>
               [[5, 6],
                [7, 8]]])
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
        """
        Shape of the interval array (excluding the last [min,max] dimension).

        Returns
        -------
        tuple
            Shape of the interval array (excluding the last [min,max] dimension).

        Examples
        --------
        >>> import isla as ia
        >>> ia.ndarray([1,2]).shape
        ()

        >>> ia.ndarray([[1,2], [3,4]]).shape
        (2,)

        >>> ia.ndarray([[[1,2], [3,4]], [[5,6], [7,8]]]).shape
        (2, 2)
        """
        return self.data.shape[:-1]

    @property
    def ndim(self):
        """
        Number of dimensions (excluding the last [min,max] dimension).

        Returns
        -------
        int
            Number of dimensions (excluding the last [min,max] dimension).

        Examples
        --------
        >>> import isla as ia
        >>> ia.ndarray([1,2]).ndim
        0

        >>> ia.ndarray([[1,2], [3,4]]).ndim
        1

        >>> ia.ndarray([[[1,2], [3,4]], [[5,6], [7,8]]]).ndim
        2
        """
        return len(self.shape)

    @property
    def size(self):
        """
        Number of intervals in the array.

        Returns
        -------
        int
            Number of intervals in the array.

        Examples
        --------
        >>> import isla as ia
        >>> ia.ndarray([1,2]).size
        1

        >>> ia.ndarray([[1,2], [3,4]]).size
        2

        >>> ia.ndarray([[[1,2], [3,4]], [[5,6], [7,8]]]).size
        4
        """
        result = 1
        for dim in self.shape:
            result *= dim
        return result

    @property
    def lower(self):
        """
        Lower bounds of all intervals.

        Returns
        -------
        np.ndarray | np.float64
            Lower bounds of all intervals.
            If only a single interval, then a float is returned.
            Otherwise, an `np.ndarray` of floats is returned.

        Examples
        --------
        >>> import isla as ia
        >>> ia.ndarray([1,2]).lower
        1

        >>> ia.ndarray([[1,2], [3,4]]).lower
        array([1, 3])

        >>> ia.ndarray([[[1,2], [3,4]], [[5,6], [7,8]]]).lower
        array([[1, 3],
               [5, 7]])
        """
        result = self.data[..., 0]
        if result.ndim == 0:
            return result.item()
        else:
            return result

    @property
    def upper(self):
        """
        Upper bounds of all intervals.

        Returns
        -------
        np.ndarray | np.float64
            Upper bounds of all intervals.
            If only a single interval, then a float is returned.
            Otherwise, an `np.ndarray` of floats is returned.

        Examples
        --------
        >>> import isla as ia
        >>> ia.ndarray([1,2]).upper
        2

        >>> ia.ndarray([[1,2], [3,4]]).upper
        array([2, 4])

        >>> ia.ndarray([[[1,2], [3,4]], [[5,6], [7,8]]]).upper
        array([[2, 4],
               [6, 8]])
        """
        result = self.data[..., 1]
        if result.ndim == 0:
            return result.item()
        else:
            return result

    @property
    def width(self):
        """
        Width of all intervals.

        Returns
        -------
        np.ndarray | np.float64
            Width of all intervals.
            If only a single interval, then a float is returned.
            Otherwise, an `np.ndarray` of floats is returned.

        Examples
        --------
        >>> import isla as ia
        >>> ia.ndarray([1,2]).width
        1

        >>> ia.ndarray([[1,2], [3,4]]).width
        array([1, 1])

        >>> ia.ndarray([[[1,2], [3,4]], [[5,6], [7,8]]]).width
        array([[1, 1],
               [1, 1]])
        """
        return self.upper - self.lower

    @property
    def midpoint(self):
        """
        Midpoint of all intervals.

        Returns
        -------
        np.ndarray | np.float64
            Midpoint of all intervals.
            If only a single interval, then a float is returned.
            Otherwise, an `np.ndarray` of floats is returned.

        Examples
        --------
        >>> import isla as ia
        >>> ia.ndarray([1,2]).midpoint
        1.5

        >>> ia.ndarray([[1,2], [3,4]]).midpoint
        array([1.5, 3.5])

        >>> ia.ndarray([[[1,2], [3,4]], [[5,6], [7,8]]]).midpoint
        array([[1.5, 3.5],
               [5.5, 7.5]])
        """
        return (self.lower + self.upper) / 2

    @property
    def as_np(self):
        """
        Get the underlying numpy array with shape (..., 2).

        Returns
        -------
        np.ndarray
            Underlying numpy array with shape (..., 2).

        Examples
        --------
        >>> import isla as ia
        >>> ia.ndarray([1,2]).as_np
        array([1, 2])

        >>> type(ia.ndarray([1,2]).as_np)
        <class 'numpy.ndarray'>

        >>> ia.ndarray([[1,2], [3,4]]).as_np
        array([[1, 2],
               [3, 4]])
        """
        return self.data

    @property
    def T(self):
        """
        Transpose of the array.

        Returns
        -------
        isla.ndarray
            Transposed array.

        Examples
        --------
        >>> import isla as ia
        >>> ia.ndarray([[[1,2], [3,4]], [[5,6], [7,8]]]).T
        array([[[1, 2],
                [5, 6]],
        <BLANKLINE>
               [[3, 4],
                [7, 8]]])
        """
        return transpose(self)


    def contains(self, value):
        """
        Check if intervals contain the given value(s).

        Returns
        -------
        np.ndarray | bool
            Boolean array indicating which intervals contain the value(s).
            If only a single interval, then a boolean is returned.
            Otherwise, an `np.ndarray` of booleans is returned.

        Examples
        --------
        >>> import isla as ia
        >>> ia.ndarray([1,2]).contains(2)
        True

        >>> ia.ndarray([[1,2], [3,4]]).contains(2)
        array([ True, False])

        >>> ia.ndarray([[[1,2], [3,4]], [[5,6], [7,8]]]).contains(2)
        array([[ True, False],
               [False, False]])
        """
        return (self.lower <= value) & (value <= self.upper)


    def is_empty(self):
        """
        Check if intervals are empty (represented as [nan, nan]).

        Returns
        -------
        np.ndarray | bool
            Boolean array indicating which intervals are empty.
            If only a single interval, then a boolean is returned.
            Otherwise, an `np.ndarray` of booleans is returned.

        Examples
        --------
        >>> import isla as ia
        >>> ia.ndarray([np.nan, np.nan]).is_empty()
        True

        >>> ia.ndarray([[1,2], [np.nan, np.nan]]).is_empty()
        array([False,  True])
        """
        result = np.isnan(self.lower) & np.isnan(self.upper)
        if result.ndim == 0:
            return bool(result)
        else:
            return result


    def intersect(self, other):
        """
        Compute intersection with another interval array.

        Note that the intersection may be empty.

        Returns
        -------
        isla.ndarray
            Intersection of intervals.

        Examples
        --------
        >>> import isla as ia
        >>> ia.ndarray([1,2]).intersect(ia.ndarray([-1,1.5]))
        array([1. , 1.5])

        >>> ia.ndarray([1,2]).intersect(ia.ndarray([3,4])) # empty intersection
        array([nan, nan])

        >>> ia.ndarray([[1,2], [3,4]]).intersect(ia.ndarray([[2,3], [4,5]]))
        array([[2., 2.],
               [4., 4.]])
        """
        return intersect(self, other)

    def __repr__(self):
        """String representation (of the underlying numpy array)"""
        return repr(self.data)

    def __str__(self):
        """Readable representation (of the underlying numpy array)"""
        return str(self.data)


## CONSTRUCTORS ##

def array(data=None, lower=None, upper=None, copy: bool = True, intervals: bool = True) -> 'ndarray':
    """
    Create an isla.ndarray from various inputs.

    Parameters
    ----------
    data : array_like or scalar, optional
        Input data. Can be a scalar, list, or numpy array.
        - If scalar: creates a point interval [data, data]
        - If array-like with last dimension 2 and intervals=True: treats as intervals
        - Otherwise: converts each element to point interval [x, x]
    lower : array_like, optional
        Lower bounds for intervals. Must be used with upper.
    upper : array_like, optional
        Upper bounds for intervals. Must be used with lower.
    copy : bool
        Whether to copy the input data. Default is True.
    intervals : bool
        If True (default), treat data as intervals when last dimension is 2.
        If False, convert each element to point interval [x, x].

    Returns
    -------
    isla.ndarray
        An interval array.

    Raises
    ------
    ValueError
        If neither data nor (lower, upper) are provided.
        If both data and (lower, upper) are provided.
        If lower and upper have different shapes.

    Examples
    --------
    >>> import isla as ia
    >>> ia.array(5) # point interval
    array([5, 5])

    >>> ia.array([[1, 2], [3, 4]])
    array([[1, 2],
           [3, 4]])

    >>> ia.array(lower=[1, 3], upper=[2, 4]) # separate bounds
    array([[1, 2],
           [3, 4]])

    >>> ia.array([1, 2, 3]) # vector of point intervals (last dim is not 2)
    array([[1, 1],
           [2, 2],
           [3, 3]])

    >>> ia.array([[1, 2], [3, 4]], intervals=False) # force point intervals (intervals=False)
    array([[[1, 1],
            [2, 2]],
    <BLANKLINE>
           [[3, 3],
            [4, 4]]])
    """
    # Case 1: Both lower and upper provided
    if lower is not None and upper is not None:
        if data is not None:
            raise ValueError("Cannot provide both 'data' and 'lower'/'upper'")

        lower = np.asarray(lower)
        upper = np.asarray(upper)

        if lower.shape != upper.shape:
            raise ValueError("Lower and upper bounds must have the same shape")

        # Stack along last dimension: [..., 2]
        intervals_data = np.stack([lower, upper], axis=-1)
        return ndarray(intervals_data, copy=copy)

    # Case 2: Data provided
    elif data is not None:
        # Handle scalars - make point interval
        if np.isscalar(data):
            return ndarray([data, data], copy=copy)

        data = np.asarray(data)

        # Check if last dimension is 2 and intervals=True
        if intervals and data.ndim > 0 and data.shape[-1] == 2:
            # Treat as intervals
            return ndarray(data, copy=copy)
        else:
            # Convert to point intervals [x, x]
            point_intervals = np.stack([data, data], axis=-1)
            return ndarray(point_intervals, copy=copy)

    else:
        raise ValueError("Must provide either 'data' or both 'lower' and 'upper'")


def full(shape, fill_value, copy: bool = True) -> 'ndarray':
    """
    Create an isla.ndarray filled with a given interval value.

    Parameters
    ----------
    shape : int or sequence of ints
        Shape of the output array.
    fill_value : array_like or isla.ndarray
        Interval to fill with. Can be [lower, upper] or an isla.ndarray with single interval.
    copy : bool
        Whether to copy the fill_value. Default is True.

    Returns
    -------
    isla.ndarray
        Array filled with fill_value intervals.

    Raises
    ------
    ValueError
        If fill_value is not a valid interval specification.
        If fill_value is an isla.ndarray with more than one interval.

    Examples
    --------
    >>> import isla as ia
    >>> ia.full((2, 3), [1, 2])
    array([[[1, 2],
            [1, 2],
            [1, 2]],
    <BLANKLINE>
           [[1, 2],
            [1, 2],
            [1, 2]]])

    >>> interval = ia.array([3, 4])
    >>> ia.full((2, 2), interval)
    array([[[3, 4],
            [3, 4]],
    <BLANKLINE>
           [[3, 4],
            [3, 4]]])
    """
    # Convert fill_value to isla.ndarray first
    fill_interval = array(fill_value, copy=copy)

    # Ensure it's a single interval
    if fill_interval.size != 1:
        raise ValueError("fill_value must be a single interval")

    # Use numpy's full to create the shape, then broadcast the interval
    full_data = np.full(shape + (2,), fill_interval.data.flatten(), dtype=float)

    return ndarray(full_data, copy=False)


def zeros(shape) -> 'ndarray':
    """
    Create an isla.ndarray filled with zero intervals [0, 0].

    Parameters
    ----------
    shape : int or sequence of ints
        Shape of the output array.

    Returns
    -------
    isla.ndarray
        Array filled with [0, 0] intervals.

    Examples
    --------
    >>> import isla as ia
    >>> ia.zeros((2, 3))
    array([[[0, 0],
            [0, 0],
            [0, 0]],
    <BLANKLINE>
           [[0, 0],
            [0, 0],
            [0, 0]]])
    """
    return full(shape, [0, 0])


def ones(shape) -> 'ndarray':
    """
    Create an isla.ndarray filled with unit intervals [1, 1].

    Parameters
    ----------
    shape : int or sequence of ints
        Shape of the output array.

    Returns
    -------
    isla.ndarray
        Array filled with [1, 1] intervals.

    Examples
    --------
    >>> import isla as ia
    >>> ia.ones((2, 2))
    array([[[1, 1],
            [1, 1]],
    <BLANKLINE>
           [[1, 1],
            [1, 1]]])
    """
    return full(shape, [1, 1])


def eye(n: int) -> 'ndarray':
    """
    Create an interval identity matrix.

    Parameters
    ----------
    n : int
        Size of the identity matrix (n x n).

    Returns
    -------
    isla.ndarray
        Identity matrix with [1, 1] on diagonal and [0, 0] elsewhere.

    Examples
    --------
    >>> import isla as ia
    >>> ia.eye(3)
    array([[[1, 1],
            [0, 0],
            [0, 0]],
    <BLANKLINE>
           [[0, 0],
            [1, 1],
            [0, 0]],
    <BLANKLINE>
           [[0, 0],
            [0, 0],
            [1, 1]]])
    """
    return array(lower=np.eye(n), upper=np.eye(n))

# should have _as_array internal function for the operations to make things more natural


## ARITHMETIC OPERATIONS ##

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


def reciprocal(A):
    """Compute the reciprocal of an interval [a,b] = [1/b, 1/a] (like np.reciprocal)"""
    if A.contains(0).any():
        raise ValueError("Cannot compute reciprocal of interval containing zero")
    return array(lower=1/A.upper, upper=1/A.lower)


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

def dot(A, B):
    """
    Matrix multiplication: A @ B

    Args:
        A: isla.ndarray
        B: isla.ndarray
    """
    return ndarray(np.dot(A.data, B.data))