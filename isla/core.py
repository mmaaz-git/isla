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

        # Promote dtype if needed to avoid truncation
        if self.data.dtype != value.data.dtype:
            promoted_dtype = np.result_type(self.data.dtype, value.data.dtype)
            if self.data.dtype != promoted_dtype:
                self.data = self.data.astype(promoted_dtype)

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
        # Handle existing isla.ndarray - just return it (with optional copy)
        if isinstance(data, ndarray):
            return ndarray(data.data, copy=copy)

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

    return array(lower=np.full(shape, fill_interval.lower), upper=np.full(shape, fill_interval.upper))


def zeros(shape) -> 'ndarray':
    """
    Create an isla.ndarray filled with zero intervals [0.0, 0.0] (float).

    Parameters
    ----------
    shape : int or sequence of ints
        Shape of the output array.

    Returns
    -------
    isla.ndarray
        Array filled with [0.0, 0.0] intervals.

    Examples
    --------
    >>> import isla as ia
    >>> ia.zeros((2, 3))
    array([[[0., 0.],
            [0., 0.],
            [0., 0.]],
    <BLANKLINE>
           [[0., 0.],
            [0., 0.],
            [0., 0.]]])
    """
    return full(shape, [0.0, 0.0])


def ones(shape) -> 'ndarray':
    """
    Create an isla.ndarray filled with unit intervals [1.0, 1.0] (float).

    Parameters
    ----------
    shape : int or sequence of ints
        Shape of the output array.

    Returns
    -------
    isla.ndarray
        Array filled with [1.0, 1.0] intervals.

    Examples
    --------
    >>> import isla as ia
    >>> ia.ones((2, 2))
    array([[[1., 1.],
            [1., 1.]],
    <BLANKLINE>
           [[1., 1.],
            [1., 1.]]])
    """
    return full(shape, [1.0, 1.0])


def eye(n: int) -> 'ndarray':
    """
    Create an interval identity matrix (float).

    Parameters
    ----------
    n : int
        Size of the identity matrix (n x n).

    Returns
    -------
    isla.ndarray
        Identity matrix with [1.0, 1.0] on diagonal and [0.0, 0.0] elsewhere.

    Examples
    --------
    >>> import isla as ia
    >>> ia.eye(3)
    array([[[1., 1.],
            [0., 0.],
            [0., 0.]],
    <BLANKLINE>
           [[0., 0.],
            [1., 1.],
            [0., 0.]],
    <BLANKLINE>
           [[0., 0.],
            [0., 0.],
            [1., 1.]]])
    """
    return array(lower=np.eye(n), upper=np.eye(n))


## ARITHMETIC OPERATIONS ##

def negate(A) -> 'ndarray':
    """
    Negate an interval array: -[a,b] = [-b,-a].

    Parameters
    ----------
    A : array_like or isla.ndarray
        Input interval array.

    Returns
    -------
    isla.ndarray
        Negated intervals.

    Examples
    --------
    >>> import isla as ia
    >>> ia.negate([1, 3])
    array([-3, -1])

    >>> ia.negate([[1, 2], [3, 4]])
    array([[-2, -1],
           [-4, -3]])

    >>> ia.negate(5)  # scalar becomes point interval [5,5], then negated
    array([-5, -5])
    """
    if not isinstance(A, ndarray):
        A = array(A)

    result = np.stack([
        -A.upper,  # new min is negative of old max
        -A.lower   # new max is negative of old min
    ], axis=-1)

    return ndarray(result)


def add(A, B) -> 'ndarray':
    """
    Add two interval arrays: [a,b] + [c,d] = [a+c, b+d].

    Parameters
    ----------
    A : array_like or isla.ndarray
        First input interval array.
    B : array_like or isla.ndarray
        Second input interval array.

    Returns
    -------
    isla.ndarray
        Sum of intervals.

    Examples
    --------
    >>> import isla as ia
    >>> ia.add([1, 2], [3, 4])
    array([4, 6])

    >>> ia.add([[1, 2], [3, 4]], [0.5, 1.5])  # broadcasting
    array([[1.5, 3.5],
           [3.5, 5.5]])

    >>> ia.add([1, 3], 2)  # scalar becomes [2, 2]
    array([3, 5])
    """
    if not isinstance(A, ndarray):
        A = array(A)
    if not isinstance(B, ndarray):
        B = array(B)

    result = np.stack([
        A.lower + B.lower,  # Add minimums
        A.upper + B.upper   # Add maximums
    ], axis=-1)

    return ndarray(result)


def subtract(A, B) -> 'ndarray':
    """
    Subtract interval arrays: A - B = A + (-B).

    Parameters
    ----------
    A : array_like or isla.ndarray
        Minuend interval array.
    B : array_like or isla.ndarray
        Subtrahend interval array.

    Returns
    -------
    isla.ndarray
        Difference of intervals.

    Examples
    --------
    >>> import isla as ia
    >>> ia.subtract([5, 7], [1, 2])
    array([3, 6])

    >>> ia.subtract([[3, 5], [7, 9]], [1, 1])  # broadcasting
    array([[2, 4],
           [6, 8]])

    >>> ia.subtract([4, 6], 2)  # scalar becomes [2, 2]
    array([2, 4])
    """
    return add(A, negate(B))


def multiply(A, B) -> 'ndarray':
    """
    Multiply interval arrays: [a,b] * [c,d] = [min(ac,ad,bc,bd), max(ac,ad,bc,bd)].

    Parameters
    ----------
    A : array_like or isla.ndarray
        First input interval array.
    B : array_like or isla.ndarray
        Second input interval array.

    Returns
    -------
    isla.ndarray
        Product of intervals.

    Examples
    --------
    >>> import isla as ia
    >>> ia.multiply([2, 3], [4, 5])
    array([ 8, 15])

    >>> ia.multiply([-1, 1], [2, 3])  # includes negative values
    array([-3,  3])

    >>> ia.multiply([[1, 2], [3, 4]], 2)  # scalar becomes [2, 2]
    array([[2, 4],
           [6, 8]])
    """
    if not isinstance(A, ndarray):
        A = array(A)
    if not isinstance(B, ndarray):
        B = array(B)

    # Multiply each combination of min/max
    products = np.array([
        A.lower * B.lower,
        A.lower * B.upper,
        A.upper * B.lower,
        A.upper * B.upper
    ])

    result = np.stack([
        np.min(products, axis=0),
        np.max(products, axis=0)
    ], axis=-1)

    return ndarray(result)


def reciprocal(A) -> 'ndarray':
    """
    Compute the reciprocal of an interval: 1/[a,b] = [1/b, 1/a].

    Parameters
    ----------
    A : array_like or isla.ndarray
        Input interval array. Must not contain zero.

    Returns
    -------
    isla.ndarray
        Reciprocal of intervals.

    Raises
    ------
    ValueError
        If any interval contains zero.

    Examples
    --------
    >>> import isla as ia
    >>> ia.reciprocal([2, 4])
    array([0.25, 0.5 ])

    >>> ia.reciprocal([[-2, -1], [1, 3]])
    array([[-1.        , -0.5       ],
           [ 0.33333333,  1.        ]])

    >>> ia.reciprocal(2)  # scalar becomes [2, 2]
    array([0.5, 0.5])
    """
    if not isinstance(A, ndarray):
        A = array(A)

    if np.any(A.contains(0)):
        raise ValueError("Cannot compute reciprocal of interval containing zero")

    return array(lower=1/A.upper, upper=1/A.lower)


def divide(A, B) -> 'ndarray':
    """
    Divide interval arrays: A / B = A * (1/B).

    Parameters
    ----------
    A : array_like or isla.ndarray
        Numerator interval array.
    B : array_like or isla.ndarray
        Denominator interval array. Must not contain zero.

    Returns
    -------
    isla.ndarray
        Division result.

    Raises
    ------
    ValueError
        If any interval in B contains zero.

    Examples
    --------
    >>> import isla as ia
    >>> ia.divide([6, 8], [2, 4])
    array([1.5, 4. ])

    >>> ia.divide([[4, 6], [8, 12]], [2, 2])  # broadcasting
    array([[2., 3.],
           [4., 6.]])

    >>> ia.divide([10, 15], 5)  # scalar becomes [5, 5]
    array([2., 3.])
    """
    if not isinstance(A, ndarray):
        A = array(A)
    if not isinstance(B, ndarray):
        B = array(B)

    # A / B = A * (1/B)
    B_recip = reciprocal(B)
    return multiply(A, B_recip)


def intersect(A, B) -> 'ndarray':
    """
    Intersect two interval arrays: [a,b] ∩ [c,d] = [max(a,c), min(b,d)].

    Parameters
    ----------
    A : array_like or isla.ndarray
        First input interval array.
    B : array_like or isla.ndarray
        Second input interval array.

    Returns
    -------
    isla.ndarray
        Intersection of intervals. Non-overlapping intervals become [nan, nan].

    Examples
    --------
    >>> import isla as ia
    >>> ia.intersect([1, 3], [2, 4])
    array([2., 3.])

    >>> ia.intersect([1, 2], [3, 4])  # no overlap
    array([nan, nan])

    >>> ia.intersect([[1, 3], [2, 5]], [2, 4])  # broadcasting
    array([[2., 3.],
           [2., 4.]])
    """
    if not isinstance(A, ndarray):
        A = array(A)
    if not isinstance(B, ndarray):
        B = array(B)

    # Compute intersection bounds
    lower_bounds = np.maximum(A.lower, B.lower)  # Max of lower bounds
    upper_bounds = np.minimum(A.upper, B.upper)  # Min of upper bounds

    # Check for empty intersections (lower > upper)
    empty_mask = lower_bounds > upper_bounds

    # Set empty intersections to [nan, nan]
    lower_bounds = np.where(empty_mask, np.nan, lower_bounds)
    upper_bounds = np.where(empty_mask, np.nan, upper_bounds)

    return array(lower=lower_bounds, upper=upper_bounds)


def transpose(A) -> 'ndarray':
    """
    Transpose an interval array.

    Parameters
    ----------
    A : array_like or isla.ndarray
        Input interval array to transpose.

    Returns
    -------
    isla.ndarray
        Transposed interval array.

    Examples
    --------
    >>> import isla as ia
    >>> ia.transpose([[1, 2], [3, 4]])
    array([[1, 2],
           [3, 4]])

    >>> A = ia.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    >>> ia.transpose(A)
    array([[[1, 2],
            [5, 6]],
    <BLANKLINE>
           [[3, 4],
            [7, 8]]])
    """
    if not isinstance(A, ndarray):
        A = array(A)

    # Transpose the data array, keeping the interval dimension last
    # Original shape: (..., 2) -> Transposed: (...transposed, 2)
    transposed_data = np.transpose(A.data, axes=tuple(range(A.data.ndim-1)[::-1]) + (A.data.ndim-1,))

    return ndarray(transposed_data)

def dot(A, B) -> 'ndarray':
    """
    Dot product and matrix multiplication for interval arrays.

    Parameters
    ----------
    A : array_like or isla.ndarray
        First input array. Can be vector or matrix.
    B : array_like or isla.ndarray
        Second input array. Can be vector or matrix.

    Returns
    -------
    isla.ndarray
        Result of dot product or matrix multiplication.
        - Vector · Vector → Scalar interval
        - Matrix @ Vector → Vector of intervals
        - Matrix @ Matrix → Matrix of intervals

    Examples
    --------
    >>> import isla as ia
    >>> # Vector dot product (2 intervals)
    >>> ia.dot([[1, 2], [3, 4]], [[0.5, 1.5], [2, 3]])
    array([ 6.5, 15. ])

    >>> # Vector dot product with point intervals
    >>> ia.dot([[1, 1], [2, 2]], [[3, 3], [4, 4]])  # [1]·[3] + [2]·[4] = [3] + [8] = [11]
    array([11, 11])

    >>> # Matrix-vector multiplication
    >>> A = ia.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # 2x2 matrix of intervals
    >>> v = ia.array([[1, 1], [1, 1]])  # vector of 2 point intervals
    >>> ia.dot(A, v)
    array([[ 4,  6],
           [12, 14]])
    """
    if not isinstance(A, ndarray):
        A = array(A)
    if not isinstance(B, ndarray):
        B = array(B)

    # Use vectorized interval matrix multiplication
    # For intervals [a,b] and [c,d], we need to compute all products and find min/max

    if A.ndim == 1 and B.ndim == 1:
        # Vector · Vector → Scalar
        if A.shape[0] != B.shape[0]:
            raise ValueError(f"Vector dimensions don't match: {A.shape[0]} vs {B.shape[0]}")

        # Vectorized element-wise multiplication then sum
        products = multiply(A, B)  # This handles interval multiplication properly
        # Sum along the vector dimension
        result_lower = np.sum(products.data[..., 0])
        result_upper = np.sum(products.data[..., 1])
        return ndarray([result_lower, result_upper])

    elif A.ndim == 2 and B.ndim == 1:
        # Matrix @ Vector → Vector (vectorized)
        if A.shape[1] != B.shape[0]:
            raise ValueError(f"Matrix-vector dimensions don't match: {A.shape} @ {B.shape}")

        # Expand B to broadcast with A: (m, n, 2) * (n, 2) -> (m, n, 2)
        B_expanded = np.broadcast_to(B.data, A.data.shape)

        # Element-wise interval multiplication
        products = multiply(ndarray(A.data), ndarray(B_expanded))

        # Sum along the second dimension (columns)
        result_lower = np.sum(products.data[..., 0], axis=1)
        result_upper = np.sum(products.data[..., 1], axis=1)
        result_data = np.stack([result_lower, result_upper], axis=-1)

        return ndarray(result_data)

    elif A.ndim == 2 and B.ndim == 2:
        # Matrix @ Matrix → Matrix (fully vectorized using einsum)
        if A.shape[1] != B.shape[0]:
            raise ValueError(f"Matrix dimensions don't match: {A.shape} @ {B.shape}")

        # For interval matrix multiplication, we need to handle all combinations
        # A: (m, k, 2), B: (k, n, 2) -> Result: (m, n, 2)

        # Extract bounds
        A_lower, A_upper = A.data[..., 0], A.data[..., 1]
        B_lower, B_upper = B.data[..., 0], B.data[..., 1]

        # Compute all four possible products using broadcasting
        # Shape: (m, k, n) for each product
        prod_ll = np.einsum('mk,kn->mkn', A_lower, B_lower)  # A_lower @ B_lower
        prod_lu = np.einsum('mk,kn->mkn', A_lower, B_upper)  # A_lower @ B_upper
        prod_ul = np.einsum('mk,kn->mkn', A_upper, B_lower)  # A_upper @ B_lower
        prod_uu = np.einsum('mk,kn->mkn', A_upper, B_upper)  # A_upper @ B_upper

        # Sum along k dimension and find min/max across the four products
        sum_ll = np.sum(prod_ll, axis=1)  # (m, n)
        sum_lu = np.sum(prod_lu, axis=1)  # (m, n)
        sum_ul = np.sum(prod_ul, axis=1)  # (m, n)
        sum_uu = np.sum(prod_uu, axis=1)  # (m, n)

        # Find the min and max across all combinations
        all_sums = np.stack([sum_ll, sum_lu, sum_ul, sum_uu], axis=0)
        result_lower = np.min(all_sums, axis=0)
        result_upper = np.max(all_sums, axis=0)

        result_data = np.stack([result_lower, result_upper], axis=-1)
        return ndarray(result_data)

    else:
        raise ValueError(f"Unsupported dimensions for dot product: {A.shape} @ {B.shape}")