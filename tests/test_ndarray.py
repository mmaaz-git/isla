"""
Tests for ia.ndarray class - internal methods, properties, and class behavior.
"""

import pytest
import numpy as np
import isla as ia


def test_basic_construction():
    A = ia.ndarray([[1, 2], [3, 4]])
    expected = np.array([[1, 2], [3, 4]])
    np.testing.assert_array_equal(A.data, expected)


def test_construction_with_copy():
    original = np.array([[1, 2], [3, 4]])
    A = ia.ndarray(original, copy=True)
    A.data[0, 0] = 999
    assert original[0, 0] == 1  # Original unchanged

    B = ia.ndarray(original, copy=False)
    B.data[0, 0] = 999
    assert original[0, 0] == 999  # Original changed


def test_invalid_shape():
    with pytest.raises(ValueError):
        ia.ndarray([1, 2, 3])  # Not (..., 2) shape

    with pytest.raises(ValueError):
        ia.ndarray([[1, 2, 3], [4, 5, 6]])  # Last dim not 2


def test_invalid_bounds():
    with pytest.raises(ValueError):
        ia.ndarray([[2, 1]])  # Lower > upper

    with pytest.raises(ValueError):
        ia.ndarray([[1, 2], [4, 3]])  # Second interval invalid


def test_shape_property():
    A = ia.ndarray([[1, 2]])
    assert A.shape == (1,)

    B = ia.ndarray([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    assert B.shape == (2, 2)


def test_ndim_property():
    A = ia.ndarray([[1, 2]])  # 1D intervals
    assert A.ndim == 1

    B = ia.ndarray([[[1, 2], [3, 4]]])  # 2D intervals
    assert B.ndim == 2


def test_size_property():
    A = ia.ndarray([[1, 2], [3, 4]])
    assert A.size == 2

    B = ia.ndarray([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    assert B.size == 4


def test_lower_upper_properties():
    A = ia.ndarray([[1, 3], [2, 5]])
    np.testing.assert_array_equal(A.lower, [1, 2])
    np.testing.assert_array_equal(A.upper, [3, 5])


def test_width_property():
    A = ia.ndarray([[1, 3], [2, 6]])
    np.testing.assert_array_equal(A.width, [2, 4])


def test_midpoint_property():
    A = ia.ndarray([[1, 3], [2, 6]])
    np.testing.assert_array_equal(A.midpoint, [2, 4])


def test_as_np_property():
    data = np.array([[1, 2], [3, 4]])
    A = ia.ndarray(data)
    np.testing.assert_array_equal(A.as_np, data)
    assert A.as_np is A.data  # Should be same object


def test_transpose_property():
    A = ia.ndarray([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # 2x2
    At = A.T
    assert At.shape == (2, 2)  # Should still be 2x2 (square matrix)
    np.testing.assert_array_equal(At.data[0, 1], [5, 6])  # A[1,0] -> At[0,1]


def test_basic_indexing():
    A = ia.ndarray([[1, 2], [3, 4], [5, 6]])

    # Single element
    elem = A[1]
    assert isinstance(elem, ia.ndarray)
    np.testing.assert_array_equal(elem.data, [3,4])

    # Slice
    subset = A[1:3]
    assert isinstance(subset, ia.ndarray)
    assert subset.shape == (2,)
    np.testing.assert_array_equal(subset.data, [[3, 4], [5, 6]])


def test_matrix_indexing():
    A = ia.ndarray([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

    # Row indexing
    row = A[0, :]
    assert row.shape == (2,)
    np.testing.assert_array_equal(row.data, [[1, 2], [3, 4]])

    # Element indexing
    elem = A[1, 0]
    assert elem.shape == ()
    np.testing.assert_array_equal(elem.data, [5,6])


def test_assignment():
    A = ia.ndarray([[1, 2], [3, 4]])
    new_interval = ia.ndarray([10,20])

    A[0] = new_interval
    np.testing.assert_array_equal(A[0].data, [10,20])

    # Assignment with raw data
    A[1] = [30, 40]
    np.testing.assert_array_equal(A[1].data, [30, 40])


def test_addition_operator():
    A = ia.ndarray([[1, 2], [3, 4]])
    B = ia.ndarray([[0.5, 1.5], [1, 2]])
    C = A + B

    assert isinstance(C, ia.ndarray)
    expected = [[1.5, 3.5], [4, 6]]
    np.testing.assert_array_equal(C.data, expected)


def test_subtraction_operator():
    A = ia.ndarray([[3, 5], [2, 4]])
    B = ia.ndarray([[1, 2], [1, 1]])
    C = A - B

    expected = [[1, 4], [1, 3]]
    np.testing.assert_array_equal(C.data, expected)


def test_multiplication_operator():
    A = ia.ndarray([[1, 2], [2, 3]])
    B = ia.ndarray([[2, 3], [1, 2]])
    C = A * B

    # [1,2] * [2,3] = [2, 6]
    # [2,3] * [1,2] = [2, 6]
    expected = [[2, 6], [2, 6]]
    np.testing.assert_array_equal(C.data, expected)


def test_division_operator():
    A = ia.ndarray([[4, 8], [6, 9]])
    B = ia.ndarray([[2, 4], [3, 3]])
    C = A / B

    assert isinstance(C, ia.ndarray)
    # Should use interval division
    assert C.shape == A.shape


def test_negation_operator():
    A = ia.ndarray([[1, 2], [-3, 4]])
    C = -A

    expected = [[-2, -1], [-4, 3]]
    np.testing.assert_array_equal(C.data, expected)


def test_matmul_operator():
    # Vector dot product
    u = ia.ndarray([[1, 2], [2, 3]])
    v = ia.ndarray([[2, 3], [1, 2]])
    result = u @ v

    assert isinstance(result, ia.ndarray)
    # Should be scalar result for 1D @ 1D
    assert result.shape == ()


def test_contains_method():
    A = ia.ndarray([[1, 3], [2, 5]])

    # Values inside intervals
    result = A.contains(2)
    np.testing.assert_array_equal(result, [True, True])

    # Values outside intervals
    result = A.contains(0)
    np.testing.assert_array_equal(result, [False, False])

    # Mixed
    result = A.contains([2, 6])
    np.testing.assert_array_equal(result, [True, False])


def test_is_empty_method():
    # Regular intervals
    A = ia.ndarray([[1, 2], [3, 4]])
    result = A.is_empty()
    np.testing.assert_array_equal(result, [False, False])

    # Empty intervals (NaN)
    B = ia.ndarray([[np.nan, np.nan], [1, 2]])
    result = B.is_empty()
    np.testing.assert_array_equal(result, [True, False])


def test_intersect_method():
    A = ia.ndarray([[1, 3], [5, 7]])
    B = ia.ndarray([[2, 4], [8, 9]])
    C = A.intersect(B)

    assert isinstance(C, ia.ndarray)
    # [1,3] ∩ [2,4] = [2,3], [5,7] ∩ [8,9] = empty
    np.testing.assert_array_equal(C.data[0], [2, 3])
    assert np.isnan(C.data[1, 0]) and np.isnan(C.data[1, 1])


def test_repr():
    A = ia.ndarray([[1, 2], [3, 4]])
    repr_str = repr(A)
    assert "array(" in repr_str
    assert "[[1" in repr_str


def test_str_0d():
    A = ia.ndarray([1,2])
    assert str(A) == "[1 2]"


def test_str_1d():
    A = ia.ndarray([[1,2], [3,4]])
    assert str(A) == "[[1 2]\n [3 4]]"


def test_empty_construction():
    with pytest.raises(ValueError):
        ia.ndarray([])


def test_single_interval():
    A = ia.ndarray([[1, 2]])
    assert A.shape == (1,)
    assert A.size == 1


def test_zero_width_intervals():
    A = ia.ndarray([[1, 1], [2, 2]])
    np.testing.assert_array_equal(A.width, [0, 0])
    np.testing.assert_array_equal(A.midpoint, [1, 2])


def test_broadcasting_compatibility():
    A = ia.ndarray([[1, 2]])  # Shape (1,)
    B = ia.ndarray([[3, 4], [5, 6]])  # Shape (2,)

    # Should broadcast
    C = A + B  # (1,) + (2,) -> (2,)
    assert C.shape == (2,)