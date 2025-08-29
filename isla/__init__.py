"""
Isla - Interval Linear Algebra Package

A Python package for interval linear algebra operations.
"""

__version__ = "0.1.0"
__author__ = "Muhammad Maaz"
__email__ = "m.maaz@mail.utoronto.ca"

from .core import ndarray, array, full, zeros, ones, add, subtract, multiply, divide, negate, intersect, dot, transpose, eye, reciprocal
from . import linalg

__all__ = [
    "__version__",
    "ndarray",
    "array",
    "full",
    "zeros",
    "ones",
    "add",
    "subtract",
    "multiply",
    "divide",
    "negate",
    "intersect",
    "dot",
    "transpose",
    "eye",
    "reciprocal",
    "linalg",
]