"""
Cost functions module.

This package provides a common interface for different cost functions
used in machine learning models.
"""

from .cost import CostFunction, MeanSquaredError

__all__ = [
    "CostFunction",
    "MeanSquaredError",
]