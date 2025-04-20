from abc import ABCMeta, abstractmethod
from enum import Enum

from numpy.typing import NDArray
import numpy as np
import torch 

dtype = np.float32


class grid_methods(Enum):
    rectangle = 1
    triangle = 2
    triangle_with_mask = 3


class Shape(metaclass=ABCMeta):

    def __init__(self, dim):
        self.dim = dim

    @abstractmethod
    def get_random_points(
        self,
        density: float,
    ) -> NDArray[dtype]:
        """
        Generate random points from geometry with given density.

        :param density: density of generated points
        :type density: float
        :return: Numpy array with shape (count, dim)
        :rtype: NDArray
        """
        raise NotImplementedError("Not implemented for base class")

    @abstractmethod
    def get_grid_points(
        self,
        resolution: float,
        method: grid_methods,
        mask: NDArray[dtype] | None = None,
    ) -> NDArray[dtype]:
        """
        Generate points inside geometry with given method.

        :param resolution: parameter for determinate size grid in specific method
        :type resolution: float
        :param method: Selecting method for creating grid
        :type method: grid_method
        :param mask: Need for method with variable density of points
        :type mask: NDArray[dtype] | None
        :return: Numpy array with shape (count_points, dim). Count points cant be determinate before this call
        :rtype: NDArray
        """
        raise NotImplementedError("Not implemented for base class")

    @abstractmethod
    def get_boundary_points(self, density: float) -> NDArray[dtype]:
        """
        Generate points on boundary of geometry.

        :param density: density of generated points
        :type density: float
        :return: Numpy array with shape (count, dim)
        :rtype: NDArray
        """
        raise NotImplementedError("Not implemented for base class")

    @abstractmethod
    def contains(self, points: NDArray[dtype]) -> NDArray[np.bool_]:
        """
        Check points belongs to domain.

        :param points: points to check
        :type count: NDArray[dtype]
        :return: Numpy array with point.shape size
        :rtype: NDArray[np.bool_]
        """
        raise NotImplementedError("Not implemented for base class")

    def apply(self, x: torch.Tensor)

    @property
    def dimension(self) -> int:
        """
        :return: return dimmension of this Shape
        :rtype: NDArray[np.bool_]
        """
        return self.dim
