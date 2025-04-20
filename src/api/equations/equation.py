from api.geometry.shapes.shape import Shape

from abc import ABC, abstractmethod

import torch

torch.set_default_dtype(torch.float32)


class Equation(ABC):
    @abstractmethod
    def compute_loss(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Abstract class")


class SystemOfEquations(ABC):
    def __init__(self, domain_geom: Shape, boundary_geom: Shape, initial_geom: Shape):
        self._domain_geometry = domain_geom
        self._boundary_geometry = boundary_geom
        self._initial_geometry = initial_geom
    
    def compute_loss(self, x: torch.Tensor) -> torch.Tensor:
        domain = self._domain_geometry.apply(x)
        boundary = self._boundary_geometry.apply(x)
        initial = self._initial_geometry.apply(x)

    @abstractmethod
    def compute_loss(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Abstract class")
