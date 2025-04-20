from api.equations.equation import Equation
from api.geometry.shapes.shape import Shape

from abc import ABC, abstractmethod

import torch


class BoundaryCondition(ABC):
    def __init__(self, boundary_points: torch.Tensor):
        self.boundary_points = boundary_points

    @abstractmethod
    def compute_bc_loss(self, model) -> torch.Tensor:
        pass



class DirichletBC(BoundaryCondition):
    def __init__(self, boundary_points: torch.Tensor, boundary_values: torch.Tensor):
        super().__init__(boundary_points)
        self.boundary_values = boundary_values
    
    def compute_bc_loss(self, model) -> torch.Tensor:
        pred = model(self.boundary_points)
        return torch.mean((pred - self.boundary_values) ** 2)