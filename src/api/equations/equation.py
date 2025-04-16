from abc import ABCMeta, abstractmethod
from typing import List, Dict, Optional

import torch
from torch import Tensor

torch.set_default_dtype(torch.float32)


class Equation(metaclass=ABCMeta):
    @abstractmethod
    def evaluate(
        self, inputs: Tensor, parameters: Optional[Dict[str, Tensor]] = None
    ) -> Tensor:
        """
        Evaluate the equation at given input points

        Args:
            inputs: Tensor of shape (batch_size, input_dim)
            parameters: Dictionary of equation parameters (if any)

        Returns:
            Tensor of shape (batch_size, output_dim)
        """
        raise NotImplementedError

    @abstractmethod
    def parameters(self) -> Dict[str, Tensor]:
        """
        Returns trainable parameters for the equation
        """
        raise NotImplementedError

    @abstractmethod
    def boundary_conditions(self) -> List["Equation"]:
        """
        Returns boundary condition equations
        """
        raise NotImplementedError

    def residual(
        self, inputs: Tensor, parameters: Optional[Dict[str, Tensor]] = None
    ) -> Tensor:
        """
        Compute residual (for minimization), defaults to L2 norm of evaluation
        """
        return self.evaluate(inputs, parameters).pow(2).mean()


class EquationSystem(Equation):
    @abstractmethod
    def equations(self) -> List[Equation]:
        """
        List of component equations in the system
        """
        raise NotImplementedError

    def evaluate(
        self, inputs: Tensor, parameters: Optional[Dict[str, Tensor]] = None
    ) -> Tensor:
        outputs = [eq.evaluate(inputs, parameters) for eq in self.equations()]
        return torch.cat(outputs, dim=-1)

    def parameters(self) -> Dict[str, Tensor]:
        params = {}
        for eq in self.equations():
            params.update(eq.parameters())
        return params

    def boundary_conditions(self) -> List[Equation]:
        bcs = []
        for eq in self.equations():
            bcs.extend(eq.boundary_conditions())
        return bcs

    def residual(
        self, inputs: Tensor, parameters: Optional[Dict[str, Tensor]] = None
    ) -> Tensor:
        residuals = [eq.residual(inputs, parameters) for eq in self.equations()]
        return sum(residuals)  # Sum of individual residuals


class PDESystem(EquationSystem):
    """
    Specialized system for partial differential equations
    """

    def __init__(
        self, differential_equations: List[Equation], bc_equations: List[Equation]
    ):
        self._diff_eqs = differential_equations
        self._bc_eqs = bc_equations

    def equations(self) -> List[Equation]:
        return self._diff_eqs + self._bc_eqs

    def domain_loss(self, domain_points: Tensor) -> Tensor:
        return sum(eq.residual(domain_points) for eq in self._diff_eqs)

    def boundary_loss(self, boundary_points: Tensor) -> Tensor:
        return sum(eq.residual(boundary_points) for eq in self._bc_eqs)

    def composite_loss(self, domain_points: Tensor, boundary_points: Tensor) -> Tensor:
        return self.domain_loss(domain_points) + self.boundary_loss(boundary_points)
