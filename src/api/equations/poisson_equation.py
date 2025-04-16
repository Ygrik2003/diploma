from equation import *


class PoissonEquation(Equation):
    def __init__(self):
        # Example parameter (could be learned)
        self.source_term = torch.nn.Parameter(torch.tensor(1.0))

    def evaluate(
        self, inputs: Tensor, parameters: Optional[Dict[str, Tensor]] = None
    ) -> Tensor:
        x = inputs.requires_grad_(True)

        # Compute u values
        u = self.network(x)

        # First derivatives
        du = torch.autograd.grad(u.sum(), x, create_graph=True)[0]

        # Second derivatives
        d2u = torch.autograd.grad(du.pow(2).sum(), x, create_graph=True)[0]

        # Poisson equation: ∇²u = f
        return d2u.sum(dim=1, keepdim=True) - self.source_term

    def parameters(self) -> Dict[str, Tensor]:
        return {"source_term": self.source_term}

    def boundary_conditions(self) -> List[Equation]:
        return [DirichletBC(0.0)]  # Example boundary condition


class DirichletBC(Equation):
    def __init__(self, target_value: float):
        self.target = torch.tensor(target_value)

    def evaluate(
        self, inputs: Tensor, parameters: Optional[Dict[str, Tensor]] = None
    ) -> Tensor:
        # Assume network is defined elsewhere
        u = self.network(inputs)
        return u - self.target.to(inputs.device)

    def parameters(self) -> Dict[str, Tensor]:
        return {}
