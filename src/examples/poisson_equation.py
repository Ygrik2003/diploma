from api.equations.poisson_equation import *
from api.geometry.shapes.rectangle import *

system = PDESystem(
    differential_equations=[PoissonEquation()], bc_equations=[DirichletBC(0.0)]
)

optimizer = torch.optim.Adam(system.parameters().values())

for epoch in range(1000):
    # Sample points
    domain_pts = torch.rand(1000, 2).requires_grad_(True)
    boundary_pts = np.asarray([0, 0], dtype=dtype)

    # Compute loss
    loss = system.composite_loss(domain_pts, boundary_pts)

    # Optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
