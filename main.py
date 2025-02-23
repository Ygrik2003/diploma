import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from flavors.activate_function import ActivateFunctionController, ActivateFunctions

Lx = 1.0
Ly = 1.0
nu = 0.05



class NavierStokesModel(nn.Module):
    def __init__(self):
        super(NavierStokesModel, self).__init__()

        self.activation_func = ActivateFunctionController(
            activate_func=ActivateFunctions.AdaptiveBlendingUnit, args=dict()
        ).get()

        self.fc1 = nn.Linear(2, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, 3)

    def forward(self, x):
        x = self.activation_func(self.fc1(x))
        x = self.activation_func(self.fc2(x))
        x = self.activation_func(self.fc3(x))
        return self.fc4(x)


model = NavierStokesModel()



def compute_pde_residual(xy):
    xy.requires_grad_(True)
    up = model(xy)
    u, v, p = up[:, 0], up[:, 1], up[:, 2]

    
    u_x = torch.autograd.grad(u.sum(), xy, create_graph=True)[0][:, 0]
    u_y = torch.autograd.grad(u.sum(), xy, create_graph=True)[0][:, 1]
    v_x = torch.autograd.grad(v.sum(), xy, create_graph=True)[0][:, 0]
    v_y = torch.autograd.grad(v.sum(), xy, create_graph=True)[0][:, 1]
    p_x = torch.autograd.grad(p.sum(), xy, create_graph=True)[0][:, 0]
    p_y = torch.autograd.grad(p.sum(), xy, create_graph=True)[0][:, 1]

    
    u_xx = torch.autograd.grad(u_x.sum(), xy, create_graph=True)[0][:, 0]
    u_yy = torch.autograd.grad(u_y.sum(), xy, create_graph=True)[0][:, 1]
    v_xx = torch.autograd.grad(v_x.sum(), xy, create_graph=True)[0][:, 0]
    v_yy = torch.autograd.grad(v_y.sum(), xy, create_graph=True)[0][:, 1]

    
    continuity = u_x + v_y

    
    momentum_x = u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
    momentum_y = u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)

    return continuity, momentum_x, momentum_y



def boundary_conditions():
    num_points = 100
    bottom_bc = torch.tensor(
        np.stack([np.random.uniform(0, Lx, num_points), np.zeros(num_points)], axis=-1),
        requires_grad=False,
    ).float()
    left_bc = torch.tensor(
        1

        np.stack([np.zeros(num_points), np.random.uniform(0, Ly, num_points)], axis=-1),
        requires_grad=False,
    ).float()
    top_bc = torch.tensor(
        np.stack(
            [np.random.uniform(0, Lx, num_points), np.full(num_points, Ly)], axis=-1
        ),
        requires_grad=False,
    ).float()
    right_bc = torch.tensor(
        np.stack(
            [np.full(num_points, Lx), np.random.uniform(0, Ly, num_points)], axis=-1
        ),
        requires_grad=False,
    ).float()

    bottom_predict = model(bottom_bc)
    left_predict = model(left_bc)
    top_predict = model(top_bc)
    right_predict = model(right_bc)

    
    u, v, p = left_predict[:, 0], left_predict[:, 1], left_predict[:, 2]
    bc_loss = torch.mean((u - (0.5 - torch.abs(0.5 - left_bc[:, 1] / Ly))) ** 2)

    
    for outflow in [bottom_predict, right_predict, top_predict]:
        u, v, p = outflow[:, 0], outflow[:, 1], outflow[:, 2]
        bc_loss += torch.mean(p**2)

    return bc_loss



def generate_data(num_points):
    x = np.random.uniform(0, Lx, num_points)
    y = np.random.uniform(0, Ly, num_points)
    xy = np.stack([x, y], axis=-1)
    return torch.tensor(xy, requires_grad=True).float()



def loss_function(xy):
    continuity, momentum_x, momentum_y = compute_pde_residual(xy)
    pde_loss = (
        torch.mean(continuity**2)
        + torch.mean(momentum_x**2)
        + torch.mean(momentum_y**2)
    )
    bc_loss = boundary_conditions()
    total_loss = pde_loss + bc_loss
    return total_loss



optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


num_epochs = 10000
num_points = 1000
for epoch in range(num_epochs):
    xy = generate_data(num_points)
    optimizer.zero_grad()
    loss = loss_function(xy)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

print("Training completed.")


nx, ny = 50, 50
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y)
XY = np.stack([X.flatten(), Y.flatten()], axis=-1)
XY_tensor = torch.tensor(XY, requires_grad=False).float()


with torch.no_grad():
    predictions = model(XY_tensor).numpy()

U = predictions[:, 0].reshape((ny, nx))
V = predictions[:, 1].reshape((ny, nx))
P = predictions[:, 2].reshape((ny, nx))


fig, axs = plt.subplots(1, 2, figsize=(15, 5))
axs[0].quiver(X, Y, U, V, scale=10, scale_units="xy")
axs[0].set_title("Velocity Vector Field (u, v)")
axs[0].set_xlabel("x")
axs[0].set_ylabel("y")

c3 = axs[1].contourf(X, Y, P, levels=50, cmap="viridis")
axs[1].set_title("Pressure")
fig.colorbar(c3, ax=axs[1])
axs[1].set_xlabel("x")
axs[1].set_ylabel("y")

plt.tight_layout()
plt.show()
