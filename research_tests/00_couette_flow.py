import os

# os.environ["RAY_PICKLE_VERBOSE_DEBUG"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["MPLBACKEND"] = "Agg"
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt

from ray import tune
from ray import train
from ray.tune.schedulers import ASHAScheduler

Lx = 5.0
Ly = 1.0
T = 1.0
nu = 0.01

U = 1


class REAct(nn.Module):
    def __init__(
        self,
        a=1.0,
        b=1.0,
        c=1.0,
        d=1.0,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.a = a
        self.b = b
        self.c = c
        self.d = d

        self.acts = lambda x: (1 - torch.exp(a * x + b)) / (1 + torch.exp(c * x + d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        out = self.acts(x)

        return out


class NavierStokesModel(nn.Module):
    def __init__(self, neurons, react_params):
        super(NavierStokesModel, self).__init__()

        self.activation_func = REAct(*react_params)

        self.fc = []

        self.fc1 = nn.Linear(3, neurons[0])
        self.fc.append(self.fc1)
        try:
            self.fc2 = nn.Linear(neurons[0], neurons[1])
            self.fc.append(self.fc2)
            self.fc3 = nn.Linear(neurons[1], neurons[2])
            self.fc.append(self.fc3)
            self.fc4 = nn.Linear(neurons[2], neurons[3])
            self.fc.append(self.fc4)
        except IndexError as e:
            pass
        self.fc5 = nn.Linear(neurons[-1], 3)
        self.fc.append(self.fc5)

    def forward(self, x):
        for layer in self.fc:
            x = self.activation_func(layer(x))
        return x


def compute_pde(model, xyt, writer):
    global global_epoch
    xyt.requires_grad_(True)
    up = model(xyt)
    u, v, p = up[:, 0], up[:, 1], up[:, 2]

    u_x = torch.autograd.grad(u.sum(), xyt, create_graph=True)[0][:, 0]
    u_y = torch.autograd.grad(u.sum(), xyt, create_graph=True)[0][:, 1]
    u_t = torch.autograd.grad(u.sum(), xyt, create_graph=True)[0][:, 2]
    v_x = torch.autograd.grad(v.sum(), xyt, create_graph=True)[0][:, 0]
    v_y = torch.autograd.grad(v.sum(), xyt, create_graph=True)[0][:, 1]
    v_t = torch.autograd.grad(v.sum(), xyt, create_graph=True)[0][:, 2]
    p_x = torch.autograd.grad(p.sum(), xyt, create_graph=True)[0][:, 0]
    p_y = torch.autograd.grad(p.sum(), xyt, create_graph=True)[0][:, 1]

    u_xx = torch.autograd.grad(u_x.sum(), xyt, create_graph=True)[0][:, 0]
    u_yy = torch.autograd.grad(u_y.sum(), xyt, create_graph=True)[0][:, 1]
    v_xx = torch.autograd.grad(v_x.sum(), xyt, create_graph=True)[0][:, 0]
    v_yy = torch.autograd.grad(v_y.sum(), xyt, create_graph=True)[0][:, 1]

    continuity = u_x + v_y

    momentum_x = u_t + u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
    momentum_y = v_t + u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)

    writer.add_scalar("pde/continuity", torch.abs(torch.mean(continuity)), global_epoch)
    writer.add_scalar("pde/momentum_x", torch.abs(torch.mean(momentum_x)), global_epoch)
    writer.add_scalar("pde/momentum_y", torch.abs(torch.mean(momentum_y)), global_epoch)

    return continuity, momentum_x, momentum_y


def boundary_conditions(model, writer):
    num_points = 1000
    t = T * np.random.random(num_points)
    bottom_bc = torch.tensor(
        np.stack(
            [np.random.uniform(0, Lx, num_points), np.zeros(num_points), t], axis=-1
        ),
        requires_grad=False,
    ).float()
    left_bc = torch.tensor(
        np.stack(
            [np.zeros(num_points), np.random.uniform(0, Ly, num_points), t], axis=-1
        ),
        requires_grad=False,
    ).float()
    top_bc = torch.tensor(
        np.stack(
            [np.random.uniform(0, Lx, num_points), np.full(num_points, Ly), t], axis=-1
        ),
        requires_grad=False,
    ).float()
    right_bc = torch.tensor(
        np.stack(
            [np.full(num_points, Lx), np.random.uniform(0, Ly, num_points), t], axis=-1
        ),
        requires_grad=False,
    ).float()

    bottom_predict = model(bottom_bc)
    left_predict = model(left_bc)
    top_predict = model(top_bc)
    right_predict = model(right_bc)

    u, v, p = left_predict[:, 0], left_predict[:, 1], left_predict[:, 2]
    left_loss = torch.mean(p**2)
    u, v, p = top_predict[:, 0], top_predict[:, 1], top_predict[:, 2]
    top_loss = torch.mean((u - U) ** 2 + v**2)
    u, v, p = bottom_predict[:, 0], bottom_predict[:, 1], bottom_predict[:, 2]
    bottom_loss = torch.mean(u**2 + v**2)
    u, v, p = right_predict[:, 0], right_predict[:, 1], right_predict[:, 2]
    right_loss = torch.mean(p**2)

    writer.add_scalar("bc/bottom", torch.abs(torch.mean(bottom_loss)), global_epoch)
    writer.add_scalar("bc/left", torch.abs(torch.mean(left_loss)), global_epoch)
    writer.add_scalar("bc/right", torch.abs(torch.mean(right_loss)), global_epoch)
    writer.add_scalar("bc/top", torch.abs(torch.mean(top_loss)), global_epoch)

    return right_loss + bottom_loss + top_loss + left_loss


def generate_data(num_points):
    x = np.random.uniform(0, Lx, num_points)
    y = np.random.uniform(0, Ly, num_points)
    t = np.random.uniform(0, T, num_points)
    xyt = np.stack([x, y, t], axis=-1)
    return torch.tensor(xyt, requires_grad=True).float()


def loss_function(model, xyt, writer):
    continuity, momentum_x, momentum_y = compute_pde(model, xyt, writer)
    pde_loss = (
        torch.mean(continuity**2)
        + torch.mean(momentum_x**2)
        + torch.mean(momentum_y**2)
    )
    bc_loss = boundary_conditions(model, writer)
    total_loss = pde_loss + bc_loss
    writer.add_scalar("total/loss", total_loss, global_epoch)
    return total_loss


def plot_flow_data(model, writer: SummaryWriter):
    nx, ny, nt = 50, 50, 4
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    t = np.linspace(0, T, nt)
    X, Y, t_grid = np.meshgrid(x, y, t)
    XYT = np.stack([X.flatten(), Y.flatten(), t_grid.flatten()], axis=-1)
    XYT_tensor = torch.tensor(XYT, requires_grad=False).float()

    X, Y = np.meshgrid(x, y)
    with torch.no_grad():
        predictions = model(XYT_tensor).numpy()

    U_calc = predictions[:, 0].reshape((ny, nx, nt))
    V_calc = predictions[:, 1].reshape((ny, nx, nt))
    # P_calc = predictions[:, 2].reshape((ny, nx, nt))

    U_exact = np.tile(U / Ly * y, (nx)).reshape(nx, ny).T
    V_exact = 0 * y[None, :] * x[:, None]

    U_error = np.abs(U_exact - U_calc[:, :, 0])
    V_error = np.abs(V_exact - V_calc[:, :, 0])

    error = np.sqrt(U_error**2 + V_error**2)

    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(111)
    c3 = ax.contourf(X, Y, error, levels=50, cmap="viridis")
    ax.set_title("Error")
    fig.colorbar(c3)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    plt.tight_layout()
    writer.add_figure("total/error", figure=fig, global_step=global_epoch)

    plt.close()


global_epoch = 0


def train(config):
    global global_epoch
    writer = SummaryWriter()

    if config["optimizer"] == 1:
        optim = torch.optim.Adam
    elif config["optimizer"] == 2:
        optim = torch.optim.Adagrad
    elif config["optimizer"] == 3:
        optim = torch.optim.Adamax
    elif config["optimizer"] == 4:
        optim = torch.optim.ASGD
    elif config["optimizer"] == 5:
        optim = torch.optim.RMSprop

    model = NavierStokesModel(config["neurons"], config["react_params"])
    optimizer = optim(model.parameters(), lr=config["lr"])
    for epoch in range(config["num_epochs"]):
        global_epoch += 1
        xyt = generate_data(config["num_points"])
        optimizer.zero_grad()
        loss = loss_function(model, xyt, writer)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            plot_flow_data(model, writer)
    tune.report({"loss": float(loss), "accuracy": 1 / float(loss)})


config = {
    "react_params": tune.choice(
        [
            [1, -1, 0.5, 1],
            [0.2, 1.2, 3, -1.1],
            [0.8, -4, -3.2, 0.2],
            [1, 1, 1, 1],
        ]
    ),
    "neurons": tune.choice(
        [
            # [32, 64, 32],
            # [64, 32, 64],
            # [16, 32, 64],
            # [64, 32, 16],
            # [64, 16, 64],
            # [16, 64, 16],
            # [16, 64, 32],
            # [64, 16, 32],
            # [32, 64, 16],
            # [32, 16, 64],
            [16, 16],
            # [32, 32],
            # [64, 64],
            # [128, 128],
        ]
    ),
    "num_points": tune.choice([100, 1000]),
    "num_epochs": 10,
    "optimizer": tune.choice([1, 2, 3, 4, 5]),
    "lr": tune.choice([1e-1, 1e-2, 1e-3]),
}

scheduler = ASHAScheduler(metric="loss", mode="min")

result = tune.run(
    train,
    resources_per_trial={"cpu": 2, "gpu": 0},
    config=config,
    scheduler=scheduler,
    num_samples=4 * 1 * 2 * 1 * 5 * 3,
    storage_path="D:/checkpoints",
)
