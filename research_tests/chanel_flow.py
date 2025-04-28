from itertools import product
from pathlib import Path
from ray import tune
from ray.tune import Checkpoint
from ray.tune.schedulers import ASHAScheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor
import argparse
import io
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
import ray.cloudpickle as pickle
import torch
import torch.nn as nn
import torch.optim as optim


torch.set_default_dtype(torch.float64)

from flavors.activate_function import ActivateFunctionController, ActivateFunctions


config = {}

config["Lx"] = 20.0
config["Ly"] = 3.0

config["U"] = 1


config["Re"] = 1000


exact_solution = torch.tensor(
    pd.read_csv("data/vector_field_data.csv", header=None).to_numpy()
)
exact_solution = exact_solution[~torch.any(exact_solution.isnan(), dim=1)]

exact_solution_coords = torch.tensor(np.array(exact_solution)[:, :2])
exact_solution_u = torch.tensor(np.array(exact_solution)[:, 2])
exact_solution_v = torch.tensor(np.array(exact_solution)[:, 3])


global_epoch = 0
writer = SummaryWriter()


def plot_result(model, display=False):
    nx, ny = 50, 50
    x = np.linspace(0, config["Lx"], nx)
    y = np.linspace(0, config["Ly"], ny)
    X, Y = np.meshgrid(x, y)
    XY = np.stack([X.flatten(), Y.flatten()], axis=-1)
    XY_tensor = torch.tensor(XY, requires_grad=False)

    X, Y = np.meshgrid(x, y)

    with torch.no_grad():
        predictions = model(XY_tensor).numpy()

    U = predictions[:, 0].reshape((ny, nx))
    V = predictions[:, 1].reshape((ny, nx))
    P = predictions[:, 2].reshape((ny, nx))

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].quiver(
        X,
        Y,
        U,
        V,
        scale=1,
        scale_units="xy",
    )

    axs[0].set_title(
        f"optimizer_adam, Velocity Vector Field (u, v), Re = {config['Re']}"
    )
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")

    c3 = axs[1].contourf(X, Y, P, levels=50, cmap="viridis")
    axs[1].set_title("Pressure")
    fig.colorbar(c3, ax=axs[1])
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("y")

    buf = io.BytesIO()
    plt.savefig(buf, format="jpeg")
    buf.seek(0)

    image = PIL.Image.open(buf)
    image = ToTensor()(image).unsqueeze(0)
    writer.add_image("Result", image, global_epoch, dataformats="NCHW")
    if display:
        plt.show()
    else:
        plt.close()


class NavierStokesModel(nn.Module):
    def __init__(
        self,
        a=1,
        b=-1,
        c=0.5,
        d=-1,
    ):
        super(NavierStokesModel, self).__init__()

        self.activation_func = nn.Tanh()
        # self.activation_func = ActivateFunctionController(
        #     activate_func=ActivateFunctions.REAct,
        #     args=(
        #         a,
        #         b,
        #         c,
        #         d,
        #     ),
        # ).get()

        self.fc1 = nn.Linear(2, 30)
        self.fc2 = nn.Linear(30, 30)
        # self.fc3 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(30, 30)
        self.fc4 = nn.Linear(30, 30)
        self.fc5 = nn.Linear(30, 30)
        self.fc6 = nn.Linear(30, 3)
        # self.fc7 = nn.Linear(30, 3)

    def forward(self, x):
        x = self.activation_func(self.fc1(x))
        x = self.activation_func(self.fc2(x))
        x = self.activation_func(self.fc3(x))
        x = self.activation_func(self.fc4(x))
        x = self.activation_func(self.fc5(x))
        x = self.activation_func(self.fc6(x))
        # x = self.activation_func(self.fc7(x))
        # x = self.activation_func(self.fc8(x))
        return x


def compute_pde(u, v, p, xy):

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

    momentum_x = u * u_x + v * u_y + p_x - (u_xx + u_yy) / config["Re"]
    momentum_y = u * v_x + v * v_y + p_y - (v_xx + v_yy) / config["Re"]

    writer.add_scalar("pde/continuity", torch.mean(torch.abs(continuity)), global_epoch)
    writer.add_scalar("pde/momentum_x", torch.mean(torch.abs(momentum_x)), global_epoch)
    writer.add_scalar("pde/momentum_y", torch.mean(torch.abs(momentum_y)), global_epoch)

    return continuity, momentum_x, momentum_y


def boundary_conditions(model):
    num_points_x = 100
    num_points_y = 100

    x = np.linspace(0, config["Lx"], num_points_x)
    y = np.linspace(0, config["Ly"], num_points_y)

    selected_XYT_1 = np.array(
        list(product(x[(x > 0) & (x < 1)], y[(y > 1) & (y < 3)])), dtype=np.float64
    )
    selected_XYT_2 = np.array(
        list(product(x[(x >= 1) & (x < 2)], y[(y > 2) & (y < 3)])), dtype=np.float64
    )
    selected_XYT_3 = np.array(
        list(product(x[(x > 3) & (x < 20)], y[(y > 0) & (y < 2)])), dtype=np.float64
    )

    selected_XY_inflow = np.array(
        list(product(x[np.isclose(x, 0)], y[(y > 0) & (y < 1)])),
        dtype=np.float64,
    )
    selected_XY_outflow = np.array(
        list(product(x[np.isclose(x, config["Lx"])], y[(y > 2) & (y < 3)])),
        dtype=np.float64,
    )

    predict_1 = model(torch.tensor(selected_XYT_1))
    predict_2 = model(torch.tensor(selected_XYT_2))
    predict_3 = model(torch.tensor(selected_XYT_3))
    predict_inflow = model(torch.tensor(selected_XY_inflow))
    predict_outflow = model(torch.tensor(selected_XY_outflow))

    u, v, p = predict_1[:, 0], predict_1[:, 1], predict_1[:, 2]
    loss_1 = torch.mean(torch.sqrt(u**2 + v**2))

    u, v, p = predict_2[:, 0], predict_2[:, 1], predict_2[:, 2]
    loss_2 = torch.mean(torch.sqrt(u**2 + v**2))

    u, v, p = predict_3[:, 0], predict_3[:, 1], predict_3[:, 2]
    loss_3 = torch.mean(torch.sqrt(u**2 + v**2))

    u, v, p = predict_inflow[:, 0], predict_inflow[:, 1], predict_inflow[:, 2]
    loss_inflow = 10 * torch.mean(
        torch.sqrt((u - config["U"]) ** 2 + v**2 + (p - 1) ** 2)
    )

    u, v, p = predict_outflow[:, 0], predict_outflow[:, 1], predict_outflow[:, 2]
    loss_outflow = 10 * torch.mean(torch.abs(p))

    writer.add_scalar("bc/loss_1", loss_1, global_epoch)
    writer.add_scalar("bc/loss_2", loss_2, global_epoch)
    writer.add_scalar("bc/loss_3", loss_3, global_epoch)
    writer.add_scalar("bc/loss_inflow", loss_inflow, global_epoch)
    writer.add_scalar("bc/loss_outflow", loss_outflow, global_epoch)

    return loss_1 + loss_2 + loss_3 + loss_inflow + loss_outflow


def loss_function(model, xy):
    xy.requires_grad_(True)
    up = model(xy)
    u, v, p = up[:, 0], up[:, 1], up[:, 2]

    continuity, momentum_x, momentum_y = compute_pde(u, v, p, xy)
    pde_loss = (
        torch.sum(torch.sqrt(continuity**2))
        + torch.sum(torch.sqrt(momentum_x**2))
        + torch.sum(torch.sqrt(momentum_y**2))
    )
    bc_loss = boundary_conditions(model)

    matches = []
    for coord in exact_solution_coords:
        matches.append((xy == coord).all(dim=1).any())

    u_exact = exact_solution_u[matches]
    v_exact = exact_solution_v[matches]

    exact_loss = torch.mean(torch.abs(u - u_exact) + 100 * torch.abs(v - v_exact))

    writer.add_scalar("exact_loss", exact_loss, global_epoch)

    total_loss = pde_loss + exact_loss + bc_loss

    return total_loss


def generate_data(num_points):
    train_random_indices = np.random.choice(
        exact_solution_coords.shape[0],
        size=num_points,
        replace=False,
    )
    train = exact_solution_coords[train_random_indices]

    return torch.tensor(train, requires_grad=True)


g_trainset = generate_data(10000)


def load_data():
    global g_trainset
    return g_trainset


def train_model(config_hyperparams):
    model = NavierStokesModel(
        config_hyperparams["a"],
        config_hyperparams["b"],
        config_hyperparams["c"],
        config_hyperparams["d"],
    )

    device = "cpu"
    model.to(device)

    optimizer_adagrad = optim.Adagrad(model.parameters(), lr=config_hyperparams["lr"])
    optimizer_adam = optim.Adam(model.parameters(), lr=config_hyperparams["lr"])
    optimizer_asgd = optim.ASGD(model.parameters(), lr=config_hyperparams["lr"])
    optimizer_rmsprop = optim.RMSprop(model.parameters(), lr=config_hyperparams["lr"])

    trainset = load_data()

    def train(optimizer, epoch):
        global global_epoch
        start_epoch = 0
        for epoch in range(start_epoch, epoch):
            print(f"Epoch: {epoch}")
            global_epoch += 1
            optimizer.zero_grad()

            loss = loss_function(model, trainset)
            if torch.any(loss.isnan()):
                breakpoint()
                loss = loss_function(model, trainset)

            loss.backward()
            optimizer.step()

            if epoch % 100 == 0:
                plot_result(model)

    try:
        # train(optimizer_adagrad, 10000)
        train(optimizer_adam, 100000)
        # train(optimizer_asgd, 10000)
        # train(optimizer_rmsprop, 10000)
    except KeyboardInterrupt as e:
        pass
    print("Finished Training")

    return model


def main():
    config = {
        "a": 1,
        "b": 1,
        "c": 1,
        "d": 1,
        "lr": 1e-4,
    }

    result = train_model(config)
    plot_result(result, display=True)


if __name__ == "__main__":
    main()
