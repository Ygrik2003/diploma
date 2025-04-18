import tempfile
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from ray import tune
from ray.tune import Checkpoint
from ray.tune.schedulers import ASHAScheduler
import ray.cloudpickle as pickle
import numpy as np

from flavors.activate_function import ActivateFunctionController, ActivateFunctions

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse


config = {}

config["Lx"] = 5.0
config["Ly"] = 1.0
config["T"] = 1.0
config["mu"] = 0.001
config["U"] = 5


config["barrier_Lx"] = 0.5
config["barrier_Ly"] = 0.5

config["barrier_x"] = 1
config["barrier_y"] = config["Ly"] / 2 - config["barrier_Ly"] / 2

config["Re"] = 1 * config["U"] * config["barrier_Ly"] / config["mu"]


writer = SummaryWriter()


def visualize(model):
    nx, ny, nt = 50, 50, 4
    x = np.linspace(0, config["Lx"], nx)
    y = np.linspace(0, config["Ly"], ny)
    t = np.linspace(0, config["T"], nt)
    X, Y, t_grid = np.meshgrid(x, y, t)
    XYT = np.stack([X.flatten(), Y.flatten(), t_grid.flatten()], axis=-1)
    XYT_tensor = torch.tensor(XYT, requires_grad=False).float()

    X, Y = np.meshgrid(x, y)

    with torch.no_grad():
        predictions = model(XYT_tensor).numpy()

    U = predictions[:, 0].reshape((ny, nx, nt))
    V = predictions[:, 1].reshape((ny, nx, nt))
    P = predictions[:, 2].reshape((ny, nx, nt))

    for t in [0, nt - 1]:
        rect = patches.Rectangle(
            (config["barrier_x"], config["barrier_y"]),
            config["barrier_Lx"],
            config["barrier_Ly"],
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        axs[0].quiver(
            X,
            Y,
            U[:, :, int(t)],
            V[:, :, int(t)],
            scale=10,
            scale_units="xy",
        )

        axs[0].set_title(
            f"Velocity Vector Field (u, v), t = {config['T'] * t / (nt-1)} Re = {config['Re']}"
        )
        axs[0].set_xlabel("x")
        axs[0].set_ylabel("y")
        axs[0].add_patch(rect)

        c3 = axs[1].contourf(X, Y, P[:, :, int(t)], levels=50, cmap="viridis")
        axs[1].set_title("Pressure")
        fig.colorbar(c3, ax=axs[1])
        axs[1].set_xlabel("x")
        axs[1].set_ylabel("y")

        ################################################################################

        # c1 = axs[0].contourf(X, Y, U[:, :, int(t)], levels=50, cmap="viridis")
        # axs[0].set_title(
        #     f"Velocity Field u, t = {config['T'] * t / (nt-1)} Re = {config['Re']}"
        # )
        # fig.colorbar(c1, ax=axs[0])
        # axs[0].set_xlabel("x")
        # axs[0].set_ylabel("y")

        # c2 = axs[1].contourf(X, Y, V[:, :, int(t)], levels=50, cmap="viridis")
        # axs[1].set_title(
        #     f"Velocity Field v, t = {config['T'] * t / (nt-1)} Re = {config['Re']}"
        # )
        # fig.colorbar(c2, ax=axs[1])
        # axs[1].set_xlabel("x")
        # axs[1].set_ylabel("y")

        #############################################################

        # c1 = axs[0].contourf(
        #     X,
        #     Y,
        #     np.sqrt(U[:, :, int(t)] ** 2 + V[:, :, int(t)] ** 2),
        #     levels=50,
        #     cmap="viridis",
        # )
        # axs[0].set_title(
        #     f"Velocity Field u, t = {config['T'] * t / (nt-1)} Re = {config['Re']}"
        # )
        # fig.colorbar(c1, ax=axs[0])
        # axs[0].set_xlabel("x")
        # axs[0].set_ylabel("y")

        # c2 = axs[1].contourf(X, Y, P[:, :, int(t)], levels=50, cmap="viridis")
        # axs[1].set_title("Pressure")
        # fig.colorbar(c2, ax=axs[1])
        # axs[1].set_xlabel("x")
        # axs[1].set_ylabel("y")

        ###################################################################

        plt.tight_layout()
        plt.show()


class NavierStokesModel(nn.Module):
    def __init__(
        self,
        scale_sin=1,
        scale_tanh=1,
        scale_swish=1,
        scale_quadratic=1,
        scale_softplus=1,
    ):
        super(NavierStokesModel, self).__init__()

        self.activation_func = ActivateFunctionController(
            activate_func=ActivateFunctions.AdaptiveBlendingUnit,
            args=(
                5,
                scale_sin,
                scale_tanh,
                scale_swish,
                scale_quadratic,
                scale_softplus,
            ),
        ).get()

        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 3)

    def forward(self, x):
        x = self.activation_func(self.fc1(x))
        x = self.activation_func(self.fc2(x))
        x = self.activation_func(self.fc3(x))
        return x


def compute_pde(u, v, p, xyt):

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

    momentum_x = u_t + u * u_x + v * u_y + p_x - (u_xx + u_yy) / config["Re"]
    momentum_y = v_t + u * v_x + v * v_y + p_y - (v_xx + v_yy) / config["Re"]

    writer.add_scalar("pde/continuity", torch.mean(continuity), global_epoch)
    writer.add_scalar("pde/momentum_x", torch.mean(momentum_x), global_epoch)
    writer.add_scalar("pde/momentum_y", torch.mean(momentum_y), global_epoch)

    return continuity, momentum_x, momentum_y


def boundary_conditions(model):
    num_points = 1000
    t = config["T"] * np.random.random(num_points)
    bottom_bc = torch.tensor(
        np.stack(
            [np.random.uniform(0, config["Lx"], num_points), np.zeros(num_points), t],
            axis=-1,
        ),
        requires_grad=False,
    ).float()
    left_bc = torch.tensor(
        np.stack(
            [np.zeros(num_points), np.random.uniform(0, config["Ly"], num_points), t],
            axis=-1,
        ),
        requires_grad=False,
    ).float()
    top_bc = torch.tensor(
        np.stack(
            [
                np.random.uniform(0, config["Lx"], num_points),
                np.full(num_points, config["Ly"]),
                t,
            ],
            axis=-1,
        ),
        requires_grad=False,
    ).float()
    right_bc = torch.tensor(
        np.stack(
            [
                np.full(num_points, config["Lx"]),
                np.random.uniform(0, config["Ly"], num_points),
                t,
            ],
            axis=-1,
        ),
        requires_grad=False,
    ).float()

    bottom_predict = model(bottom_bc)
    left_predict = model(left_bc)
    top_predict = model(top_bc)
    right_predict = model(right_bc)

    u, v, p = left_predict[:, 0], left_predict[:, 1], left_predict[:, 2]
    left_loss = torch.mean(
        (u - (0.5 - config["U"] * (0.5 - left_bc[:, 1] / config["Ly"]) ** 2)) ** 2
        + v**2
        + (p - 1) ** 2
        # torch.sqrt((u - config["U"]) ** 2 + v**2 + (p - 1) ** 2)
    )

    u, v, p = bottom_predict[:, 0], bottom_predict[:, 1], bottom_predict[:, 2]
    bottom_loss = torch.mean(u**2 + v**2)
    u, v, p = top_predict[:, 0], top_predict[:, 1], top_predict[:, 2]
    top_loss = torch.mean(u**2 + v**2)
    u, v, p = right_predict[:, 0], right_predict[:, 1], right_predict[:, 2]
    right_loss = torch.mean(p**2)

    writer.add_scalar("bc/bottom_loss", bottom_loss, global_epoch)
    writer.add_scalar("bc/left_loss", left_loss, global_epoch)
    writer.add_scalar("bc/right_loss", right_loss, global_epoch)
    writer.add_scalar("bc/top_loss", top_loss, global_epoch)

    return bottom_loss + left_loss + right_loss + top_loss


global_epoch = 0


def boundary_conditions_barrier(model):
    xys = generate_bc_barrier_data()

    bottom_bc, left_bc, top_bc, right_bc, inside_bc = xys

    bottom_predict = model(bottom_bc)
    left_predict = model(left_bc)
    top_predict = model(top_bc)
    right_predict = model(right_bc)
    # inside_predict = model(inside_bc)

    # u, v, p = inside_predict[:, 0], inside_predict[:, 1], inside_predict[:, 2]
    # bc_loss = torch.mean(u**2 + v**2)

    u, v, p = bottom_predict[:, 0], bottom_predict[:, 1], bottom_predict[:, 2]
    bottom_loss = torch.mean(u**2 + v**2)
    u, v, p = left_predict[:, 0], left_predict[:, 1], left_predict[:, 2]
    left_loss = torch.mean(u**2 + v**2)
    u, v, p = right_predict[:, 0], right_predict[:, 1], right_predict[:, 2]
    right_loss = torch.mean(u**2 + v**2)
    u, v, p = top_predict[:, 0], top_predict[:, 1], top_predict[:, 2]
    top_loss = torch.mean(u**2 + v**2)

    writer.add_scalar("bc_barrier/bottom_loss", bottom_loss, global_epoch)
    writer.add_scalar("bc_barrier/left_loss", left_loss, global_epoch)
    writer.add_scalar("bc_barrier/right_loss", right_loss, global_epoch)
    writer.add_scalar("bc_barrier/top_loss", top_loss, global_epoch)

    return bottom_loss + left_loss + right_loss + top_loss


def loss_function(model, xyt):
    xyt.requires_grad_(True)
    up = model(xyt)
    u, v, p = up[:, 0], up[:, 1], up[:, 2]

    continuity, momentum_x, momentum_y = compute_pde(u, v, p, xyt)
    pde_loss = torch.sqrt(
        torch.mean(continuity**2)
        + torch.mean(momentum_x**2)
        + torch.mean(momentum_y**2)
    )
    bc_loss = boundary_conditions(model)
    bc_barrier_loss = boundary_conditions_barrier(model)

    total_loss = pde_loss + 10 * bc_loss + bc_barrier_loss

    return total_loss


def generate_data(num_points, test_fill=0.8):
    x_train = np.random.uniform(0, config["Lx"], int((1 - test_fill) * num_points))
    y_train = np.random.uniform(0, config["Ly"], int((1 - test_fill) * num_points))
    t_train = np.random.uniform(0, config["T"], int((1 - test_fill) * num_points))
    x_test = np.random.uniform(0, config["Lx"], int(test_fill * num_points))
    y_test = np.random.uniform(0, config["Ly"], int(test_fill * num_points))
    t_test = np.random.uniform(0, config["T"], int(test_fill * num_points))
    xyt_train = np.stack([x_train, y_train, t_train], axis=-1)
    xyt_test = np.stack([x_test, y_test, t_test], axis=-1)
    return (
        torch.tensor(xyt_train, requires_grad=True).float(),
        torch.tensor(xyt_test, requires_grad=True).float(),
    )


def generate_bc_barrier_data():
    num_points = 1000
    x = np.random.uniform(
        config["barrier_x"], config["barrier_x"] + config["barrier_Lx"], num_points
    )
    y = np.random.uniform(
        config["barrier_y"], config["barrier_y"] + config["barrier_Ly"], num_points
    )
    t = config["T"] * np.random.random(num_points)

    bottom = config["barrier_y"] * np.ones(num_points)
    left = config["barrier_x"] * np.ones(num_points)
    top = (config["barrier_y"] + config["barrier_Ly"]) * np.ones(num_points)
    right = (config["barrier_x"] + config["barrier_Lx"]) * np.ones(num_points)

    bottom_bc = np.stack([x, bottom, t], axis=-1)
    left_bc = np.stack([left, y, t], axis=-1)
    top_bc = np.stack([x, top, t], axis=-1)
    right_bc = np.stack([right, y, t], axis=-1)
    inside_bc = np.stack([x, y, t], axis=-1)

    return (
        torch.tensor(bottom_bc, requires_grad=True).float(),
        torch.tensor(left_bc, requires_grad=True).float(),
        torch.tensor(top_bc, requires_grad=True).float(),
        torch.tensor(right_bc, requires_grad=True).float(),
        torch.tensor(inside_bc, requires_grad=True).float(),
    )


g_trainset, g_testset = generate_data(1000)


def load_data():
    global g_trainset, g_testset
    return g_trainset, g_testset


def train_model(config_hyperparams):
    model = NavierStokesModel(
        config_hyperparams["scale_sin"],
        config_hyperparams["scale_tanh"],
        config_hyperparams["scale_swish"],
        config_hyperparams["scale_quadratic"],
        config_hyperparams["scale_softplus"],
    )

    device = "cpu"
    model.to(device)

    optimizer_adagrad = optim.Adagrad(model.parameters(), lr=config_hyperparams["lr"])
    optimizer_adam = optim.Adam(model.parameters(), lr=config_hyperparams["lr"])
    optimizer_asgd = optim.ASGD(model.parameters(), lr=config_hyperparams["lr"])

    trainset, testset = load_data()

    def train(optimizer, epoch):
        global global_epoch
        start_epoch = 0
        for epoch in range(start_epoch, epoch):
            global_epoch += 1
            optimizer.zero_grad()

            loss = loss_function(model, trainset)
            loss.backward()
            optimizer.step()

            # checkpoint_data = {
            #     "epoch": epoch,
            #     "net_state_dict": model.state_dict(),
            #     "optimizer_state_dict": optimizer.state_dict(),
            #     "scale_sin": config_hyperparams["scale_sin"],
            #     "scale_tanh": config_hyperparams["scale_tanh"],
            #     "scale_swish": config_hyperparams["scale_swish"],
            #     "scale_quadratic": config_hyperparams["scale_quadratic"],
            #     "scale_softplus": config_hyperparams["scale_softplus"],
            # }
            # with tempfile.TemporaryDirectory(dir="D:/tune") as checkpoint_dir:
            #     data_path = Path(checkpoint_dir) / "data.pkl"
            #     with open(data_path, "wb") as fp:
            #         pickle.dump(checkpoint_data, fp)

            #     checkpoint = Checkpoint.from_directory(checkpoint_dir)
            #     tune.report(
            #         {
            #             "loss": float(torch.mean(loss)),
            #             "accuracy": 1 / float(torch.mean(loss)),
            #         },
            #         checkpoint=checkpoint,
            #     )

    train(optimizer_adagrad, 1000)
    train(optimizer_adam, 8000)
    train(optimizer_asgd, 1000)

    print("Finished Training")

    return model


def test_accuracy(model):
    trainset, testset = load_data()

    loss = loss_function(model, testset)

    return 1 / loss


def main(num_samples=10, max_num_epochs=10, gpus_per_trial=2):
    config = {
        "scale_sin": 1,
        "scale_tanh": 0.1,
        "scale_swish": 0.7,
        "scale_quadratic": 1,
        "scale_softplus": 0.7,
        "lr": 1e-3,
    }
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2,
    )
    # result = tune.run(
    #     train_model,
    #     resources_per_trial={"cpu": 1, "gpu": gpus_per_trial},
    #     config=config,
    #     num_samples=num_samples,
    #     scheduler=scheduler,
    #     storage_path="D:/checkpoints",
    # )
    result = train_model(config)
    visualize(result)

    # best_trial = result.get_best_trial("loss", "min", "last")
    # print(f"Best trial config: {best_trial.config}")
    # print(f"Best trial final loss: {best_trial.last_result['loss']}")

    # best_trained_model = NavierStokesModel(
    #     best_trial.config["scale_sin"],
    #     best_trial.config["scale_tanh"],
    #     best_trial.config["scale_swish"],
    #     best_trial.config["scale_quadratic"],
    #     best_trial.config["scale_softplus"],
    # )
    # best_trained_model.to(device)

    # best_checkpoint = result.get_best_checkpoint(
    #     trial=best_trial, metric="accuracy", mode="max"
    # )
    # with best_checkpoint.as_directory() as checkpoint_dir:
    #     data_path = Path(checkpoint_dir) / "data.pkl"
    #     with open(data_path, "rb") as fp:
    #         best_checkpoint_data = pickle.load(fp)

    #     best_trained_model.load_state_dict(best_checkpoint_data["net_state_dict"])
    #     test_acc = test_accuracy(best_trained_model)
    #     print("Best trial test set accuracy: {}".format(test_acc))

    # visualize(best_trained_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load and visualize a Navier-Stokes model checkpoint."
    )
    parser.add_argument("--checkpoint", type=str, help="Path to the checkpoint file.")
    args = parser.parse_args()

    if args.checkpoint:
        # Load the checkpoint
        with open(args.checkpoint, "rb") as fp:
            checkpoint_data = pickle.load(fp)

        # Create a model and load its state
        model = NavierStokesModel(
            checkpoint_data["scale_sin"],
            checkpoint_data["scale_tanh"],
            checkpoint_data["scale_swish"],
            checkpoint_data["scale_quadratic"],
            checkpoint_data["scale_softplus"],
        )
        device = "cpu"
        model.to(device)
        model.load_state_dict(checkpoint_data["net_state_dict"])

        # Visualize the model
        visualize(model)
    else:
        main(num_samples=100, max_num_epochs=10000, gpus_per_trial=0)
