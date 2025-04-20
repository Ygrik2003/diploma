from datetime import datetime
from typing import Dict, Any, Optional
import argparse
import os
import logging
import tempfile
from pathlib import Path


os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from ray import tune
from ray.tune import Checkpoint
from ray.tune.schedulers import ASHAScheduler
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import ray.cloudpickle as pickle
import torch
import torch.nn as nn


class NavierStockesModel(nn.Module):
    def __init__(self):
        super(NavierStockesModel, self).__init__()
        self.activation_func = torch.sin

        self.fc1 = nn.Linear(2, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 3)

    def forward(self, x):
        x = self.activation_func(self.fc1(x))
        x = self.activation_func(self.fc2(x))
        x = self.activation_func(self.fc3(x))
        return x


class NavierStockesSolver:
    def __init__(self, writer: SummaryWriter):
        self.loss = nn.MSELoss()
        self.writer: SummaryWriter = writer

    def domain_loss(
        self,
        xy: torch.Tensor,
        output: torch.Tensor,
        config: Dict[str, Any],
        current_epoch: int,
    ):
        u: torch.Tensor = output[:, 0]
        v: torch.Tensor = output[:, 1]
        p: torch.Tensor = output[:, 2]

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

        self.writer.add_scalar(
            "pde/continuity", torch.mean(torch.abs(continuity)), current_epoch
        )
        self.writer.add_scalar(
            "pde/momentum_x", torch.mean(torch.abs(momentum_x)), current_epoch
        )
        self.writer.add_scalar(
            "pde/momentum_y", torch.mean(torch.abs(momentum_y)), current_epoch
        )

        full_error = torch.cat((continuity, momentum_x, momentum_y))

        return self.loss(full_error, torch.zeros(full_error.shape))

    def wall_loss(
        self,
        xy: torch.Tensor,
        output: torch.Tensor,
        config: Dict[str, Any],
        current_epoch: int,
    ):
        u: torch.Tensor = output[:, 0]
        v: torch.Tensor = output[:, 1]
        p: torch.Tensor = output[:, 2]

        full_error = torch.cat((u, v))

        return self.loss(full_error, torch.zeros(full_error.shape))

    def inflow_loss(
        self,
        xy: torch.Tensor,
        output: torch.Tensor,
        config: Dict[str, Any],
        current_epoch: int,
    ):
        u: torch.Tensor = output[:, 0]
        v: torch.Tensor = output[:, 1]
        p: torch.Tensor = output[:, 2]

        return (
            self.loss(u, config["u_inflow"] * torch.ones(u.shape))
            + self.loss(v, torch.zeros(v.shape))
            + self.loss(p, config["p_inflow"] * torch.ones(p.shape))
        )

    def outflow_loss(
        self,
        xy: torch.Tensor,
        output: torch.Tensor,
        config: Dict[str, Any],
        current_epoch: int,
    ):
        u: torch.Tensor = output[:, 0]
        v: torch.Tensor = output[:, 1]
        p: torch.Tensor = output[:, 2]

        return self.loss(p, torch.zeros(p.shape))

    def exact_loss(
        self,
        xy: torch.Tensor,
        output: torch.Tensor,
        target: torch.Tensor,
        current_epoch: int,
    ):
        u: torch.Tensor = output[:, 0]
        v: torch.Tensor = output[:, 1]
        p: torch.Tensor = output[:, 2]

        full_loss = self.loss(u, target[:, 0]) + self.loss(v, target[:, 1])
        self.writer.add_scalar("loss/exact", full_loss, current_epoch)
        return full_loss


class ProblemTaskData:
    """
    Solve channel flow problem with next domain:
        wwwwwww
        wwdddddo
        wddwwww
       idddwwww
        wwwwwww
        where w -- wall, i -- inflow, 0 -- outflow and d -- domain
    """

    def __init__(self, path_to_data):
        exact = torch.tensor(pd.read_csv(path_to_data, header=None).to_numpy()).float()
        # exact = exact[~torch.any(exact.isnan(), dim=1)]

        self.exact_solution_coords = torch.tensor(exact.numpy()[:, :2]).float()
        self.exact_solution = torch.tensor(exact.numpy()[:, 2:]).float()

        dx = torch.min(self.exact_solution_coords[:, 0])
        dy = torch.min(self.exact_solution_coords[:, 1])

        domain_mask = torch.any(
            torch.stack(
                (
                    torch.all(
                        torch.stack(
                            (
                                torch.all(
                                    torch.stack(
                                        (
                                            dx < self.exact_solution_coords[:, 0],
                                            self.exact_solution_coords[:, 0] < 3 + dx,
                                        ),
                                        dim=-1,
                                    ),
                                    dim=-1,
                                ),
                                torch.all(
                                    torch.stack(
                                        (
                                            dy < self.exact_solution_coords[:, 1],
                                            self.exact_solution_coords[:, 1] < 1 + dy,
                                        ),
                                        dim=-1,
                                    ),
                                    dim=-1,
                                ),
                            ),
                        ),
                        dim=0,
                    ),
                    torch.all(
                        torch.stack(
                            (
                                torch.all(
                                    torch.stack(
                                        (
                                            1 + dx < self.exact_solution_coords[:, 0],
                                            self.exact_solution_coords[:, 0] < 3 + dx,
                                        ),
                                        dim=-1,
                                    ),
                                    dim=-1,
                                ),
                                torch.all(
                                    torch.stack(
                                        (
                                            1 + dy <= self.exact_solution_coords[:, 1],
                                            self.exact_solution_coords[:, 1] < 2 + dy,
                                        ),
                                        dim=-1,
                                    ),
                                    dim=-1,
                                ),
                            )
                        ),
                        dim=0,
                    ),
                    torch.all(
                        torch.stack(
                            (
                                torch.all(
                                    torch.stack(
                                        (
                                            2 + dx < self.exact_solution_coords[:, 0],
                                            self.exact_solution_coords[:, 0] < 3 + dx,
                                        ),
                                        dim=-1,
                                    ),
                                    dim=-1,
                                ),
                                torch.all(
                                    torch.stack(
                                        (
                                            2 + dy <= self.exact_solution_coords[:, 1],
                                            self.exact_solution_coords[:, 1] < 3 + dy,
                                        ),
                                        dim=-1,
                                    ),
                                    dim=-1,
                                ),
                            )
                        ),
                        dim=0,
                    ),
                    torch.all(
                        torch.stack(
                            (
                                torch.all(
                                    torch.stack(
                                        (
                                            3 + dx < self.exact_solution_coords[:, 0],
                                            self.exact_solution_coords[:, 0]
                                            < torch.max(
                                                self.exact_solution_coords[:, 0]
                                            ),
                                        ),
                                        dim=-1,
                                    ),
                                    dim=-1,
                                ),
                                torch.all(
                                    torch.stack(
                                        (
                                            2 + dy < self.exact_solution_coords[:, 1],
                                            self.exact_solution_coords[:, 1] < 3 + dy,
                                        ),
                                        dim=-1,
                                    ),
                                    dim=-1,
                                ),
                            )
                        ),
                        dim=0,
                    ),
                ),
                dim=-1,
            ),
            dim=-1,
        )
        self.channel_flow_domain = PINNDataset(
            self.exact_solution_coords[domain_mask, :],
            self.exact_solution[domain_mask, :],
        )

        wall_mask = torch.any(
            torch.stack(
                (
                    torch.all(
                        torch.stack(
                            (
                                torch.all(
                                    torch.stack(
                                        (
                                            dx <= self.exact_solution_coords[:, 0],
                                            self.exact_solution_coords[:, 0] <= 1 + dx,
                                        ),
                                        dim=-1,
                                    ),
                                    dim=-1,
                                ),
                                torch.all(
                                    torch.stack(
                                        (
                                            1 + dy <= self.exact_solution_coords[:, 1],
                                            self.exact_solution_coords[:, 1] <= 3 + dy,
                                        ),
                                        dim=-1,
                                    ),
                                    dim=-1,
                                ),
                            ),
                        ),
                        dim=0,
                    ),
                    torch.all(
                        torch.stack(
                            (
                                torch.all(
                                    torch.stack(
                                        (
                                            1 + dx <= self.exact_solution_coords[:, 0],
                                            self.exact_solution_coords[:, 0] <= 2 + dx,
                                        ),
                                        dim=-1,
                                    ),
                                    dim=-1,
                                ),
                                torch.all(
                                    torch.stack(
                                        (
                                            2 + dy <= self.exact_solution_coords[:, 1],
                                            self.exact_solution_coords[:, 1] <= 3 + dy,
                                        ),
                                        dim=-1,
                                    ),
                                    dim=-1,
                                ),
                            )
                        ),
                        dim=0,
                    ),
                    torch.all(
                        torch.stack(
                            (
                                torch.all(
                                    torch.stack(
                                        (
                                            3 + dx <= self.exact_solution_coords[:, 0],
                                            self.exact_solution_coords[:, 0]
                                            <= torch.max(
                                                self.exact_solution_coords[:, 0]
                                            ),
                                        ),
                                        dim=-1,
                                    ),
                                    dim=-1,
                                ),
                                torch.all(
                                    torch.stack(
                                        (
                                            dy <= self.exact_solution_coords[:, 1],
                                            self.exact_solution_coords[:, 1] <= 2 + dy,
                                        ),
                                        dim=-1,
                                    ),
                                    dim=-1,
                                ),
                            )
                        ),
                        dim=0,
                    ),
                ),
                dim=-1,
            ),
            dim=-1,
        )
        self.channel_flow_wall = PINNDataset(
            self.exact_solution_coords[wall_mask, :], self.exact_solution[wall_mask, :]
        )

        inflow_mask = torch.all(
            torch.stack(
                (
                    self.exact_solution_coords[:, 0] == dx,
                    torch.all(
                        torch.stack(
                            (
                                dy < self.exact_solution_coords[:, 1],
                                self.exact_solution_coords[:, 1] < 1 + dy,
                            ),
                            dim=-1,
                        ),
                        dim=-1,
                    ),
                )
            ),
            dim=0,
        )
        self.channel_flow_inflow = PINNDataset(
            self.exact_solution_coords[inflow_mask, :],
            self.exact_solution[inflow_mask, :],
        )

        outflow_mask = torch.all(
            torch.stack(
                (
                    self.exact_solution_coords[:, 0]
                    == torch.max(self.exact_solution_coords[:, 0]),
                    torch.all(
                        torch.stack(
                            (
                                2 + dy < self.exact_solution_coords[:, 1],
                                self.exact_solution_coords[:, 1] < 3 + dy,
                            ),
                            dim=-1,
                        ),
                        dim=-1,
                    ),
                )
            ),
            dim=0,
        )
        self.channel_flow_outflow = PINNDataset(
            self.exact_solution_coords[outflow_mask, :],
            self.exact_solution[outflow_mask, :],
        )

    def domain(self):
        return DataLoader(self.channel_flow_domain)

    def wall(self):
        return DataLoader(self.channel_flow_wall)

    def inflow(self):
        return DataLoader(self.channel_flow_inflow)

    def outflow(self):
        return DataLoader(self.channel_flow_outflow)

    def full(self):
        return DataLoader(
            PINNDataset(
                self.exact_solution_coords,
                self.exact_solution,
            )
        )


class ModelRunner:
    def __init__(
        self,
        model: nn.Module,
        optimizer_class: torch.optim.Optimizer,
        config: Dict[str, Any],
        writer: SummaryWriter = None,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.config = config

        self.optimizer_class = optimizer_class
        self.solver: NavierStockesSolver = config["solver"]

        self.writer = writer

        self.optimizer = None
        self.current_epoch = 0
        self.best_metric = float("inf")

    def setup_optimizer(self, layer_wise_params: Optional[Dict] = None):
        if layer_wise_params:
            param_groups = []
            for name, param in self.model.named_parameters():
                layer_config = next(
                    (
                        cfg
                        for layer_name, cfg in layer_wise_params.items()
                        if layer_name in name
                    ),
                    {"lr": self.config["learning_rate"]},
                )
                param_groups.append({"params": param, **layer_config})
            self.optimizer = self.optimizer_class(param_groups)
        else:
            self.optimizer = self.optimizer_class(
                self.model.parameters(), lr=self.config["learning_rate"]
            )

    def train_epoch(self, train_loader: DataLoader):

        self.model.train()
        total_loss = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            data.requires_grad_(True)

            output = self.model(data)
            loss = (
                self.solver.domain_loss(
                    data, output, config=self.config, current_epoch=self.current_epoch
                )
                + self.solver.wall_loss(
                    data, output, config=self.config, current_epoch=self.current_epoch
                )
                + self.solver.inflow_loss(
                    data, output, config=self.config, current_epoch=self.current_epoch
                )
                + self.solver.outflow_loss(
                    data, output, config=self.config, current_epoch=self.current_epoch
                )
                + self.solver.exact_loss(
                    data,
                    output,
                    target=target,
                    current_epoch=self.current_epoch,
                )
            )
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            self.writer.add_scalar(
                "Training/BatchLoss",
                loss.item(),
                self.current_epoch * len(train_loader) + batch_idx,
            )

        return total_loss / len(train_loader)

    def validate(self, val_loader: DataLoader):

        self.model.eval()
        total_loss = 0

        for data, target in val_loader:
            data, target = data.to(self.device), target.to(self.device)
            data.requires_grad_(True)
            output = self.model(data)
            loss = self.solver.domain_loss(
                data, output, config=self.config, current_epoch=self.current_epoch
            )
            total_loss += loss.item()

        return total_loss / len(val_loader)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        layer_wise_params: Optional[Dict] = None,
    ):

        self.setup_optimizer(layer_wise_params)

        for epoch in range(self.current_epoch, self.current_epoch + num_epochs):
            self.current_epoch = epoch

            train_loss = self.train_epoch(train_loader)

            val_loss = self.validate(val_loader)

            self.writer.add_scalar("Training/EpochLoss", train_loss, epoch)
            self.writer.add_scalar("Validation/Loss", val_loss, epoch)

    def tune_hyperparameters(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        param_space: Dict[str, Any],
        num_samples: int = 10,
        num_epochs: int = 1,
    ):

        def training_function(config):
            self.config.update(config)
            self.setup_optimizer()

            for epoch in range(num_epochs):
                train_loss = self.train_epoch(train_loader)
                val_loss = self.validate(val_loader)
                checkpoint_data = {
                    # "epoch": epoch,
                    # "net_state_dict": model.state_dict(),
                    # "optimizer_state_dict": optimizer.state_dict(),
                    # "scale_sin": config_hyperparams["scale_sin"],
                    # "scale_tanh": config_hyperparams["scale_tanh"],
                    # "scale_swish": config_hyperparams["scale_swish"],
                    # "scale_quadratic": config_hyperparams["scale_quadratic"],
                    # "scale_softplus": config_hyperparams["scale_softplus"],
                }
                with tempfile.TemporaryDirectory(dir="D:/tune") as checkpoint_dir:
                    data_path = Path(checkpoint_dir) / "data.pkl"
                    with open(data_path, "wb") as fp:
                        pickle.dump(checkpoint_data, fp)

                checkpoint = Checkpoint.from_directory(checkpoint_dir)
                tune.report(
                    {"loss": val_loss, "accuracy": 1 / val_loss},
                    training_loss=train_loss,
                    validation_loss=val_loss,
                    epoch=epoch,
                    checkpoint=checkpoint,
                )

        scheduler = ASHAScheduler(max_t=num_epochs, grace_period=1, reduction_factor=2)

        analysis = tune.run(
            training_function,
            config=param_space,
            num_samples=num_samples,
            scheduler=scheduler,
            checkpoint_at_end=True,
            storage_path="checkpoints",
        )

        best_trial = analysis.get_best_trial("validation_loss", "min", "last")
        print(f"Best trial config: {best_trial.config}")
        print(
            f"Best trial final validation loss: {best_trial.last_result['validation_loss']}"
        )

        self.config.update(best_trial.config)


class PINNDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume_from", help="Resume training from saved checkpoint", type=str
    )
    parser.add_argument("--experiment_name", default="experiment_1", type=str)

    args = parser.parse_args()

    tensorboard_dir = os.path.join("runs", args.experiment_name)
    task = ProblemTaskData("D:/work/diploma/data/vector_field_data.csv")
    writer = SummaryWriter(tensorboard_dir)
    model = NavierStockesModel()
    solver = NavierStockesSolver(writer)
    runner = ModelRunner(
        model=model,
        optimizer_class=torch.optim.Adam,
        config={
            "learning_rate": 0.001,
            "solver": solver,
            "Re": 0.001,
            "u_inflow": 1,
            "p_inflow": 1,
        },
        writer=writer,
    )

    layer_wise_params = {"conv": {"lr": 0.001}, "fc": {"lr": 0.0001}}

    if args.resume_from:
        runner.train(
            train_loader=task.full(),
            val_loader=task.full(),
            num_epochs=10,
            resume_from=args.resume_from,
        )
    else:
        runner.train(
            train_loader=task.full(),
            val_loader=task.full(),
            num_epochs=10,
            layer_wise_params=layer_wise_params,
        )

    # param_space = {
    #     "learning_rate": tune.loguniform(1e-3, 1e-4),
    #     "batch_size": tune.choice([16, 32, 64, 128]),
    # }

    # runner.tune_hyperparameters(
    #     train_loader=task.full(),
    #     val_loader=task.full(),
    #     param_space=param_space,
    #     num_samples=10,
    # )
