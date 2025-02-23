import torch
from torch import nn, Tensor

from enum import Enum
from typing import Union


DTYPE = Union[float, nn.Parameter]


class ActivateFunctions(Enum):
    AdaptiveBlendingUnit = 1


class ActivateFunctionController:
    def __init__(self, activate_func: ActivateFunctions, args: dict):
        self.activate_func = activate_func
        self.args = args

    def get(self):
        if self.activate_func == ActivateFunctions.AdaptiveBlendingUnit:
            return AdaptiveBlendingUnit(*self.args)
        else:
            raise ValueError(
                f"Activation function {self.activate_func} not implemented"
            )


class SinActivation(nn.Module):
    def __init__(
        self, a: DTYPE = 1.0, trainable: bool = False, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.trainable = trainable
        self.a = nn.Parameter(torch.tensor(a), requires_grad=trainable)

    def forward(self, x: Tensor) -> Tensor:
        return torch.sin(self.a * x)


class TanhActivation(nn.Module):
    def __init__(
        self, a: DTYPE = 1.0, trainable: bool = False, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.trainable = trainable
        self.a = nn.Parameter(
            torch.tensor(a, dtype=torch.get_default_dtype(), requires_grad=trainable)
        )
        self.act = nn.Tanh()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.a * x)


class SiLUActivation(nn.Module):
    def __init__(
        self, a: DTYPE = 1.0, trainable: bool = False, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.trainable = trainable
        self.a = nn.Parameter(
            torch.tensor(a, dtype=torch.get_default_dtype(), requires_grad=trainable)
        )
        self.act = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.a * x)


class QuadraticActivation(nn.Module):
    def __init__(
        self, a: DTYPE = 1.0, trainable: bool = False, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.trainable = trainable
        self.a = nn.Parameter(torch.tensor(a), requires_grad=trainable)

    def forward(self, x: Tensor) -> Tensor:
        return 1 / (1 + (self.a * x) ** 2)


class SoftplusActivation(nn.Module):
    def __init__(
        self, a: DTYPE = 1.0, trainable: bool = False, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.trainable = trainable
        self.a = torch.tensor(
            a, dtype=torch.get_default_dtype(), requires_grad=trainable
        )
        self.act = nn.Softplus()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.a * x)


class AdaptiveBlendingUnit(nn.Module):
    def __init__(self, count_act_func: int = 5, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        assert 1 < count_act_func
        assert count_act_func < 6

        self.count_act_func = count_act_func

        self.weights = nn.Parameter(
            torch.zeros(
                count_act_func, dtype=torch.get_default_dtype(), requires_grad=True
            )
        )
        self.softmax = nn.Softmax(dim=0)

        if count_act_func == 2:
            self.scale_sin = 1.0
            self.sin = SinActivation(a=self.scale_sin, trainable=True)
            self.scale_tanh = 1.0
            self.tanh = TanhActivation(a=self.scale_tanh, trainable=True)
            self.acts = lambda x: torch.stack([self.sin(x), self.tanh(x)], dim=-1)

        elif count_act_func == 3:
            self.scale_sin = 1.0
            self.sin = SinActivation(a=self.scale_sin, trainable=True)
            self.scale_tanh = 1.0
            self.tanh = TanhActivation(a=self.scale_tanh, trainable=True)
            self.scale_swish = 1.0
            self.swish = SiLUActivation(a=self.scale_swish, trainable=True)
            self.acts = lambda x: torch.stack(
                [self.sin(x), self.tanh(x), self.swish(x)], dim=-1
            )

        elif count_act_func == 4:
            self.scale_sin = 1.0
            self.sin = SinActivation(a=self.scale_sin, trainable=True)
            self.scale_tanh = 1.0
            self.tanh = TanhActivation(a=self.scale_tanh, trainable=True)
            self.scale_swish = 1.0
            self.swish = SiLUActivation(a=self.scale_swish, trainable=True)
            self.scale_quadratic = 1.0
            self.quadratic = QuadraticActivation(a=self.scale_quadratic, trainable=True)
            self.acts = lambda x: torch.stack(
                [self.sin(x), self.tanh(x), self.swish(x), self.quadratic(x)], dim=-1
            )

        elif count_act_func == 5:
            self.scale_sin = 1.0
            self.sin = SinActivation(a=self.scale_sin, trainable=True)
            self.scale_tanh = 1.0
            self.tanh = TanhActivation(a=self.scale_tanh, trainable=True)
            self.scale_swish = 1.0
            self.swish = SiLUActivation(a=self.scale_swish, trainable=True)
            self.scale_quadratic = 1.0
            self.quadratic = QuadraticActivation(a=self.scale_quadratic, trainable=True)
            self.scale_softplus = 1.0
            self.softplus = SoftplusActivation(a=self.scale_softplus, trainable=True)
            self.acts = lambda x: torch.stack(
                [
                    self.sin(x),
                    self.tanh(x),
                    self.swish(x),
                    self.quadratic(x),
                    self.softplus(x),
                ],
                dim=-1,
            )

    def forward(self, x: Tensor) -> Tensor:
        weights_softmax = self.softmax(self.weights)  # [num_act,]

        # Shape: [*x.shape, num_act] @ [num_act,] = x.shape
        out = torch.matmul(self.acts(x), weights_softmax)

        return out
