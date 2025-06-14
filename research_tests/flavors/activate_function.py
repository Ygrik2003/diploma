import torch
from torch import nn, Tensor

from enum import Enum
from typing import Union

# torch.set_default_dtype(torch.float64)


DTYPE = Union[float, nn.Parameter]


class ActivateFunctions(Enum):
    SinActivation = 1
    TanhActivation = 2
    SiLUActivation = 3
    QuadraticActivation = 4
    SoftplusActivation = 5
    AdaptiveBlendingUnit = 6
    REAct = 7


class ActivateFunctionController:
    def __init__(self, activate_func: ActivateFunctions, args: dict):
        self.activate_func = activate_func
        self.args = args

    def get(self):
        if self.activate_func == ActivateFunctions.SinActivation:
            return SinActivation(*self.args)
        elif self.activate_func == ActivateFunctions.TanhActivation:
            return TanhActivation(*self.args)
        elif self.activate_func == ActivateFunctions.SiLUActivation:
            return SiLUActivation(*self.args)
        elif self.activate_func == ActivateFunctions.QuadraticActivation:
            return QuadraticActivation(*self.args)
        elif self.activate_func == ActivateFunctions.SoftplusActivation:
            return SoftplusActivation(*self.args)
        elif self.activate_func == ActivateFunctions.AdaptiveBlendingUnit:
            return AdaptiveBlendingUnit(*self.args)
        elif self.activate_func == ActivateFunctions.REAct:
            return REAct(*self.args)
        else:
            raise ValueError(
                f"Activation function {self.activate_func} not implemented"
            )


class SinActivation(nn.Module):
    def __init__(self, a=1.0, trainable: bool = False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.trainable = trainable
        self.a = a

    def forward(self, x: Tensor) -> Tensor:
        return torch.sin(self.a * x)


class TanhActivation(nn.Module):
    def __init__(self, a=1.0, trainable: bool = False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.trainable = trainable
        self.a = a
        self.act = nn.Tanh()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.a * x)


class SiLUActivation(nn.Module):
    def __init__(
        self, a: DTYPE = 1.0, trainable: bool = False, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.trainable = trainable
        self.a = a
        self.act = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.a * x)


class QuadraticActivation(nn.Module):
    def __init__(
        self, a: DTYPE = 1.0, trainable: bool = False, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.trainable = trainable
        self.a = a

    def forward(self, x: Tensor) -> Tensor:
        return 1 / (1 + (self.a * x) ** 2)


class SoftplusActivation(nn.Module):
    def __init__(
        self, a: DTYPE = 1.0, trainable: bool = False, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.trainable = trainable
        self.a = a
        self.act = nn.Softplus()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.a * x)


class AdaptiveBlendingUnit(nn.Module):
    def __init__(
        self,
        count_act_func: int = 5,
        scale_sin=1.0,
        scale_tanh=1.0,
        scale_swish=1.0,
        scale_quadratic=1.0,
        scale_softplus=1.0,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        assert 1 < count_act_func
        assert count_act_func < 6

        self.count_act_func = count_act_func
        self.scale_sin = scale_sin
        self.scale_tanh = scale_tanh
        self.scale_swish = scale_swish
        self.scale_quadratic = scale_quadratic
        self.scale_softplus = scale_softplus

        self.weights = nn.Parameter(
            torch.zeros(
                count_act_func, dtype=torch.get_default_dtype(), requires_grad=True
            )
        )
        self.softmax = nn.Softmax(dim=0)

        if count_act_func == 2:
            self.sin = SinActivation(a=self.scale_sin, trainable=True)
            self.tanh = TanhActivation(a=self.scale_tanh, trainable=True)
            self.acts = lambda x: torch.stack([self.sin(x), self.tanh(x)], dim=-1)

        elif count_act_func == 3:
            self.sin = SinActivation(a=self.scale_sin, trainable=True)
            self.tanh = TanhActivation(a=self.scale_tanh, trainable=True)
            self.swish = SiLUActivation(a=self.scale_swish, trainable=True)
            self.acts = lambda x: torch.stack(
                [self.sin(x), self.tanh(x), self.swish(x)], dim=-1
            )

        elif count_act_func == 4:
            self.sin = SinActivation(a=self.scale_sin, trainable=True)
            self.tanh = TanhActivation(a=self.scale_tanh, trainable=True)
            self.swish = SiLUActivation(a=self.scale_swish, trainable=True)
            self.quadratic = QuadraticActivation(a=self.scale_quadratic, trainable=True)
            self.acts = lambda x: torch.stack(
                [self.sin(x), self.tanh(x), self.swish(x), self.quadratic(x)], dim=-1
            )

        elif count_act_func == 5:
            self.sin = SinActivation(a=self.scale_sin, trainable=True)
            self.tanh = TanhActivation(a=self.scale_tanh, trainable=True)
            self.swish = SiLUActivation(a=self.scale_swish, trainable=True)
            self.quadratic = QuadraticActivation(a=self.scale_quadratic, trainable=True)
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
        weights_softmax = self.softmax(self.weights)

        out = torch.matmul(self.acts(x), weights_softmax)

        return out


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

    def forward(self, x: Tensor) -> Tensor:

        out = self.acts(x)

        return out
