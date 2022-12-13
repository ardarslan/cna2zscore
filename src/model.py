import math
from typing import Dict, Any

import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from layer import NonLinearLayer, OutputLayer


class LinearModel(nn.Module):
    def __init__(self, cfg: Dict[str, Any], input_dimension: int, output_dimension: int):
        super().__init__()
        self.cfg = cfg
        self.linear = OutputLayer(cfg=cfg, input_dimension=input_dimension, output_dimension=output_dimension)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class MLP(nn.Module):
    def __init__(self, cfg: Dict[str, Any], input_dimension: int, output_dimension: int):
        super().__init__()
        self.cfg = cfg

        if self.cfg["num_nonlinear_layers"] == 0:
            raise Exception("Please use LinearModel if num_nonlinear_layers == 0.")

        self.layers = nn.ModuleList()

        for i in range(self.cfg["num_nonlinear_layers"]):
            if i == 0:
                self.layers.append(NonLinearLayer(cfg=cfg, input_dimension=input_dimension, output_dimension=cfg["hidden_dimension"]))
            else:
                self.layers.append(NonLinearLayer(cfg=cfg, input_dimension=cfg["hidden_dimension"], output_dimension=cfg["hidden_dimension"]))

        self.layers.append(OutputLayer(cfg=cfg, input_dimension=cfg["hidden_dimension"], output_dimension=output_dimension))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.clone()
        for layer in self.layers:
            y = layer(y)
        return y


class ResConMLP(nn.Module):
    def __init__(self, cfg: Dict[str, Any], input_dimension: int, output_dimension: int):
        super().__init__()
        self.cfg = cfg

        if self.cfg["num_nonlinear_layers"] == 0:
            raise Exception("Please use LinearModel if num_nonlinear_layers == 0.")

        self.input_dimension = input_dimension
        self.output_dimension = output_dimension

        self.layers = nn.ModuleList()

        for i in range(self.cfg["num_nonlinear_layers"]):
            if i == 0:
                self.layers.append(NonLinearLayer(cfg=cfg, input_dimension=input_dimension, output_dimension=cfg["hidden_dimension"]))
            else:
                self.layers.append(NonLinearLayer(cfg=cfg, input_dimension=cfg["hidden_dimension"], output_dimension=cfg["hidden_dimension"]))

        self.layers.append(OutputLayer(cfg=cfg, input_dimension=cfg["hidden_dimension"], output_dimension=output_dimension))

        self._prepare_ResCon_W()
        self._prepare_ResCon_B()

    def _prepare_ResCon_W(self):
        if self.cfg["rescon_diagonal_W"]:
            self.ResConW = Parameter(torch.zeros((self.output_dimension, self.output_dimension)), requires_grad=False)
            ResConW_diagonal = torch.empty((1, self.output_dimension))
            nn.init.kaiming_uniform_(ResConW_diagonal, a=math.sqrt(5))
            for i in range(self.ResConW.shape[0]):
                self.ResConW[i, i] += ResConW_diagonal[0][i]
                self.ResConW[i, i].requires_grad = True
        else:
            self.ResConW = Parameter(torch.zeros((self.output_dimension, self.output_dimension)), requires_grad=True)
            nn.init.kaiming_uniform_(self.ResConW, a=math.sqrt(5))

    def _prepare_ResCon_B(self):
        self.ResConB = Parameter(torch.zeros(self.output_dimension), requires_grad=True)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.ResConW)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.ResConB, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.clone()
        for layer in self.layers:
            y = layer(y)
        return y + F.linear(x[:, :self.output_dimension], self.ResConW, self.ResConB)
