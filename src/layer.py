from typing import Dict, Any

import torch
import torch.nn as nn


class InputLayer(nn.Module):
    def __init__(self, cfg: Dict[str, Any], input_dimension: int):
        super().__init__()
        self.cfg = cfg

        if cfg["hidden_activation"] == "leaky_relu":
            self.activation = nn.LeakyReLU(inplace=True)
        elif cfg["hidden_activation"] == "relu":
            self.activation = nn.ReLU(inplace=True)
        else:
            raise Exception(f"Hidden layer activation {cfg['hidden_activation']} is not implemented yet.")

        self.w = nn.Linear(input_dimension, cfg["hidden_dimension"], bias=bool(1 - cfg["use_batch_normalization"]))
        self.bn = nn.BatchNorm1d(cfg["hidden_dimension"])
        self.dropout = nn.Dropout(cfg["dropout"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.w(x)
        if self.cfg["use_batch_normalization"]:
            y = self.bn(y)
        y = self.activation(y)
        if self.cfg["dropout"] > 0.0:
            y = self.dropout(y)
        return y


class HiddenLayer(nn.Module):
    def __init__(self, cfg: Dict[str, Any], input_dimension: int, output_dimension: int):
        super().__init__()

        self.cfg = cfg

        if cfg["hidden_activation"] == "leaky_relu":
            self.activation = nn.LeakyReLU(inplace=True)
        elif cfg["hidden_activation"] == "relu":
            self.activation = nn.ReLU(inplace=True)
        else:
            raise Exception(f"Hidden layer activation {cfg['hidden_activation']} is not implemented yet.")

        if self.cfg["dropout"] > 0.0:
            self.dropout = nn.Dropout(cfg['dropout'])

        if self.cfg["use_residual_connection"]:
            self.w1 = nn.Linear(input_dimension, cfg["hidden_dimension"], bias=bool(1 - cfg["use_batch_normalization"]))
            self.w2 = nn.Linear(cfg["hidden_dimension"], output_dimension)
            if cfg["use_batch_normalization"]:
                self.bn1 = nn.BatchNorm1d(cfg["hidden_dimension"])
                self.bn2 = nn.BatchNorm1d(output_dimension)
        else:
            self.w1 = nn.Linear(input_dimension, output_dimension, bias=bool(1 - cfg["use_batch_normalization"]))
            if cfg["use_batch_normalization"]:
                self.bn1 = nn.BatchNorm1d(output_dimension)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.w1(x)
        if self.cfg["use_batch_normalization"]:
            y = self.bn1(y)
        y = self.activation(y)
        if self.cfg["dropout"] > 0.0:
            y = self.dropout(y)

        if self.cfg["use_residual_connection"]:
            y = self.w2(y)
            if self.cfg["use_batch_normalization"]:
                y = self.bn2(y)
            y = self.activation(y)
            if self.cfg["dropout"] > 0.0:
                y = self.dropout(y)
            return y + x
        else:
            return y


class OutputLayer(nn.Module):
    def __init__(self, cfg: Dict[str, Any], output_dimension: int):
        super().__init__()
        self.cfg = cfg
        self.w = nn.Linear(cfg["hidden_dimension"], output_dimension)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.w(x)
        return y
