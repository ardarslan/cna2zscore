from typing import Dict, Any

import torch
import torch.nn as nn


def normalize(cfg: Dict[str, Any], normalization, y: torch.Tensor):
    if cfg["normalization_type"] == "instance_normalization":
        y = y.view(*y.shape, 1).permute((0, 2, 1))
        y = normalization(y)
        y = y[:, 0, :]
    elif cfg["normalization_type"] == "batch_normalization":
        y = normalization(y)
    else:
        raise Exception(f"{cfg['normalization_type']} is not an implemented normalization type.")
    return y


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

        self.linear = nn.Linear(input_dimension, cfg["hidden_dimension"], bias=False)

        if cfg["normalization_type"] == "batch_normalization":
            self.normalization = nn.BatchNorm1d(num_features=cfg["hidden_dimension"])
        elif cfg["normalization_type"] == "instance_normalization":
            self.normalization = nn.InstanceNorm1d(num_features=cfg["hidden_dimension"])
        else:
            raise Exception(f"{cfg['normalization_type']} is not an implemented normalization type.")

        self.dropout = nn.Dropout(cfg["dropout"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.linear(x)
        y = normalize(cfg=self.cfg, normalization=self.normalization, y=y)
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
            self.linear_1 = nn.Linear(input_dimension, cfg["hidden_dimension"], bias=False)
            self.linear_2 = nn.Linear(cfg["hidden_dimension"], output_dimension, bias=False)

            if cfg["normalization_type"] == "batch_normalization":
                self.normalization_1 = nn.BatchNorm1d(num_features=cfg["hidden_dimension"])
                self.normalization_2 = nn.BatchNorm1d(num_features=output_dimension)
            elif cfg["normalization_type"] == "instance_normalization":
                self.normalization_1 = nn.InstanceNorm1d(num_features=cfg["hidden_dimension"])
                self.normalization_2 = nn.InstanceNorm1d(num_features=output_dimension)
            else:
                raise Exception(f"{cfg['normalization_type']} is not an implemented normalization type.")
        else:
            self.linear_1 = nn.Linear(input_dimension, output_dimension, bias=False)
            if cfg["normalization_type"] == "batch_normalization":
                self.normalization_1 = nn.BatchNorm1d(num_features=output_dimension)
            elif cfg["normalization_type"] == "instance_normalization":
                self.normalization_1 = nn.InstanceNorm1d(num_features=output_dimension)
            else:
                raise Exception(f"{cfg['normalization_type']} is not an implemented normalization type.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.linear_1(x)
        y = normalize(cfg=self.cfg, normalization=self.normalization_1, y=y)
        y = self.activation(y)
        if self.cfg["dropout"] > 0.0:
            y = self.dropout(y)

        if self.cfg["use_residual_connection"]:
            y = self.linear_2(y)
            y = normalize(cfg=self.cfg, normalization=self.normalization_2, y=y)
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
        self.linear = nn.Linear(cfg["hidden_dimension"], output_dimension)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.linear(x)
        return y
