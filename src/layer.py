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


class NonLinearLayer(nn.Module):
    def __init__(self, cfg: Dict[str, Any], input_dimension: int, output_dimension: int):
        super().__init__()
        self.cfg = cfg

        if cfg["hidden_activation"] == "leaky_relu":
            self.activation = nn.LeakyReLU(inplace=True)
        elif cfg["hidden_activation"] == "relu":
            self.activation = nn.ReLU(inplace=True)
        else:
            raise Exception(f"Hidden layer activation {cfg['hidden_activation']} is not implemented yet.")

        self.linear = nn.Linear(input_dimension, output_dimension, bias=False)

        if cfg["normalization_type"] == "batch_normalization":
            self.normalization = nn.BatchNorm1d(num_features=output_dimension)
        elif cfg["normalization_type"] == "instance_normalization":
            self.normalization = nn.InstanceNorm1d(num_features=output_dimension)
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


class OutputLayer(nn.Module):
    def __init__(self, cfg: Dict[str, Any], input_dimension: int, output_dimension: int):
        super().__init__()
        self.cfg = cfg
        self.linear = nn.Linear(input_dimension, output_dimension)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.linear(x)
        return y
