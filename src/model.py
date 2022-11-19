import torch
import torch.nn as nn

from typing import Dict, Any
from layer import HiddenLayer, OutputLayer


class MLP(nn.Module):
    def __init__(self, cfg: Dict[str, Any], input_dimension: int, output_dimension: int):
        super().__init__()
        self.cfg = cfg
        if cfg["num_hidden_layers"] % 2 == 1 and cfg["use_residual_connection"]:
            raise Exception("When residual connection will be used in hidden layers, number of hidden layers should be even.")

        self.layers = nn.ModuleList()

        # Append hidden layers
        num_hidden_layers_appended = 0
        while num_hidden_layers_appended < cfg["num_hidden_layers"]:
            if num_hidden_layers_appended == 0:
                current_linear_input_dimension = input_dimension
            else:
                current_linear_input_dimension = cfg["hidden_dimension"]

            current_linear_output_dimension = cfg["hidden_dimension"]

            if cfg["use_residual_connection"] and current_linear_input_dimension == current_linear_output_dimension:
                current_linear_residual_connection = True
                num_hidden_layers_appended += 2
            else:
                current_linear_residual_connection = False
                num_hidden_layers_appended += 1

            self.layers.append(HiddenLayer(cfg=cfg, input_dimension=current_linear_input_dimension, output_dimension=current_linear_output_dimension, use_residual_connection=current_linear_residual_connection))

        # Append output layer
        self.layers.append(OutputLayer(cfg=cfg, output_dimension=output_dimension))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.clone()
        for layer in self.layers:
            y = layer(y)
        return y
