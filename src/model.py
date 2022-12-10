import torch
import torch.nn as nn

from typing import Dict, Any


from layer import NonLinearLayer, OutputLayer

class MLP(nn.Module):
    def __init__(self, cfg: Dict[str, Any], input_dimension: int, output_dimension: int):
        super().__init__()
        self.cfg = cfg

        self.layers = nn.ModuleList()

        for i in range(self.cfg["num_nonlinear_layers"]):
            if i == 0:
                self.layers.append(NonLinearLayer(cfg=cfg, input_dimension=input_dimension, output_dimension=cfg["hidden_dimension"]))
            else:
                self.layers.append(NonLinearLayer(cfg=cfg, input_dimension=cfg["hidden_dimension"], output_dimension=output_dimension))

        if self.cfg["num_nonlinear_layers"] == 0:
            self.layers.append(OutputLayer(cfg=cfg, input_dimension=input_dimension, output_dimension=output_dimension))
        else:
            self.layers.append(OutputLayer(cfg=cfg, input_dimension=cfg["hidden_dimension"], output_dimension=output_dimension))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.clone()
        for layer in self.layers:
            y = layer(y)
        return y


# from layer import InputLayer, HiddenLayer, OutputLayer
# class MLP(nn.Module):
#     def __init__(self, cfg: Dict[str, Any], input_dimension: int, output_dimension: int):
#         super().__init__()
#         self.cfg = cfg
#         if cfg["num_hidden_layers"] % 2 == 1 and cfg["use_residual_connection"]:
#             raise Exception("When residual connection will be used in hidden layers, number of hidden layers should be even.")

#         self.layers = nn.ModuleList()

#         self.layers.append(InputLayer(cfg=cfg, input_dimension=input_dimension))

#         # Append hidden layers
#         num_hidden_layers_appended = 0
#         while num_hidden_layers_appended < cfg["num_hidden_layers"]:
#             if cfg["use_residual_connection"]:
#                 num_hidden_layers_appended += 2
#             else:
#                 num_hidden_layers_appended += 1
#             self.layers.append(HiddenLayer(cfg=cfg, input_dimension=cfg["hidden_dimension"], output_dimension=cfg["hidden_dimension"]))

#         # Append output layer
#         self.layers.append(OutputLayer(cfg=cfg, output_dimension=output_dimension))

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         y = x.clone()
#         for layer in self.layers:
#             y = layer(y)
#         return y

# class LinearModel(nn.Module):
#     def __init__(self, cfg: Dict[str, Any], input_dimension: int, output_dimension: int):
#         super().__init__()
#         self.cfg = cfg
#         self.linear = nn.Linear(in_features=input_dimension, out_features=output_dimension)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.linear(x)
