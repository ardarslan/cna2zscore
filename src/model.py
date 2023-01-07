import gc
import math
from typing import Any, Dict, List

import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from layer import NonLinearLayer, OutputLayer, SelfAttention


def get_single_model(cfg: Dict[str, Any], input_dimension: int, output_dimension: int):
    if cfg["model"] == "baseline":
        ModelClass = BaselineModel
    elif cfg["model"] in ["linear", "mlp"]:
        ModelClass = MLP
    elif cfg["model"] == "rescon_mlp":
        ModelClass = ResConMLP
    elif cfg["model"] == "transformer":
        ModelClass = Transformer
    else:
        raise NotImplementedError(f"{cfg['model']} is not an implemented model.")
    return ModelClass(cfg=cfg, input_dimension=input_dimension, output_dimension=output_dimension)


class BaselineModel(nn.Module):
    def __init__(self, cfg: Dict[str, Any], input_dimension: int, output_dimension: int):
        super().__init__()
        self.cfg = cfg
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        for _ in range(self.output_dimension):
            current_weight = Parameter(torch.empty((1 + self.input_dimension - self.output_dimension, 1)))
            nn.init.kaiming_uniform_(current_weight, a=math.sqrt(5))

            current_bias = Parameter(torch.empty((1, )))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(current_weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(current_bias, -bound, bound)

            self.weights.append(current_weight)
            self.biases.append(current_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        nongene_inputs = x[:, -(self.input_dimension - self.output_dimension):]
        y = torch.zeros(size=(x.shape[0], self.output_dimension), device=self.cfg["device"])
        for j in range(self.output_dimension):
            y[:, j] = F.linear(torch.concat((x[:, j:j+1], nongene_inputs), dim=1), self.weights[j], self.biases[j])
        return y


class MLP(nn.Module):
    def __init__(self, cfg: Dict[str, Any], input_dimension: int, output_dimension: int):
        super().__init__()
        self.cfg = cfg
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension

        self._set_model_hidden_dimension()

        if self.cfg["num_nonlinear_layers"] != 0 and ((self.cfg["l1_reg_diagonal_coeff"] != self.cfg["l1_reg_nondiagonal_coeff"]) or (self.cfg["l2_reg_diagonal_coeff"] != self.cfg["l2_reg_nondiagonal_coeff"])):
            raise Exception("Custom regularization is supported only when num_nonlinear_layers is not 0.")

        self.layers = nn.ModuleList()

        for i in range(self.cfg["num_nonlinear_layers"]):
            if i == 0:
                self.layers.append(NonLinearLayer(cfg=cfg, input_dimension=input_dimension, output_dimension=self.hidden_dimension))
            else:
                self.layers.append(NonLinearLayer(cfg=cfg, input_dimension=self.hidden_dimension, output_dimension=self.hidden_dimension))

        if self.cfg["num_nonlinear_layers"] == 0:
            self.layers.append(OutputLayer(cfg=cfg, input_dimension=input_dimension, output_dimension=output_dimension))
        else:
            self.layers.append(OutputLayer(cfg=cfg, input_dimension=self.hidden_dimension, output_dimension=output_dimension))

    def _set_model_hidden_dimension(self) -> None:
        try:
            self.hidden_dimension = int(np.max([self.input_dimension, self.output_dimension]) * float(self.cfg["hidden_dimension_ratio"]))
        except ValueError:
            raise Exception(f"{self.cfg['hidden_dimension_ratio']} is not a valid hidden_dimension_ratio.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = None
        for layer in self.layers:
            if y is None:
                y = layer(x)
            else:
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

        self._set_model_hidden_dimension()

        self.layers = nn.ModuleList()

        for i in range(self.cfg["num_nonlinear_layers"]):
            if i == 0:
                self.layers.append(NonLinearLayer(cfg=cfg, input_dimension=input_dimension, output_dimension=self.hidden_dimension))
            else:
                self.layers.append(NonLinearLayer(cfg=cfg, input_dimension=self.hidden_dimension, output_dimension=self.hidden_dimension))

        self.layers.append(OutputLayer(cfg=cfg, input_dimension=self.hidden_dimension, output_dimension=output_dimension))

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

    def _set_model_hidden_dimension(self) -> None:
        try:
            self.hidden_dimension = int(np.max([self.input_dimension, self.output_dimension]) * float(self.cfg["hidden_dimension_ratio"]))
        except ValueError:
            raise Exception(f"{self.cfg['hidden_dimension_ratio']} is not a valid hidden_dimension_ratio.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = None
        for layer in self.layers:
            if y is None:
                y = layer(x)
            else:
                y = layer(y)
        return y + F.linear(x[:, :self.output_dimension], self.ResConW, self.ResConB)


class Transformer(nn.Module):
    """
    A Transformer consisting of a self attention and a fully connected layer.
    """
    def __init__(self, cfg: Dict[str, Any], input_dimension: int, output_dimension: int):
        super().__init__()

        self.cfg = cfg
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension

        self.gene_embedding = torch.nn.Embedding(num_embeddings=self.output_dimension, embedding_dim=self.cfg["gene_embedding_size"])
        self.attention = SelfAttention(d=self.cfg["gene_embedding_size"]+1, n_heads=self.cfg["num_attention_heads"])
        self.normalization = nn.LayerNorm(self.cfg["gene_embedding_size"]+1)

        fcn_input_dim = self.cfg["gene_embedding_size"] + 1
        if "purity" in self.cfg["dataset"]:
            fcn_input_dim += 1
        if self.cfg["cancer_type"] == "all":
            fcn_input_dim += 29

        self.fc = nn.Linear(fcn_input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (N, num_genes+(1, optional, purity column)+(29, optional, cancer type one hot columns)).

        Returns:
            Output tensor of shape (N, num_genes).
        """
        attention_inputs = torch.concat((self.gene_embedding.weight.unsqueeze(0).repeat((x.shape[0], 1, 1)), x[:, :self.output_dimension].unsqueeze(-1)), dim=-1) # (N, num_genes, d+1)
        out = self.attention(attention_inputs) + attention_inputs # (N, num_genes, d+1)
        del attention_inputs
        gc.collect()
        torch.cuda.empty_cache()
        out = self.normalization(out) # (N, num_genes, d+1)
        out = torch.concat((out, x[:, self.output_dimension:].unsqueeze(1).repeat(1, self.output_dimension, 1)), dim=-1) # (N, num_genes, d+1) or (N, num_genes, d+2) or (N, num_genes, d+30) or (N, num_genes, d+31)
        out = self.fc(out) # (N, num_genes, 1)
        out = out[:, :, 0] # (N, num_genes)
        return out


class PerChromosomeModel(nn.Module):
    def __init__(self, cfg: Dict[str, Any], chromosome_name_X_column_ids_mapping: Dict[str, List[int]]):
        super().__init__()
        self.cfg = cfg
        self.input_dimension = cfg["input_dimension"]
        self.output_dimension = cfg["output_dimension"]
        self.X_column_ids = []
        self.y_column_ids = []
        self.models = nn.ModuleList()
        nonchromosome_X_column_ids = chromosome_name_X_column_ids_mapping["nonchromosome"]

        for chromosome_name, current_X_column_ids in chromosome_name_X_column_ids_mapping.items():
            if chromosome_name == "nonchromosome":
                continue
            self.X_column_ids.append(current_X_column_ids + nonchromosome_X_column_ids)
            self.y_column_ids.append(current_X_column_ids)
            current_model = get_single_model(cfg=cfg, input_dimension=len(current_X_column_ids)+len(nonchromosome_X_column_ids), output_dimension=len(current_X_column_ids))
            self.models.append(current_model)

    def forward(self, x: torch.Tensor):
        y = torch.zeros(size=(x.shape[0], self.output_dimension), device=self.cfg["device"])
        for current_X_column_ids, current_y_column_ids, current_model in zip(self.X_column_ids, self.y_column_ids, self.models):
            y[:, current_y_column_ids] = current_model(x[:, current_X_column_ids])
        return y
