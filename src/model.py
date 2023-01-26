import gc
import math
from typing import Any, Dict, List

import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from joblib import delayed, Parallel
from torch.nn.parameter import Parameter
from sklearn.linear_model import Lasso, MultiTaskLasso

from layer import NonLinearLayer, OutputLayer, SelfAttention


def get_single_model(cfg: Dict[str, Any], input_dimension: int, output_dimension: int):
    if cfg["model"] == "dl_per_gene":
        ModelClass = DLPerGene
    elif cfg["model"] == "dl_gene_embeddings":
        ModelClass = DLGeneEmbeddings
    elif cfg["model"] in ["dl_linear", "dl_mlp"]:
        ModelClass = DLMLP
    elif cfg["model"] == "dl_linear_zero_diagonal":
        ModelClass = DLLinearZeroDiagonal
    elif cfg["model"] == "dl_interpretable_mlp":
        ModelClass = DLInterpretableMLP
    elif cfg["model"] == "dl_rescon_mlp":
        ModelClass = DLResConMLP
    elif cfg["model"] == "dl_transformer":
        ModelClass = DLTransformer
    elif cfg["model"] == "sklearn_linear":
        ModelClass = SklearnLinear
    elif cfg["model"] == "sklearn_per_gene":
        ModelClass = SklearnPerGene
    else:
        raise NotImplementedError(f"{cfg['model']} is not an implemented model.")
    return ModelClass(cfg=cfg, input_dimension=input_dimension, output_dimension=output_dimension)


class DLPerGene(nn.Module):
    def __init__(self, cfg: Dict[str, Any], input_dimension: int, output_dimension: int):
        super().__init__()
        self.cfg = cfg
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        for _ in range(self.output_dimension):
            current_weight = Parameter(torch.empty((1, 1 + self.input_dimension - self.output_dimension)))
            nn.init.kaiming_uniform_(current_weight, a=math.sqrt(5))

            current_bias = Parameter(torch.empty((1, 1)))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(current_weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(current_bias, -bound, bound)

            self.weights.append(current_weight)
            self.biases.append(current_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        nongene_inputs = x[:, -(self.input_dimension - self.output_dimension):]
        y = torch.zeros(size=(x.shape[0], self.output_dimension), device=self.cfg["device"])
        for j in range(self.output_dimension):
            y[:, j] = F.linear(torch.concat((x[:, j:j+1], nongene_inputs), dim=1), self.weights[j], self.biases[j]).squeeze()
        return y


class DLGeneEmbeddings(nn.Module):
    def __init__(self, cfg: Dict[str, Any], input_dimension: int, output_dimension: int):
        super().__init__()
        self.cfg = cfg
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.gene_embeddings = nn.Embedding(num_embeddings=cfg["num_genes"], embedding_dim=cfg["gene_embedding_size"])
        self.fc = nn.Linear((self.input_dimension - self.output_dimension) + cfg["gene_embedding_size"] + 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        nongene_inputs = x[:, -(self.input_dimension - self.output_dimension):]
        y = torch.zeros(size=(x.shape[0], self.output_dimension), device=self.cfg["device"])
        for j in range(self.output_dimension):
            y[:, j] = self.fc(torch.concat((nongene_inputs, x[:, j:j+1], self.gene_embeddings.weight[j, :].unsqueeze(0).repeat(x.shape[0], 1)), axis=1)).squeeze()
        return y


class DLLinearZeroDiagonal(nn.Module):
    def __init__(self, cfg: Dict[str, Any], input_dimension: int, output_dimension: int):
        super().__init__()
        self.cfg = cfg
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        len_upper_triangle_weights = torch.triu_indices(row=output_dimension, col=input_dimension, offset=1).shape[1]
        len_lower_triangle_weights = torch.tril_indices(row=output_dimension, col=input_dimension, offset=-1).shape[1]
        self.len_diagonal_weights = np.minimum(output_dimension, input_dimension)
        self.upper_triangle_weights = Parameter(torch.empty(size=(1, len_upper_triangle_weights)))
        nn.init.kaiming_uniform_(self.upper_triangle_weights, a=math.sqrt(5))
        self.lower_triangle_weights = Parameter(torch.empty(size=(1, len_lower_triangle_weights)))
        nn.init.kaiming_uniform_(self.lower_triangle_weights, a=math.sqrt(5))
        diagonal_weights = torch.empty(size=(1, self.len_diagonal_weights))
        nn.init.kaiming_uniform_(diagonal_weights, a=math.sqrt(5))
        weight = self.reconstruct_weight(diagonal_weights=diagonal_weights)
        self.bias = Parameter(torch.empty(output_dimension))
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def reconstruct_weight(self, diagonal_weights: torch.Tensor):
        weight = torch.zeros(size=(self.output_dimension, self.input_dimension))
        upper_triangle_counter = 0
        lower_triangle_counter = 0
        for i in range(self.output_dimension):
            for j in range(self.input_dimension):
                if i < j:
                    weight[i, j] = self.upper_triangle_weights[0][upper_triangle_counter]
                    upper_triangle_counter += 1
                elif i == j:
                    weight[i, j] = diagonal_weights[0][i]
                else:
                    weight[i, j] = self.lower_triangle_weights[0][lower_triangle_counter]
                    lower_triangle_counter += 1
        return weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.reconstruct_weight(diagonal_weights=torch.zeros(size=(1, self.len_diagonal_weights)))
        return F.linear(input=x, weight=weight, bias=self.bias)


class DLMLP(nn.Module):
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
            self.hidden_dimension = np.max([int(np.max([self.input_dimension, self.output_dimension]) * float(self.cfg["hidden_dimension_ratio"])), 1])
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


class DLResConMLP(nn.Module):
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


class DLTransformer(nn.Module):
    """
    A Transformer consisting of a self attention and a fully connected layer.
    """
    def __init__(self, cfg: Dict[str, Any], input_dimension: int, output_dimension: int):
        super().__init__()

        self.cfg = cfg
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension

        self.gene_embeddings = torch.nn.Embedding(num_embeddings=self.output_dimension, embedding_dim=self.cfg["gene_embedding_size"])
        self.attention = SelfAttention(d=self.cfg["gene_embedding_size"]+1, n_heads=self.cfg["num_attention_heads"])
        self.normalization = nn.LayerNorm(self.cfg["gene_embedding_size"]+1)

        mlp_input_dim = self.cfg["gene_embedding_size"] + 1
        if "purity" in self.cfg["dataset"]:
            mlp_input_dim += 1
        if self.cfg["cancer_type"] == "all":
            mlp_input_dim += 29

        self.mlp = DLMLP(cfg=cfg, input_dimension=mlp_input_dim, output_dimension=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (N, num_genes+(1, optional, purity column)+(29, optional, cancer type one hot columns)).

        Returns:
            Output tensor of shape (N, num_genes).
        """
        attention_inputs = torch.concat((self.gene_embeddings.weight.unsqueeze(0).repeat((x.shape[0], 1, 1)), x[:, :self.output_dimension].unsqueeze(-1)), dim=-1) # (N, num_genes, d+1)
        out = self.attention(attention_inputs) + attention_inputs # (N, num_genes, d+1)
        del attention_inputs
        gc.collect()
        torch.cuda.empty_cache()
        out = self.normalization(out) # (N, num_genes, d+1)
        out = torch.concat((out, x[:, self.output_dimension:].unsqueeze(1).repeat(1, self.output_dimension, 1)), dim=-1) # (N, num_genes, d+1) or (N, num_genes, d+2) or (N, num_genes, d+30) or (N, num_genes, d+31)
        out = self.mlp(out) # (N, num_genes, 1)
        out = out[:, :, 0] # (N, num_genes)
        return out


class DLPerChromosome(nn.Module):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.input_dimension = cfg["input_dimension"]
        self.output_dimension = cfg["output_dimension"]
        self.X_column_ids = []
        self.y_column_ids = []
        self.models = nn.ModuleList()
        nonchromosome_X_column_ids = self.cfg["chromosome_name_X_column_ids_mapping"]["nonchromosome"]

        for chromosome_name, current_X_column_ids in self.cfg["chromosome_name_X_column_ids_mapping"].items():
            if chromosome_name == "nonchromosome":
                continue
            self.X_column_ids.append(current_X_column_ids + nonchromosome_X_column_ids)
            self.y_column_ids.append(current_X_column_ids)
            current_model = get_single_model(cfg=cfg, input_dimension=len(current_X_column_ids)+len(nonchromosome_X_column_ids), output_dimension=len(current_X_column_ids))
            self.models.append(current_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.zeros(size=(x.shape[0], self.output_dimension), device=self.cfg["device"])
        for current_X_column_ids, current_y_column_ids, current_model in zip(self.X_column_ids, self.y_column_ids, self.models):
            y[:, current_y_column_ids] = current_model(x[:, current_X_column_ids])
        return y


class DLInterpretableMLP(object):
    def __init__(self, cfg: Dict[str, Any], input_dimension: int, output_dimension: int):
        super().__init__()
        self.cfg = cfg
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.mlp = DLMLP(cfg=cfg, input_dimension=self.input_dimension, output_dimension=self.output_dimension * self.input_dimension + self.output_dimension)

    def get_weights_and_biases(self, x: torch.Tensor) -> torch.Tensor:
        weights_and_biases = self.mlp(x)
        weights = weights_and_biases[:, :-self.output_dimension].view(x.shape[0], self.output_dimension, self.input_dimension)  # (N, output_dim, input_dim)
        biases = weights_and_biases[:, -self.output_dimension:]  # (N, output_dim)
        return weights, biases

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights, biases = self.get_weights_and_biases(x)
        x = torch.bmm(weights, x.unsqueeze(0)) # (N, output_dim, input_dim) * (N, input_dim, 1) = (N, output_dim, 1)
        return x[:, :, 0] + biases, weights # (N, output_dim)


class SklearnPerChromosome(object):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.input_dimension = cfg["input_dimension"]
        self.output_dimension = cfg["output_dimension"]
        self.X_column_ids = []
        self.y_column_ids = []
        self.models = []
        nonchromosome_X_column_ids = self.cfg["chromosome_name_X_column_ids_mapping"]["nonchromosome"]

        for chromosome_name, current_X_column_ids in self.cfg["chromosome_name_X_column_ids_mapping"].items():
            if chromosome_name == "nonchromosome":
                continue
            self.X_column_ids.append(current_X_column_ids + nonchromosome_X_column_ids)
            self.y_column_ids.append(current_X_column_ids)
            current_model = get_single_model(cfg=cfg, input_dimension=len(current_X_column_ids)+len(nonchromosome_X_column_ids), output_dimension=len(current_X_column_ids))
            self.models.append(current_model)

    def fit_helper(self, X: np.ndarray, y: np.ndarray, current_X_column_ids: List[int], current_y_column_ids: List[int], current_model: Any) -> None:
        current_model.fit(X[:, current_X_column_ids], y[:, current_y_column_ids])
        return current_model

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.models = Parallel(n_jobs=24)([delayed(self.fit_helper)(X, y, current_X_column_ids, current_y_column_ids, current_model) for current_X_column_ids, current_y_column_ids, current_model in zip(self.X_column_ids, self.y_column_ids, self.models)])

    def predict_helper(self, X: np.ndarray, current_X_column_ids: List[int], current_model: Any) -> None:
        return current_model.predict(X[:, current_X_column_ids])

    def predict(self, X: np.ndarray) -> np.ndarray:
        yhat = np.zeros(shape=(X.shape[0], self.output_dimension))

        results = Parallel(n_jobs=24)([delayed(self.predict_helper)(X, current_X_column_ids, current_model) for current_X_column_ids, current_model in zip(self.X_column_ids, self.models)])

        for current_y_column_ids, current_yhat in zip(self.y_column_ids, results):
            yhat[:, current_y_column_ids] = current_yhat

        return yhat


class SklearnPerGene(object):
    def __init__(self, cfg: Dict[str, Any], input_dimension: int, output_dimension: int):
        self.cfg = cfg
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.models = [Lasso(alpha=cfg["l1_reg_nondiagonal_coeff"]) for _ in range(self.output_dimension)]

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.number_of_samples = y.shape[0]
        for j in range(y.shape[1]):
            self.models[j].fit(X[:, [j] + [i for i in range(self.output_dimension, self.input_dimension)]], y[:, j].ravel())

    def predict(self, X: np.ndarray) -> np.ndarray:
        yhat = []
        for j in range(self.output_dimension):
            yhat.append(self.models[j].predict(X[:, [j] + [i for i in range(self.output_dimension, self.input_dimension)]]).reshape(X.shape[0], 1))
        yhat = np.hstack(yhat)
        return yhat


class SklearnLinear(object):
    def __init__(self, cfg: Dict[str, Any], input_dimension: int, output_dimension: int):
        self.cfg = cfg
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.model = MultiTaskLasso(alpha=cfg["l1_reg_nondiagonal_coeff"], max_iter=5000)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
