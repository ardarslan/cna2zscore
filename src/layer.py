import gc
from typing import Dict, Any

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


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

        # if cfg["normalization_type"] == "batch_normalization":
        if cfg["model"] == "dl_transformer":
            self.normalization = nn.LayerNorm(normalized_shape=output_dimension)
        else:
            self.normalization = nn.BatchNorm1d(num_features=output_dimension)
        # elif cfg["normalization_type"] == "instance_normalization":
        #         self.normalization = nn.InstanceNorm1d(num_features=output_dimension)
        # else:
        #     raise Exception(f"{cfg['normalization_type']} is not an implemented normalization type.")

        self.dropout = nn.Dropout(cfg["dropout"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.linear(x)

        # if self.cfg["normalization_type"] == "instance_normalization":
        #     y = y.view(*y.shape, 1).permute((0, 2, 1))
        #     y = self.normalization(y)
        #     y = y[:, 0, :]
        # elif self.cfg["normalization_type"] == "batch_normalization":
        y = self.normalization(y)
        # else:
        #     raise Exception(f"{self.cfg['normalization_type']} is not an implemented normalization type.")
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


class SelfAttention(nn.Module):
    """
    A SelfAttention model.

    Args:
        d: The embedding dimension.
        heads: The number of attention heads.
    """
    def __init__(self, d: int, n_heads: int=8):
        super().__init__()
        self.n_heads = n_heads

        self.Wq = nn.Linear(d, d * n_heads, bias=False)
        self.Wk = nn.Linear(d, d * n_heads, bias=False)
        self.Wv = nn.Linear(d, d * n_heads, bias=False)

        # This unifies the outputs of the different heads into
        # a single k-dimensional vector.
        if self.n_heads != 1:
            self.unifyheads = nn.Linear(n_heads * d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: The input embedding of shape [b, l, d].

        Returns:
            Self attention tensor of shape [b, l, d].
        """
        b, l, d = x.size()
        h = self.n_heads

        # Transform the input embeddings x of shape [b, l, d] to queries, keys, values.
        # The output shape is [b, l, d*h] which we transform into [b, l, h, d]. Then,
        # we fold the heads into the batch dimenstion to arrive at [b*h, l, d]
        queries = self.Wq(x).view(b, l, h, d).transpose(1, 2).contiguous().view(b*h, l, d)
        keys = self.Wk(x).view(b, l, h, d).transpose(1, 2).contiguous().view(b*h, l, d)
        values = self.Wv(x).view(b, l, h, d).transpose(1, 2).contiguous().view(b*h, l, d)

        # Compute the product of queries and keys and scale with sqrt(d).
        # The tensor w' has shape (b*h, l, l) containing raw weights.
        #----------------
        w_prime = torch.bmm(queries, keys.transpose(1, 2)) / np.sqrt(d)
        #----------------
        del queries
        del keys
        gc.collect()
        torch.cuda.empty_cache()

        # Compute w by normalizing w' over the last dimension.
        # Shape: [b*h, l, l]
        #----------------
        w = F.softmax(w_prime, dim=-1)
        del w_prime
        gc.collect()
        torch.cuda.empty_cache()
        #----------------

        # Apply the self attention to the values.
        # Shape: [b, h, l, d]
        #----------------
        out = torch.bmm(w, values).view(b, h, l, d)
        del w
        del values
        gc.collect()
        torch.cuda.empty_cache()
        #----------------

        # Swap h, l back.
        # Shape: [b, l, h*d]
        out = out.transpose(1, 2).contiguous().view(b, l, h * d)

        # Unify heads to arrive at shape [b, l, d].
        if self.n_heads == 1:
            return out
        else:
            return self.unifyheads(out)
