from typing import Any, Dict

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def train(cfg: Dict[str, Any], data_loaders: Dict[str, DataLoader], model: nn.Module, loss_function, dataset: Dataset, optimizer) -> None:
    model.train()
    for batch in data_loaders["train"]:
        X = batch["X"]
        y = batch["y"]

        if cfg["use_batch_normalization"] and X.shape[0] == 1:
            continue

        optimizer.zero_grad()

        if cfg["normalize_input"]:
            X = (X - dataset.X_train_mean) / (dataset.X_train_std + 1e-10)

        if cfg["normalize_output"]:
            y = (y - dataset.y_train_mean) / (dataset.y_train_std + 1e-10)

        yhat = model(X)
        loss = loss_function(yhat, y)

        loss.backward()
        optimizer.step()
