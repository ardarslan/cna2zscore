import logging
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def evaluate_split(cfg: Dict[str, Any], data_loaders: DataLoader, split_name: str, model: nn.Module, loss_function, dataset: Dataset, epoch: int, logger: logging.Logger) -> np.float32:
    model.eval()

    with torch.no_grad():
        sample_count = 0
        total_loss = 0

        for batch in data_loaders[split_name]:
            X = batch["X"]
            y = batch["y"]

            if not cfg["normalize_input"]:
                yhat = model(X)
            else:
                X_normalized = (X - dataset.X_train_mean) / (dataset.X_train_std + 1e-10)
                yhat = model(X_normalized)

            if not cfg["normalize_output"]:
                loss = loss_function(yhat, y)
            else:
                yhat_unnormalized = yhat * (dataset.y_train_std + 1e-10) + dataset.y_train_mean
                loss = loss_function(yhat_unnormalized, y)

            sample_count += X.shape[0]
            total_loss += loss * sample_count

        loss = np.round(np.float32(total_loss / sample_count), 2)

        logger.log(level=logging.INFO, msg=f"Epoch {epoch}, {split_name.capitalize()} {cfg['loss_function']} loss is {loss}.")

        return loss


def evaluate(cfg: Dict[str, Any], data_loaders: Dict[str, DataLoader], model: nn.Module, loss_function, dataset: Dataset, epoch: int, logger: logging.Logger) -> Dict[str, np.float32]:
    loss_dict = dict(
        (
            split_name,
            evaluate_split(cfg=cfg, data_loaders=data_loaders, model=model, loss_function=loss_function, dataset=dataset, split_name=split_name, epoch=epoch, logger=logger)
        )
        for split_name in ["train", "val", "test"]
    )
    return loss_dict
