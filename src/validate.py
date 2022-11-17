import logging
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def validate_split(cfg: Dict[str, Any], data_loaders: List[DataLoader], split_name: str, model: nn.Module, loss_function, dataset: Dataset, epoch: int, logger: logging.Logger) -> np.float32:
    model.eval()

    with torch.no_grad():
        total_count = 0
        total_loss = 0

        for batch in data_loaders[split_name]:
            X = batch["X"]
            y = batch["y"]

            if cfg["normalize_input"]:
                X = (X - dataset.X_train_mean) / (dataset.X_train_std + 1e-10)

            yhat = model(X)

            if cfg["normalize_output"]:
                # then during training y was manually normalized, and yhat comes as normalized as well.
                # we should unnormalize yhat so that it is comparable to y above, which was not normalized manually during evaluation.
                yhat = yhat * (dataset.y_train_std + 1e-10) + dataset.y_train_mean

            loss = loss_function(yhat, y)

            total_count += y.shape[0] * y.shape[1]
            total_loss += loss

        loss = np.round(np.float32(total_loss / total_count), 2)

        logger.log(level=logging.INFO, msg=f"Epoch {epoch}, {split_name.capitalize()} {cfg['loss_function']} loss is {loss}.")

        return loss


def validate(cfg: Dict[str, Any], data_loaders: Dict[str, DataLoader], model: nn.Module, loss_function, dataset: Dataset, epoch: int, logger: logging.Logger) -> Dict[str, np.float32]:
    loss_dict = dict(
        (
            split_name,
            validate_split(cfg=cfg, data_loaders=data_loaders, model=model, loss_function=loss_function, dataset=dataset, split_name=split_name, epoch=epoch, logger=logger)
        )
        for split_name in ["train", "val"]
    )
    return loss_dict
