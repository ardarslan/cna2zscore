import logging
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def validate(cfg: Dict[str, Any], data_loaders: Dict[str, DataLoader], model: nn.Module, loss_function, epoch: int, logger: logging.Logger, summary_writer: SummaryWriter, val_main_loss_values: List[float]) -> np.float32:
    model.eval()

    with torch.no_grad():
        main_loss_count = 0.0
        main_loss_sum = 0.0

        for batch in data_loaders["val"]:
            X = batch["X"]
            y = batch["y"]

            yhat = model(X)

            main_loss_count += float(y.shape[0] * y.shape[1])
            main_loss_sum += float(loss_function(yhat, y))

        logger.log(level=logging.INFO, msg=f"Epoch {str(epoch).zfill(3)}, {(5 - len('val')) * ' ' + 'val'.capitalize()} {cfg['loss_function']} loss is {np.round(main_loss_sum / main_loss_count, 2)}.")

        val_loss_dict = {
            cfg["loss_function"]: main_loss_sum / main_loss_count
        }

        val_main_loss_values.append(val_loss_dict[cfg["loss_function"]])

        for loss_name, loss_value in val_loss_dict.items():
            summary_writer.add_scalar(f"val_{loss_name}", loss_value, epoch)

        return val_loss_dict
