import logging
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter


def validate(cfg: Dict[str, Any], data_loaders: Dict[str, DataLoader], model: nn.Module, loss_function, dataset: Dataset, epoch: int, logger: logging.Logger, summary_writer: SummaryWriter, main_loss_values: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
    model.eval()

    current_loss_dicts = {"train": {},
                          "val": {}}

    with torch.no_grad():
        main_loss_sum = 0.0
        main_loss_count = 0.0

        for split_name in ["train", "val"]:
            for batch in data_loaders[split_name]:
                X = batch["X"]
                y = batch["y"]

                if cfg["normalize_input"]:
                    X = (X - dataset.X_train_mean) / dataset.X_train_std

                yhat = model(X)

                if cfg["normalize_output"]:
                    # then during training y was manually normalized, and yhat comes as normalized as well.
                    # we should unnormalize yhat so that it is comparable to y above, which was not normalized manually during evaluation.
                    yhat = yhat * dataset.y_train_std + dataset.y_train_mean

                main_loss_count += float(y.shape[0] * y.shape[1])
                main_loss_sum += float(loss_function(yhat, y))

            logger.log(level=logging.INFO, msg=f"Epoch {str(epoch).zfill(3)}, {(5 - len(split_name)) * ' ' + split_name.capitalize()} {cfg['loss_function']} loss is {np.round(main_loss_sum / main_loss_count, 2)}.")

            main_loss_values[split_name].append(main_loss_sum / main_loss_count)

            current_loss_dicts[split_name][cfg["loss_function"]] = main_loss_sum / main_loss_count

            for loss_name, loss_value in current_loss_dicts[split_name].items():
                summary_writer.add_scalar(f"{split_name}_{loss_name}", loss_value, epoch)

        return current_loss_dicts
