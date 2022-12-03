import os
import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from utils import get_experiment_dir


def save_predictions_and_ground_truths_split(cfg: Dict[str, Any], data_loaders: Dict[str, DataLoader], split_name: str, model: nn.Module, loss_function, dataset: Dataset, logger: logging.Logger) -> None:
    model.eval()

    all_sample_ids = []
    all_ground_truths = []
    all_predictions = []

    total_loss = 0.0
    total_count = 0.0

    with torch.no_grad():
        for batch in data_loaders[split_name]:
            sample_id_indices = batch["sample_id_indices"]
            X = batch["X"]
            y = batch["y"]

            if cfg["normalize_input"]:
                X = (X - dataset.X_train_mean) / dataset.X_train_std

            yhat = model(X)

            if cfg["normalize_output"]:
                # then during training y was manually normalized, and yhat is produced as normalized as well.
                # we should unnormalize yhat so that it is comparable to y above, which was not normalized manually during evaluation.
                yhat = yhat * dataset.y_train_std + dataset.y_train_mean

            total_count += float(y.shape[0] * y.shape[1])
            total_loss += float(loss_function(yhat, y))

            all_sample_ids.append(np.array([dataset.sample_ids[int(sample_id_index)] for sample_id_index in sample_id_indices.numpy()]))
            all_ground_truths.append(y.cpu().numpy())
            all_predictions.append(yhat.cpu().numpy())

    all_sample_ids = np.hstack(all_sample_ids)
    all_ground_truths = np.vstack(all_ground_truths)
    all_predictions = np.vstack(all_predictions)
    all_loss = total_loss / total_count

    logger.log(level=logging.INFO, msg=f"{split_name.capitalize()} {cfg['loss_function']} loss is {all_loss}.")

    logger.log(level=logging.INFO, msg=f"Saving {split_name} results...")

    experiment_dir = get_experiment_dir(cfg=cfg)

    all_ground_truths_df = pd.DataFrame(data=all_ground_truths, columns=dataset.entrezgene_ids, index=all_sample_ids).reset_index(drop=False).rename(columns={"index": "sample_id"})
    all_predictions_df = pd.DataFrame(data=all_predictions, columns=dataset.entrezgene_ids, index=all_sample_ids).reset_index(drop=False).rename(columns={"index": "sample_id"})

    all_ground_truths_df.to_csv(os.path.join(experiment_dir, f"{split_name}_ground_truths.tsv"), sep="\t", index=False)
    all_predictions_df.to_csv(os.path.join(experiment_dir, f"{split_name}_predictions.tsv"), sep="\t", index=False)


def save_predictions_and_ground_truths(cfg: Dict[str, Any], data_loaders: Dict[str, DataLoader], model: nn.Module, loss_function, dataset: Dataset, logger: logging.Logger) -> None:
    for split_name in ["val", "test"]:
        save_predictions_and_ground_truths_split(cfg=cfg, data_loaders=data_loaders, split_name=split_name, model=model, loss_function=loss_function, dataset=dataset, logger=logger)
