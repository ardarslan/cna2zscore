import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def test(cfg: Dict[str, Any], data_loaders: List[DataLoader], model: nn.Module, loss_function, dataset: Dataset, logger: logging.Logger) -> Tuple[np.ndarray, np.ndarray, np.float32, np.float32, np.float32, List[Tuple[int, np.float32]], List[Tuple[int, np.float32]]]:
    model.eval()

    all_sample_ids = []
    all_ys = []
    all_yhats = []

    total_loss = 0.0
    total_sample_count = 0.0

    with torch.no_grad():
        for batch in data_loaders["test"]:
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

            total_loss += float(loss_function(yhat, y))
            total_sample_count += X.shape[0]

            all_sample_ids.append(np.array([dataset.sample_ids[int(sample_id_index)] for sample_id_index in sample_id_indices.numpy()]))
            all_ys.append(y.cpu().numpy())
            all_yhats.append(yhat.cpu().numpy())

    all_sample_ids = np.hstack(all_sample_ids)
    all_ys = np.vstack(all_ys)
    all_yhats = np.vstack(all_yhats)
    all_loss = total_loss / total_sample_count

    logger.log(level=logging.INFO, msg=f"Test {cfg['loss_function']} loss is {all_loss}.")

    test_results_dict = {
        "all_sample_ids": all_sample_ids,
        "all_ys": all_ys,
        "all_yhats": all_yhats,
    }

    return test_results_dict
