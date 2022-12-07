from typing import Any, Dict

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def train(cfg: Dict[str, Any], data_loaders: Dict[str, DataLoader], model: nn.Module, loss_function, dataset: Dataset, optimizer) -> None:
    model.train()
    optimizer.zero_grad()

    effective_batch_size = cfg["effective_batch_size"]
    train_data_loader = data_loaders["train"]
    num_train_samples = len(dataset.train_idx)
    num_batches = len(train_data_loader)
    observed_sample_count = 0

    for batch_idx, batch in enumerate(train_data_loader):
        X = batch["X"]
        y = batch["y"]
        observed_sample_count += X.shape[0]

        if cfg["normalization_type"] == "batch_normalization" and X.shape[0] == 1:
            continue

        if cfg["normalize_input"]:
            X = (X - dataset.X_train_mean) / dataset.X_train_std

        if cfg["normalize_output"]:
            y = (y - dataset.y_train_mean) / dataset.y_train_std

        yhat = model(X)
        loss = loss_function(yhat, y)

        if cfg["l1_reg_coeff"] > 0:
            loss += cfg["l1_reg_coeff"] * sum(p.abs().sum() for p in model.parameters())
        if cfg["l2_reg_coeff"] > 0:
            loss += cfg["l2_reg_coeff"] * sum(p.pow(2.0).sum() for p in model.parameters())

        if cfg["use_gradient_accumulation"]:
            if ((batch_idx + 1) < num_batches) or (num_train_samples % effective_batch_size == 0):
                (loss / float(effective_batch_size)).backward()
            else:
                (loss / float(num_train_samples % effective_batch_size)).backward()

            if (observed_sample_count % effective_batch_size == 0) or observed_sample_count == num_train_samples:
                optimizer.step()
                optimizer.zero_grad()
        else:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
