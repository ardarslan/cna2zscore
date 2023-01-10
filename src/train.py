import logging
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter


def calculate_current_regularization_loss(cfg: Dict[str, Any], model: nn.Module, regularization_loss_type: str):
    if regularization_loss_type == "l1":
        regularization_diagonal_coeff = cfg["l1_reg_diagonal_coeff"]
        regularization_nondiagonal_coeff = cfg["l1_reg_nondiagonal_coeff"]
        regularization_operation = lambda x: x.abs()
    elif regularization_loss_type == "l2":
        regularization_diagonal_coeff = cfg["l2_reg_diagonal_coeff"]
        regularization_nondiagonal_coeff = cfg["l2_reg_nondiagonal_coeff"]
        regularization_operation = lambda x: x.pow(2.0)
    else:
        raise Exception(f"{regularization_loss_type} is not a valid regularization loss type.")

    current_regularization_loss = 0.0
    if regularization_diagonal_coeff != regularization_nondiagonal_coeff:
        if regularization_diagonal_coeff != 0.0:
            for parameter in model.parameters():
                if len(parameter.shape) == 2:  # weight matrix
                    current_regularization_loss += (regularization_diagonal_coeff - regularization_nondiagonal_coeff) * regularization_operation(torch.diag(parameter)).sum()

        if regularization_nondiagonal_coeff != 0.0:
            for parameter in model.parameters():
                if len(parameter.shape) == 2:  # weight matrix
                    current_regularization_loss += regularization_nondiagonal_coeff * regularization_operation(parameter).sum()
    else:
        if regularization_nondiagonal_coeff != 0.0:
            for parameter in model.parameters():
                if len(parameter.shape) == 2:  # weight matrix
                    current_regularization_loss += regularization_nondiagonal_coeff * regularization_operation(parameter).sum()

    return current_regularization_loss


def train(cfg: Dict[str, Any], data_loaders: Dict[str, DataLoader], model: nn.Module, loss_function, dataset: Dataset, optimizer, epoch: int, logger: logging.Logger, summary_writer: SummaryWriter, train_main_loss_values: List[float]) -> None:
    model.train()
    optimizer.zero_grad()

    effective_batch_size = cfg["effective_batch_size"]
    train_data_loader = data_loaders["train"]
    num_train_samples = len(dataset.train_indices)
    num_batches = len(train_data_loader)
    observed_sample_count = 0.0

    main_loss_sum = 0.0
    main_loss_count = 0.0

    l1_loss_sum = 0.0
    l1_loss_count = 0.0

    l2_loss_sum = 0.0
    l2_loss_count = 0.0

    total_loss_sum = 0.0
    total_loss_count = 0.0

    for batch_idx, batch in enumerate(train_data_loader):
        X = batch["X"]
        y = batch["y"]
        current_batch_size = X.shape[0]

        if cfg["normalization_type"] == "batch_normalization" and current_batch_size == 1:
            continue

        observed_sample_count += current_batch_size

        yhat = model(X)
        current_main_loss = loss_function(yhat, y)
        main_loss_sum += float(current_main_loss) * current_batch_size
        main_loss_count += current_batch_size

        current_l1_loss = calculate_current_regularization_loss(cfg=cfg, model=model, regularization_loss_type="l1")
        l1_loss_sum += current_l1_loss
        l1_loss_count += 1

        current_l2_loss = calculate_current_regularization_loss(cfg=cfg, model=model, regularization_loss_type="l2")
        l2_loss_sum += current_l2_loss
        l2_loss_count += 1

        current_total_loss = current_main_loss + current_l1_loss + current_l2_loss
        total_loss_sum += float(current_total_loss) * current_batch_size
        total_loss_count += current_batch_size

        if cfg["use_gradient_accumulation"]:
            if ((batch_idx + 1) < num_batches) or (num_train_samples % effective_batch_size == 0):
                (current_total_loss / float(effective_batch_size)).backward()
            else:
                (current_total_loss / float(num_train_samples % effective_batch_size)).backward()

            if (observed_sample_count % effective_batch_size == 0) or observed_sample_count == num_train_samples:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_norm"])
                optimizer.step()
                optimizer.zero_grad()
        else:
            current_total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_norm"])
            optimizer.step()
            optimizer.zero_grad()

    logger.log(level=logging.INFO, msg=f"Epoch {str(epoch).zfill(3)}, {(5 - len('train')) * ' ' + 'train'.capitalize()} {cfg['loss_function']} loss is {np.round(main_loss_sum / main_loss_count, 2)}.")

    train_loss_dict = {
        cfg["loss_function"]: main_loss_sum / main_loss_count,
    }

    if cfg["l1_reg_diagonal_coeff"] > 0 or cfg["l1_reg_nondiagonal_coeff"] > 0:
        train_loss_dict["l1_loss"] = l1_loss_sum / l1_loss_count

    if cfg["l2_reg_diagonal_coeff"] > 0 or cfg["l2_reg_nondiagonal_coeff"] > 0:
        train_loss_dict["l2_loss"] = l2_loss_sum / l2_loss_count

    train_main_loss_values.append(train_loss_dict[cfg["loss_function"]])

    for loss_name, loss_value in train_loss_dict.items():
        summary_writer.add_scalar(f"train_{loss_name}", loss_value, epoch)

    return train_loss_dict
