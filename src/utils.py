import os
import sys
import time
import random
import pprint
import logging
import argparse
from uuid import uuid4
from typing import Dict, Any, Union

import numpy as np
import torch
import torch.nn as nn
import torchsummary
from torch.optim import Adam, AdamW, RMSprop
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, Subset

from dataset import UnthresholdedCNA2GEXDataset, ThresholdedCNA2GEXDataset, \
                    UnthresholdedCNAPurity2GEXDataset, ThresholdedCNAPurity2GEXDataset, \
                    RPPA2GEXDataset
from model import MLP


def set_seeds(cfg: Dict[str, Any]) -> None:
    seed = cfg["seed"]

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def set_experiment_name(cfg: Dict[str, Any]) -> None:
    cfg["experiment_name"] = f"{int(time.time())}_{uuid4().hex}"


def set_device(cfg: Dict[str, Any], logger: logging.Logger) -> None:
    if torch.cuda.is_available():
        cfg["device"] = "cuda"
    else:
        cfg["device"] = "cpu"
    logger.log(level=logging.INFO, msg=f"Using {cfg['device']}...")


def set_model_hidden_dimension(cfg: Dict[str, Any], input_dimension: int, output_dimension: int) -> None:
    if cfg["hidden_dimension"] == "max":
        cfg["hidden_dimension"] = np.max([input_dimension, output_dimension])
    elif cfg["hidden_dimension"] == "mean":
        cfg["hidden_dimension"] = np.mean([input_dimension, output_dimension])
    elif cfg["hidden_dimension"] == "min":
        cfg["hidden_dimension"] = np.min([input_dimension, output_dimension])
    else:
        try:
            cfg["hidden_dimension"] = int(cfg["hidden_dimension"])
        except ValueError:
            raise Exception(f"{cfg['hidden_dimension']} is not a valid hidden_dimension.")


def set_early_stopping_epoch(cfg: Dict[str, Any], epoch: int, logger: logging.Logger) -> None:
    logger.log(level=logging.INFO, msg=f"Stopped early at epoch {epoch}.")
    cfg["early_stopping_epoch"] = epoch


def get_experiment_dir(cfg: Dict[str, Any]) -> str:
    experiment_dir = os.path.join(cfg["checkpoints_dir"], cfg["experiment_name"])
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir


def set_hyperparameters_according_to_memory_limits(cfg: Dict[str, Any]) -> None:
    if cfg["hidden_dimension"] > 5000:
        cfg["real_batch_size"] = 1
        cfg["effective_batch_size"] = cfg["batch_size"]
        cfg["use_gradient_accumulation"] = True
        cfg["normalization_type"] = "instance_normalization"
    else:
        cfg["real_batch_size"] = cfg["batch_size"]
        cfg["effective_batch_size"] = cfg["batch_size"]
        cfg["use_gradient_accumulation"] = False
        cfg["normalization_type"] = "batch_normalization"


def get_dataset(cfg: Dict[str, Any], logger: logging.Logger) -> Dataset:
    logger.log(level=logging.INFO, msg="Creating the dataset...")

    if cfg["dataset"] == "unthresholdedcna2gex":
        dataset = UnthresholdedCNA2GEXDataset(cfg=cfg, logger=logger)
    elif cfg["dataset"] == "thresholdedcna2gex":
        dataset = ThresholdedCNA2GEXDataset(cfg=cfg, logger=logger)
    elif cfg["dataset"] == "unthresholdedcnapurity2gex":
        dataset = UnthresholdedCNAPurity2GEXDataset(cfg=cfg, logger=logger)
    elif cfg["dataset"] == "thresholdedcnapurity2gex":
        dataset = ThresholdedCNAPurity2GEXDataset(cfg=cfg, logger=logger)
    elif cfg["dataset"] == "rppa2gex":
        dataset = RPPA2GEXDataset(cfg=cfg, logger=logger)
    else:
        raise NotImplementedError(f"{cfg['dataset']} is not an implemented dataset.")

    logger.log(level=logging.INFO, msg="Created the dataset.")
    return dataset


def get_logger(cfg: Dict[str, Any]) -> logging.Logger:
    """
    Initialize logger to stdout and optionally also a log file.
    Args:
        path_to_log: optional, if given message will also be appended to file.
        level: optional, logging level, see Python's logging library.
        format_str: optional, logging format string, see Python's logging library.
    """
    if cfg["log_level"] == "debug":
        log_level = logging.DEBUG
    elif cfg["log_level"] == "info":
        log_level = logging.INFO
    else:
        raise Exception(f"{log_level} is not a valid log_level.")

    experiment_dir = get_experiment_dir(cfg=cfg)
    log_file_path = os.path.join(experiment_dir, "logs.txt")

    logger = logging.getLogger()
    logger.setLevel(log_level)

    log_format_str = "[%(asctime)s|%(levelname)s] %(message)s"
    formatter = logging.Formatter(log_format_str)

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(log_level)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def get_data_loaders(cfg: Dict[str, Any], dataset: Dataset, logger: logging.Logger) -> Dict[str, DataLoader]:
    logger.log(level=logging.INFO, msg="Creating the data loaders...")

    train_data_loader = DataLoader(Subset(dataset, dataset.train_indices), batch_size=cfg["real_batch_size"], shuffle=True)
    val_data_loader = DataLoader(Subset(dataset, dataset.val_indices), batch_size=cfg["real_batch_size"], shuffle=False)
    test_data_loader = DataLoader(Subset(dataset, dataset.test_indices), batch_size=cfg["real_batch_size"], shuffle=False)

    data_loaders = {
        "train": train_data_loader,
        "val": val_data_loader,
        "test": test_data_loader,
    }

    logger.log(level=logging.INFO, msg="Created the data loaders.")

    return data_loaders


def get_model(cfg: Dict[str, Any], input_dimension: int, output_dimension: int, logger: logging.Logger) -> torch.nn.Module:
    logger.log(level=logging.INFO, msg="Creating the model...")

    if cfg["model"] == "mlp":
        model = MLP(cfg=cfg, input_dimension=input_dimension, output_dimension=output_dimension).float().to(cfg["device"])
    else:
        raise NotImplementedError(f"{cfg['model']} is not an implemented model.")

    torchsummary.summary(model, input_size=(input_dimension, ))

    logger.log(level=logging.INFO, msg="Created the model.")

    return model


def get_optimizer(cfg: Dict[str, Any], model: torch.nn.Module):
    if cfg["optimizer"] == "adam":
        return Adam(params=model.parameters())
    elif cfg["optimizer"] == "adamw":
        return AdamW(params=model.parameters())
    elif cfg["optimizer"] == "rmsprop":
        return RMSprop(params=model.parameters())
    else:
        raise NotImplementedError(f"{cfg['optimizer']} is not an implemented optimizer.")


def get_scheduler(cfg: Dict[str, Any], optimizer):
    if cfg["scheduler"] == "reduce_lr_on_plateau":
        scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=cfg["scheduler_factor"], patience=cfg["scheduler_patience"], verbose=True)
    else:
        raise NotImplementedError(f"{scheduler} is not an implemented scheduler.")
    return scheduler


def get_loss_function(cfg: Dict[str, Any], reduction: str):
    if cfg["loss_function"] == "mse":
        return torch.nn.MSELoss(reduction=reduction)
    else:
        raise NotImplementedError(f"{cfg['loss']} is not an implemented loss function.")


def save_cfg(cfg: Dict[str, Any], logger: logging.Logger) -> None:
    logger.log(level=logging.INFO, msg="Saving the config file...")
    experiment_dir = get_experiment_dir(cfg=cfg)
    config_path = os.path.join(experiment_dir, "cfg.txt")
    with open(config_path, "w") as file_handler:
        file_handler.write(pprint.pformat(cfg, indent=4))


def save_model(cfg: Dict[str, Any], model: nn.Module, logger: logging.Logger) -> None:
    logger.log(level=logging.INFO, msg="Saving the best model...")
    experiment_dir = get_experiment_dir(cfg=cfg)
    torch.save(model.state_dict(), os.path.join(experiment_dir, "best_model"))
    logger.log(level=logging.INFO, msg="Saved the best model")


def load_model(cfg: Dict[str, Any], dataset: Dataset, logger: logging.Logger) -> nn.Module:
    logger.log(level=logging.INFO, msg="Loading the best model...")
    experiment_dir = get_experiment_dir(cfg=cfg)
    model = get_model(cfg=cfg, input_dimension=dataset.input_dimension, output_dimension=dataset.output_dimension, logger=logger)
    model.load_state_dict(torch.load(os.path.join(experiment_dir, "best_model")))
    logger.log(level=logging.INFO, msg="Loaded the best model.")
    return model


def str2bool(v: Union[str, bool]) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ('true', 't', '1'):
        return True
    elif v.lower() in ('false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_argument_parser() -> argparse.ArgumentParser:
    """Initialize basic CLI argument parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1903, help="Random seed for reproducibility.")

    # data
    parser.add_argument("--processed_data_dir", type=str, default="data/processed/", help="Directory for the processed files.")
    parser.add_argument("--dataset", type=str, default="unthresholdedcnapurity2gex", choices=["unthresholdedcnapurity2gex", "thresholdedcnapurity2gex", "unthresholdedcnapurity2gex", "thresholdedcna2gex", "unthresholdedcna2gex", "rppa2gex"], help="Name of the dataset.")
    parser.add_argument("--cancer_type", type=str, default="all", choices=["blca", "skcm", "thcm", "sarc", "prad", "pcpg", "paad", "hnsc", "esca", "coad", "cesc", "brca", "blca", "tgct", "kirp", "kirc", "laml", "read", "ov", "luad", "lihc", "ucec", "gbm", "lgg", "ucs", "thym", "stad", "dlbc", "lusc", "meso", "kich", "uvm", "chol", "acc", "all"], help="Cancer type.")
    parser.add_argument("--split_ratios", type=dict, default={"train": 0.6, "val": 0.2, "test": 0.2}, help="Ratios for train, val and test splits.")
    parser.add_argument("--normalize_input", type=str2bool, nargs='?', const=True, default=True, help="Whether to normalize the input or not.")
    parser.add_argument("--normalize_output", type=str2bool, nargs='?', const=True, default=True, help="Whether to normalize the output or not.")
    parser.add_argument("--normalization_eps", type=float, default=1e-10, help="Epsilon value used during normalizing input or output, for numerical stability.")

    # model
    parser.add_argument("--model", type=str, default="mlp", help="Which model to use.")
    parser.add_argument("--num_hidden_layers", type=int, default=2, help="Number of layers except the output layer.")
    parser.add_argument("--hidden_dimension", default=10000, help="Number of nodes in each hidden layer. Whether an integer or one of the following strings: 'max', 'min' or 'mean'. When one of these strings, the operation is applied to the input dimension and the output dimension of the model.")
    parser.add_argument("--hidden_activation", type=str, default="leaky_relu", choices=["relu", "leaky_relu"], help="Activation function used to activate each hidden layer's (batch normalized) output.")
    parser.add_argument("--use_residual_connection", type=str2bool, default=False, nargs='?', const=True, help="Whether to use residual connection between hidden layers or not.")
    parser.add_argument("--dropout", type=float, default=0.0, help="Probability of zeroing out an entry in a given vector.")

    # optimizer
    parser.add_argument("--optimizer", type=str, default="adam", help="Which optimizer to use.")

    # scheduler
    parser.add_argument("--scheduler", type=str, default="reduce_lr_on_plateau", help="Which scheduler to use.")
    parser.add_argument("--scheduler_factor", type=float, default=0.5, help="Multiplicative factor used by ReduceLROnPlateau scheduler while reducing the learning rate.")
    parser.add_argument("--scheduler_patience", type=int, default=5, help="Number of patience epochs used by ReduceLROnPlateau scheduler.")

    # training
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--loss_function", type=str, default="mse", help="Loss function.")
    parser.add_argument("--l1_reg_coeff", type=float, default=0.0, help="L1 regularization coefficient.")
    parser.add_argument("--l2_reg_coeff", type=float, default=0.0, help="L2 regularization coefficient.")
    parser.add_argument("--early_stopping_patience", type=int, default=10, help="Number of epochs to wait without an improvement in validation loss, before stopping the training.")

    # checkpoints
    parser.add_argument("--checkpoints_dir", type=str, default="cna2gex_checkpoints")

    # logging
    parser.add_argument("--log_level", type=str, default="info")

    return parser
