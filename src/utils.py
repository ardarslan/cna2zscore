import os
import sys
import time
import random
import pprint
import logging
import argparse
from uuid import uuid4
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam, AdamW, RMSprop, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, Subset
import joblib

from dataset import UnthresholdedCNA2ZScoreDataset, ThresholdedCNA2ZScoreDataset, \
                    UnthresholdedCNAPurity2ZScoreDataset, ThresholdedCNAPurity2ZScoreDataset, \
                    RPPA2ZScoreDataset
from model import get_single_model, DLPerChromosome, SklearnPerChromosome


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


def set_early_stopping_epoch(cfg: Dict[str, Any], epoch: int, logger: logging.Logger) -> None:
    logger.log(level=logging.INFO, msg=f"Stopped early at epoch {epoch}.")
    cfg["early_stopping_epoch"] = epoch


def get_experiment_dir(cfg: Dict[str, Any]) -> str:
    experiment_dir = os.path.join(cfg["checkpoints_dir"], cfg["experiment_name"])
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir


def set_number_of_parameters(cfg: Dict[str, Any], model: nn.Module) -> None:
    cfg["num_trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    cfg["num_all_parameters"] = sum(p.numel() for p in model.parameters())


def get_dataset(cfg: Dict[str, Any], logger: logging.Logger) -> Dataset:
    logger.log(level=logging.INFO, msg="Creating the dataset...")

    if cfg["dataset"] == "unthresholdedcna2zscore":
        dataset = UnthresholdedCNA2ZScoreDataset(cfg=cfg, logger=logger)
    elif cfg["dataset"] == "thresholdedcna2zscore":
        dataset = ThresholdedCNA2ZScoreDataset(cfg=cfg, logger=logger)
    elif cfg["dataset"] == "unthresholdedcnapurity2zscore":
        dataset = UnthresholdedCNAPurity2ZScoreDataset(cfg=cfg, logger=logger)
    elif cfg["dataset"] == "thresholdedcnapurity2zscore":
        dataset = ThresholdedCNAPurity2ZScoreDataset(cfg=cfg, logger=logger)
    elif cfg["dataset"] == "rppa2zscore":
        dataset = RPPA2ZScoreDataset(cfg=cfg, logger=logger)
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


def get_summary_writer(cfg: Dict[str, Any]):
    experiment_dir = get_experiment_dir(cfg=cfg)
    return SummaryWriter(os.path.join(experiment_dir, "loss_summary.txt"))


def get_data_loaders(cfg: Dict[str, Any], dataset: Dataset, logger: logging.Logger) -> Dict[str, DataLoader]:
    logger.log(level=logging.INFO, msg="Creating the data loaders...")

    train_indices, val_indices, test_indices = dataset.train_val_test_indices[0]

    train_data_loader = DataLoader(Subset(dataset, train_indices), batch_size=cfg["batch_size"], shuffle=True)
    val_data_loader = DataLoader(Subset(dataset, val_indices), batch_size=cfg["batch_size"], shuffle=False)
    test_data_loader = DataLoader(Subset(dataset, test_indices), batch_size=cfg["batch_size"], shuffle=False)

    data_loaders = {
        "train": train_data_loader,
        "val": val_data_loader,
        "test": test_data_loader,
    }

    logger.log(level=logging.INFO, msg="Created the data loaders.")

    return data_loaders


def get_model(cfg: Dict[str, Any], logger: logging.Logger) -> torch.nn.Module:
    logger.log(level=logging.INFO, msg="Creating the model...")

    if cfg["per_chromosome"]:
        if "dl" in cfg["model"]:
            model = DLPerChromosome(cfg=cfg)
        elif "sklearn" in cfg["model"]:
            model = SklearnPerChromosome(cfg=cfg)
        else:
            raise Exception(f"{cfg['model']} is not a valid model name.")
    else:
        model = get_single_model(cfg=cfg, input_dimension=cfg["input_dimension"], output_dimension=cfg["output_dimension"])

    if "dl" in cfg["model"]:
        model = model.float().to(cfg["device"])

    logger.log(level=logging.INFO, msg="Created the model.")

    return model


def get_optimizer(cfg: Dict[str, Any], model: torch.nn.Module):
    if cfg["optimizer"] == "sgd":
        return SGD(params=model.parameters(), lr=cfg["learning_rate"], momentum=0.90)
    elif cfg["optimizer"] == "adam":
        return Adam(params=model.parameters(), lr=cfg["learning_rate"])
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


def save_loss_values(cfg: Dict[str, Any], train_main_loss_values: List[float], val_main_loss_values: List[float]) -> None:
    experiment_dir = get_experiment_dir(cfg=cfg)
    loss_values_df = pd.DataFrame.from_dict({"epoch": np.arange(1, len(train_main_loss_values)+1), f"train_{cfg['loss_function']}": train_main_loss_values, f"val_{cfg['loss_function']}": val_main_loss_values})
    loss_values_df.to_csv(os.path.join(experiment_dir, "loss_values.tsv"), sep="\t")


def save_best_model(cfg: Dict[str, Any], model: nn.Module, logger: logging.Logger) -> None:
    logger.log(level=logging.INFO, msg="Saving the best model...")
    experiment_dir = get_experiment_dir(cfg=cfg)
    if "dl" in cfg["model"]:
        torch.save(model.state_dict(), os.path.join(experiment_dir, "best_model"))
    else:
        joblib.dump(model, os.path.join(experiment_dir, "best_model"))
    logger.log(level=logging.INFO, msg="Saved the best model")


def load_best_model(cfg: Dict[str, Any], logger: logging.Logger) -> nn.Module:
    logger.log(level=logging.INFO, msg="Loading the best model...")
    experiment_dir = get_experiment_dir(cfg=cfg)
    if "dl" in cfg["model"]:
        model = get_model(cfg=cfg, logger=logger)
        model.load_state_dict(torch.load(os.path.join(experiment_dir, "best_model")))
    else:
        model = joblib.load(os.path.join(experiment_dir, "best_model"))
    logger.log(level=logging.INFO, msg="Loaded the best model.")
    return model


def delete_best_model(cfg: Dict[str, Any], logger: logging.Logger) -> None:
    logger.log(level=logging.INFO, msg="Deleting the best model...")
    experiment_dir = get_experiment_dir(cfg=cfg)
    best_model_path = os.path.join(experiment_dir, "best_model")
    os.remove(path=best_model_path)
    logger.log(level=logging.INFO, msg="Deleted the best model.")


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
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")

    # data
    parser.add_argument("--processed_data_dir", type=str, default="/cluster/scratch/aarslan/cna2gex_data/processed", help="Directory for the processed files.")
    parser.add_argument("--dataset", type=str, default="rppa2zscore", choices=["unthresholdedcnapurity2zscore", "thresholdedcnapurity2zscore", "unthresholdedcnapurity2zscore", "thresholdedcna2zscore", "unthresholdedcna2zscore", "rppa2zscore"], help="Name of the dataset.")
    parser.add_argument("--gene_type", type=str, default="rppa_genes", choices=["168_highly_expressed_genes", "1000_highly_expressed_genes", "rppa_genes", "all_genes"])
    parser.add_argument("--cancer_type", type=str, default="blca", choices=["blca", "skcm", "thcm", "sarc", "prad", "pcpg", "paad", "hnsc", "esca", "coad", "cesc", "brca", "blca", "tgct", "kirp", "kirc", "laml", "read", "ov", "luad", "lihc", "ucec", "gbm", "lgg", "ucs", "thym", "stad", "dlbc", "lusc", "meso", "kich", "uvm", "chol", "acc", "all"], help="Cancer type.")
    parser.add_argument("--split_ratios", type=dict, default={"train": 0.6, "val": 0.2, "test": 0.2}, help="Ratios for train, val and test splits.")

    # model
    parser.add_argument("--model", type=str, default="dl_linear", choices=["dl_gene_embeddings", "dl_per_gene", "dl_linear", "dl_mlp", "dl_rescon_mlp", "dl_transformer", "sklearn_linear", "sklearn_per_gene"], help="Which model to use.")
    parser.add_argument("--per_chromosome", type=str2bool, nargs='?', const=True, default=False, help="Whether to use a per chromosome model or not.")
    parser.add_argument("--num_nonlinear_layers", type=int, default=1, help="Number of layers with a nonlinear activation.")
    parser.add_argument("--hidden_dimension_ratio", type=float, default=0.10, help="Ratio of number of nodes in a hidden layer to max(number of nodes in input layer, number of nodes in output layer).")
    parser.add_argument("--hidden_activation", type=str, default="relu", choices=["relu", "leaky_relu"], help="Activation function used to activate each hidden layer's (batch normalized) output.")
    parser.add_argument("--dropout", type=float, default=0.33, help="Probability of zeroing out an entry in a given vector.")

    # ResConMLP specific hyperparameters
    parser.add_argument("--rescon_diagonal_W", type=str2bool, default=True, nargs='?', const=True, help="If model is rescon_mlp, whether to use a diagonal weight matrix or not.")

    # Transformer specific hyperparameters
    parser.add_argument("--gene_embedding_size", type=int, default=4, help="Gene embedding size.")
    parser.add_argument("--num_attention_heads", type=int, default=4, help="Number of attention heads.")
    parser.add_argument("--num_mlp_blocks", type=int, default=4, help="Number of MLP blocks.")

    # optimizer
    parser.add_argument("--optimizer", type=str, default="adam", help="Which optimizer to use.")

    # scheduler
    parser.add_argument("--scheduler", type=str, default="reduce_lr_on_plateau", help="Which scheduler to use.")
    parser.add_argument("--scheduler_factor", type=float, default=0.5, help="Multiplicative factor used by ReduceLROnPlateau scheduler while reducing the learning rate.")
    parser.add_argument("--scheduler_patience", type=int, default=4, help="Number of patience epochs used by ReduceLROnPlateau scheduler.")
    parser.add_argument("--min_lr", type=float, default=2.5e-5, help="Minimum learning rate.")

    # training
    parser.add_argument("--learning_rate", type=float, default=1e-2, help="Learning rate.")
    parser.add_argument("--gradient_norm", type=float, default=10.0, help="Gradient norm.")
    parser.add_argument("--num_epochs", type=int, default=200, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--loss_function", type=str, default="mse", help="Loss function.")
    parser.add_argument("--l1_reg_diagonal_coeff", type=float, default=1e-4, help="L1 regularization coefficient for diagonal elements.")
    parser.add_argument("--l1_reg_nondiagonal_coeff", type=float, default=1e-4, help="L1 regularization coefficient for nondiagonal elements.")
    parser.add_argument("--l2_reg_diagonal_coeff", type=float, default=0.0, help="L2 regularization coefficient for diagonal elements.")
    parser.add_argument("--l2_reg_nondiagonal_coeff", type=float, default=0.0, help="L2 regularization coefficient for nondiagonal elements.")
    parser.add_argument("--early_stopping_patience", type=int, default=8, help="Number of epochs to wait without an improvement in validation loss, before stopping the training.")

    # checkpoints
    parser.add_argument("--checkpoints_dir", type=str, default="/cluster/scratch/aarslan/cna2zscore_checkpoints")

    # logging
    parser.add_argument("--log_level", type=str, default="info")

    return parser
