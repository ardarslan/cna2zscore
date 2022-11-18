import os
import sys
import time
import random
import pprint
import logging
import argparse
from uuid import uuid4
from typing import Dict, Any, List, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchsummary
from torch.optim import Adam, AdamW, RMSprop
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, Subset

from dataset import CNAPurity2GEXDataset, RPPA2GEXDataset, AverageGEXSubtype2GEXDataset
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


def get_experiment_dir(cfg: Dict[str, Any]) -> str:
    experiment_dir = os.path.join(cfg["checkpoints_dir"], cfg["experiment_name"])
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir


def get_dataset(cfg: Dict[str, Any], logger: logging.Logger) -> Dataset:
    if cfg["dataset"] == "cnapurity2gex":
        return CNAPurity2GEXDataset(cfg=cfg, logger=logger)
    elif cfg["dataset"] == "rppa2gex":
        return RPPA2GEXDataset(cfg=cfg, logger=logger)
    elif cfg["dataset"] == "avggexsubtype2gex":
        return AverageGEXSubtype2GEXDataset(cfg=cfg, logger=logger)
    else:
        raise NotImplementedError(f"{cfg['dataset']} is not an implemented dataset.")


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


def get_data_loaders(cfg: Dict[str, Any], dataset: Dataset) -> Dict[str, DataLoader]:
    batch_size = cfg["batch_size"]

    train_data_loader = DataLoader(Subset(dataset, dataset.train_idx), batch_size=batch_size, shuffle=True)
    val_data_loader = DataLoader(Subset(dataset, dataset.val_idx), batch_size=batch_size, shuffle=False)
    test_data_loader = DataLoader(Subset(dataset, dataset.test_idx), batch_size=batch_size, shuffle=False)

    data_loaders = {
        "train": train_data_loader,
        "val": val_data_loader,
        "test": test_data_loader,
    }
    return data_loaders


def get_model(cfg: Dict[str, Any], input_dimension: int, output_dimension: int) -> torch.nn.Module:
    if cfg["model"] == "mlp":
        model = MLP(cfg=cfg, input_dimension=input_dimension, output_dimension=output_dimension).float().to(cfg["device"])
    else:
        raise NotImplementedError(f"{cfg['model']} is not an implemented model.")

    torchsummary.summary(model, input_size=(input_dimension, ))
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


def save_test_results(cfg: Dict[str, Any], test_results_dict: Dict[str, Any], entrezgene_ids: List[int], logger: logging.Logger) -> None:
    logger.log(level=logging.INFO, msg="Saving test results...")
    experiment_dir = get_experiment_dir(cfg=cfg)

    all_ground_truths = test_results_dict["all_ground_truths"]
    all_predictions = test_results_dict["all_predictions"]
    all_loss = test_results_dict["all_loss"]
    cna_loss = test_results_dict["cna_loss"]
    noncna_loss = test_results_dict["noncna_loss"]
    all_corr = test_results_dict["all_corr"]
    cna_corr = test_results_dict["cna_corr"]
    noncna_corr = test_results_dict["noncna_corr"]
    all_p_value = test_results_dict["all_p_value"]
    cna_p_value = test_results_dict["cna_p_value"]
    noncna_p_value = test_results_dict["noncna_p_value"]

    best_predicted_20_genes = test_results_dict["best_predicted_20_genes"]
    worst_predicted_20_genes = test_results_dict["worst_predicted_20_genes"]

    test_ground_truths_df = pd.DataFrame(data=all_ground_truths, columns=entrezgene_ids)
    test_predictions_df = pd.DataFrame(data=all_predictions, columns=entrezgene_ids)
    test_evaluation_metrics_df = pd.DataFrame.from_dict({
        "metric_name": [
                        f"all_{cfg['loss_function']}",
                        f"cna_{cfg['loss_function']}",
                        f"noncna_{cfg['loss_function']}",
                        "all_corr",
                        "cna_corr",
                        "noncna_corr",
                        "all_p_value",
                        "cna_p_value",
                        "noncna_p_value"
                        ],
        "metric_value": [
                         all_loss,
                         cna_loss,
                         noncna_loss,
                         all_corr,
                         cna_corr,
                         noncna_corr,
                         all_p_value,
                         cna_p_value,
                         noncna_p_value
                        ]
    })
    best_predicted_20_genes_df = pd.DataFrame(data=best_predicted_20_genes, columns=["entrezgene_id", "test_mse"])
    worst_predicted_20_genes_df = pd.DataFrame(data=worst_predicted_20_genes, columns=["entrezgene_id", "test_mse"])

    os.makedirs(os.path.join(experiment_dir, "test_results"), exist_ok=True)
    test_ground_truths_df.to_csv(os.path.join(experiment_dir, "test_results", "ground_truths.tsv"), sep="\t")
    test_predictions_df.to_csv(os.path.join(experiment_dir, "test_results", "predictions.tsv"), sep="\t")
    test_evaluation_metrics_df.to_csv(os.path.join(experiment_dir, "test_results", "evaluation_metrics.tsv"), sep="\t")
    best_predicted_20_genes_df.to_csv(os.path.join(experiment_dir, "test_results", "best_predicted_20_genes.tsv"), sep="\t")
    worst_predicted_20_genes_df.to_csv(os.path.join(experiment_dir, "test_results", "worst_predicted_20_genes.tsv"), sep="\t")

    plt.figure(figsize=(12, 12))
    plt.title(f"Correlation: {np.round(all_corr, 2)}, P-value: {np.round(all_p_value, 2)}")
    plt.scatter(x=all_ground_truths.ravel(), y=all_predictions.ravel(), alpha=0.1)
    plt.imsave(os.path.join(experiment_dir, "test_results", "scatter_plot.png"))


def load_model(cfg: Dict[str, Any], dataset: Dataset, logger: logging.Logger) -> nn.Module:
    logger.log(level=logging.INFO, msg="Loading the best model...")
    experiment_dir = get_experiment_dir(cfg=cfg)
    model = get_model(cfg=cfg, input_dimension=dataset.input_dimension, output_dimension=dataset.output_dimension)
    model.load_state_dict(torch.load(os.path.join(experiment_dir, "best_model")))
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
    parser.add_argument("--processed_data_dir", type=str, default="../data/processed/", help="Directory for the processed files.")
    parser.add_argument("--dataset", type=str, default="cnapurity2gex", choices=["cnapurity2gex", "rppa2gex", "avggexsubtype2gex"], help="Name of the dataset.")
    parser.add_argument("--cancer_type", type=str, default="all", choices=["blca", "all"], help="Cancer type.")
    parser.add_argument("--split_ratios", type=dict, default={"train": 0.6, "val": 0.2, "test": 0.2}, help="Ratios for train, val and test splits.")
    parser.add_argument("--normalize_input", type=str2bool, nargs='?', const=True, default=False, help="Whether to normalize the input or not.")
    parser.add_argument("--normalize_output", type=str2bool, nargs='?', const=True, default=False, help="Whether to normalize the output or not.")

    # model
    parser.add_argument("--model", type=str, default="mlp", help="Which model to use.")
    parser.add_argument("--num_hidden_layers", type=int, default=1, help="Number of layers except the output layer.")
    parser.add_argument("--hidden_dimension", default="max", help="Number of nodes in each hidden layer. Whether an integer or one of the following strings: 'max', 'min' or 'mean'. When one of these strings, the operation is applied to the input dimension and the output dimension of the model.")
    parser.add_argument("--hidden_activation", type=str, default="relu", help="Activation function used to activate each hidden layer's (batch normalized) output.")
    parser.add_argument("--use_residual_connection", type=str2bool, default=False, nargs='?', const=True, help="Whether to use residual connection between hidden layers or not.")
    parser.add_argument("--use_batch_normalization", type=str2bool, default=True, nargs='?', const=True, help="Whether to use batch normalization after each hidden layer or not.")
    parser.add_argument("--dropout", type=float, default=0.0, help="Probability of zeroing out an entry in a given vector.")

    # optimizer
    parser.add_argument("--optimizer", type=str, default="adam", help="Which optimizer to use.")

    # scheduler
    parser.add_argument("--scheduler", type=str, default="reduce_lr_on_plateau", help="Which scheduler to use.")
    parser.add_argument("--scheduler_factor", type=float, default=0.5, help="Multiplicative factor used by ReduceLROnPlateau scheduler while reducing the learning rate.")
    parser.add_argument("--scheduler_patience", type=int, default=5, help="Number of patience epochs used by ReduceLROnPlateau scheduler.")

    # training
    parser.add_argument("--num_epochs", type=int, default=200, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size.")
    parser.add_argument("--loss_function", type=str, default="mse", help="Loss function.")
    parser.add_argument("--early_stopping_patience", type=int, default=10, help="Number of epochs to wait without an improvement in validation loss, before stopping the training.")

    # checkpoints
    parser.add_argument("--checkpoints_dir", type=str, default="../checkpoints")

    # logging
    parser.add_argument("--log_level", type=str, default="info")

    return parser
