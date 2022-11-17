import logging

import numpy as np

from train import train
from validate import validate
from test import test
from utils import (get_argument_parser, set_seeds, set_experiment_name, \
                   set_model_hidden_dimension, set_device, get_logger, \
                   get_dataset, get_data_loaders, get_model, get_optimizer, \
                   get_scheduler, get_loss_function, save_model, save_cfg,
                   load_model, remove_logger)


if __name__ == "__main__":
    argument_parser = get_argument_parser()
    cfg = argument_parser.parse_args().__dict__
    set_seeds(cfg=cfg)
    set_experiment_name(cfg=cfg)
    set_device(cfg=cfg)
    save_cfg(cfg=cfg)
    train_val_logger = get_logger(cfg=cfg, file_name="logs.txt")
    dataset = get_dataset(cfg=cfg, logger=train_val_logger)
    data_loaders = get_data_loaders(cfg=cfg, dataset=dataset)
    set_model_hidden_dimension(cfg=cfg, input_dimension=dataset.input_dimension, output_dimension=dataset.output_dimension)
    model = get_model(cfg=cfg, input_dimension=dataset.input_dimension, output_dimension=dataset.output_dimension)
    optimizer = get_optimizer(cfg=cfg, model=model)
    scheduler = get_scheduler(cfg=cfg, optimizer=optimizer)
    train_eval_loss_function = get_loss_function(cfg=cfg, reduction="mean")

    best_val_loss = np.inf
    num_epochs_val_loss_not_decreased = 0

    for epoch in range(cfg["num_epochs"]):
        train(cfg=cfg, data_loaders=data_loaders, model=model, loss_function=train_eval_loss_function, dataset=dataset, optimizer=optimizer)
        loss_dict = validate(cfg=cfg, data_loaders=data_loaders, model=model, loss_function=train_eval_loss_function, dataset=dataset, epoch=epoch, logger=train_val_logger)
        current_val_loss = loss_dict["val"]

        if current_val_loss < best_val_loss:
            num_epochs_val_loss_not_decreased = 0
            best_val_loss = current_val_loss
            save_model(cfg=cfg, model=model, logger=train_val_logger)
        else:
            num_epochs_val_loss_not_decreased += 1

        if num_epochs_val_loss_not_decreased == cfg["early_stopping_patience"]:
            train_val_logger.log(level=logging.INFO, msg=f"Stopped early at epoch {epoch}.")
            break
        else:
            scheduler.step(current_val_loss)
    remove_logger(cfg=cfg, logger=train_val_logger)

    test_logger = get_logger(cfg=cfg, file_name="results.txt")
    test_loss_function = get_loss_function(cfg=cfg, reduction="sum")
    model = load_model(cfg=cfg)
    test(cfg=cfg, data_loaders=data_loaders, model=model, loss_function=test_loss_function, dataset=dataset, logger=test_logger)
