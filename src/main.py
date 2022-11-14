import logging

import numpy as np

from train import train
from evaluate import evaluate
from utils import (get_argument_parser, set_seeds, set_experiment_name, \
                   set_model_hidden_dimension, set_device, get_logger, \
                   get_dataset, get_data_loaders, get_model, get_optimizer, \
                   get_scheduler, get_loss_function, save_model)


if __name__ == "__main__":
    argument_parser = get_argument_parser()
    cfg = argument_parser.parse_args().__dict__
    set_seeds(cfg=cfg)
    set_experiment_name(cfg=cfg)
    set_device(cfg=cfg)
    logger = get_logger(cfg=cfg)
    dataset = get_dataset(cfg=cfg)
    data_loaders = get_data_loaders(cfg=cfg, dataset=dataset)
    set_model_hidden_dimension(cfg=cfg, input_dimension=dataset.input_dimension, output_dimension=dataset.output_dimension)
    model = get_model(cfg=cfg, input_dimension=dataset.input_dimension, output_dimension=dataset.output_dimension)
    optimizer = get_optimizer(cfg=cfg, model=model)
    scheduler = get_scheduler(cfg=cfg, optimizer=optimizer)
    loss_function = get_loss_function(cfg=cfg)

    best_val_loss = np.inf
    num_epochs_val_loss_not_decreased = 0

    for epoch in range(cfg["num_epochs"]):
        train(cfg=cfg, data_loaders=data_loaders, model=model, loss_function=loss_function, dataset=dataset, optimizer=optimizer)
        loss_dict = evaluate(cfg=cfg, data_loaders=data_loaders, model=model, loss_function=loss_function, dataset=dataset, epoch=epoch, logger=logger)
        current_val_loss = loss_dict["val"]

        if current_val_loss < best_val_loss:
            num_epochs_val_loss_not_decreased = 0
            best_val_loss = current_val_loss
            save_model(cfg=cfg, model=model, logger=logger)
        else:
            num_epochs_val_loss_not_decreased += 1

        if num_epochs_val_loss_not_decreased == cfg["early_stopping_patience"]:
            logger.log(level=logging.INFO, msg=f"Stopped early at epoch {epoch}.")
            break
        else:
            scheduler.step(current_val_loss)
