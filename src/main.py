import gc
import logging

import torch
import numpy as np

from train import train
from validate import validate
from test import save_results
from utils import (get_argument_parser, set_seeds, set_experiment_name, \
                   set_model_hidden_dimension, set_device, get_logger, \
                   get_dataset, get_data_loaders, get_model, get_optimizer, \
                   get_scheduler, get_loss_function, save_model, save_cfg, \
                   load_model, set_early_stopping_epoch, get_summary_writer, \
                   set_hyperparameters_according_to_memory_limits, save_loss_values,
                   set_number_of_parameters)


if __name__ == "__main__":
    argument_parser = get_argument_parser()
    cfg = argument_parser.parse_args().__dict__

    set_seeds(cfg=cfg)
    set_experiment_name(cfg=cfg)
    logger = get_logger(cfg=cfg)
    summary_writer = get_summary_writer(cfg=cfg)
    set_device(cfg=cfg, logger=logger)
    dataset = get_dataset(cfg=cfg, logger=logger)
    set_model_hidden_dimension(cfg=cfg, input_dimension=dataset.input_dimension, output_dimension=dataset.output_dimension)
    set_hyperparameters_according_to_memory_limits(cfg=cfg)
    data_loaders = get_data_loaders(cfg=cfg, dataset=dataset, logger=logger)
    model = get_model(cfg=cfg, input_dimension=dataset.input_dimension, output_dimension=dataset.output_dimension, logger=logger)
    set_number_of_parameters(cfg=cfg, model=model)
    save_cfg(cfg=cfg, logger=logger)
    optimizer = get_optimizer(cfg=cfg, model=model)
    scheduler = get_scheduler(cfg=cfg, optimizer=optimizer)
    train_loss_function = get_loss_function(cfg=cfg, reduction="mean")
    val_test_loss_function = get_loss_function(cfg=cfg, reduction="sum")

    best_val_loss = np.inf
    num_epochs_val_loss_not_decreased = 0

    train_main_loss_values = []
    val_main_loss_values = []

    logger.log(level=logging.INFO, msg="Starting training...")
    for epoch in range(1, cfg["num_epochs"] + 1):
        current_train_loss_dict = train(cfg=cfg, data_loaders=data_loaders, model=model, loss_function=train_loss_function, dataset=dataset, optimizer=optimizer, epoch=epoch, logger=logger, summary_writer=summary_writer, train_main_loss_values=train_main_loss_values)
        current_val_loss_dict = validate(cfg=cfg, data_loaders=data_loaders, model=model, loss_function=val_test_loss_function, dataset=dataset, epoch=epoch, logger=logger, summary_writer=summary_writer, val_main_loss_values=val_main_loss_values)

        if current_val_loss_dict[cfg["loss_function"]] < best_val_loss:
            num_epochs_val_loss_not_decreased = 0
            best_val_loss = current_val_loss_dict[cfg["loss_function"]]
            save_model(cfg=cfg, model=model, logger=logger)
        else:
            num_epochs_val_loss_not_decreased += 1

        if num_epochs_val_loss_not_decreased == cfg["early_stopping_patience"]:
            set_early_stopping_epoch(cfg=cfg, epoch=epoch, logger=logger)
            break
        else:
            scheduler.step(current_val_loss_dict[cfg["loss_function"]])

        summary_writer.add_scalar(f"learning_rate", optimizer.param_groups[0]['lr'], epoch)

    logger.log(level=logging.INFO, msg="Finished training.")
    save_cfg(cfg=cfg, logger=logger)
    save_loss_values(cfg=cfg, train_main_loss_values=train_main_loss_values, val_main_loss_values=val_main_loss_values)

    del model
    del optimizer
    del scheduler
    gc.collect()
    torch.cuda.empty_cache()

    model = load_model(cfg=cfg, dataset=dataset, logger=logger)
    save_results(cfg=cfg, data_loaders=data_loaders, model=model, loss_function=val_test_loss_function, dataset=dataset, logger=logger)
