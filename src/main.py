import logging

import numpy as np

from train import train
from validate import validate
from test import test
from utils import (get_argument_parser, set_seeds, set_experiment_name, \
                   set_model_hidden_dimension, set_device, get_logger, \
                   get_dataset, get_data_loaders, get_model, get_optimizer, \
                   get_scheduler, get_loss_function, save_model, save_cfg,
                   load_model, save_test_results)


if __name__ == "__main__":
    argument_parser = get_argument_parser()
    cfg = argument_parser.parse_args().__dict__
    set_seeds(cfg=cfg)
    set_experiment_name(cfg=cfg)
    logger = get_logger(cfg=cfg)
    set_device(cfg=cfg, logger=logger)
    save_cfg(cfg=cfg, logger=logger)
    dataset = get_dataset(cfg=cfg, logger=logger)
    data_loaders = get_data_loaders(cfg=cfg, dataset=dataset)
    set_model_hidden_dimension(cfg=cfg, input_dimension=dataset.input_dimension, output_dimension=dataset.output_dimension)
    model = get_model(cfg=cfg, input_dimension=dataset.input_dimension, output_dimension=dataset.output_dimension)
    optimizer = get_optimizer(cfg=cfg, model=model)
    scheduler = get_scheduler(cfg=cfg, optimizer=optimizer)
    train_loss_function = get_loss_function(cfg=cfg, reduction="mean")
    val_test_loss_function = get_loss_function(cfg=cfg, reduction="sum")

    best_val_loss = np.inf
    num_epochs_val_loss_not_decreased = 0

    for epoch in range(cfg["num_epochs"]):
        train(cfg=cfg, data_loaders=data_loaders, model=model, loss_function=train_loss_function, dataset=dataset, optimizer=optimizer)
        loss_dict = validate(cfg=cfg, data_loaders=data_loaders, model=model, loss_function=val_test_loss_function, dataset=dataset, epoch=epoch, logger=logger)
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

    model = load_model(cfg=cfg, dataset=dataset, logger=logger)
    test_results_dict = test(cfg=cfg, data_loaders=data_loaders, model=model, loss_function=val_test_loss_function, dataset=dataset, logger=logger)
    save_test_results(cfg=cfg, test_results_dict=test_results_dict, entrezgene_ids=dataset.entrezgene_ids, logger=logger)
