import numpy as np
from pprint import pprint
from utils import (get_argument_parser, set_seeds, set_experiment_name, \
                   set_device, get_logger, get_dataset, get_model, save_cfg,
                   save_features, save_best_model)
from test import save_results_split_helper


if __name__ == "__main__":
    argument_parser = get_argument_parser()
    cfg = argument_parser.parse_args().__dict__

    set_seeds(cfg=cfg)
    set_experiment_name(cfg=cfg)
    logger = get_logger(cfg=cfg)
    set_device(cfg=cfg, logger=logger)
    dataset = get_dataset(cfg=cfg, logger=logger)

    save_cfg(cfg=cfg, logger=logger)
    pprint(cfg, indent=4)

    # val
    val_ground_truths = []
    val_predictions = []
    val_sample_ids = []
    for fold_idx in range(cfg["num_cv_folds"]):
        train_indices, val_indices, test_indices = dataset.train_val_test_indices[fold_idx]

        X_train = dataset.X[train_indices, :]
        X_val = dataset.X[val_indices, :]

        y_train = dataset.y[train_indices, :]
        y_val = dataset.y[val_indices, :]

        model = get_model(cfg=cfg, logger=logger)
        model.fit(X_train, y_train)

        yhat_val = model.predict(X_val)
        current_val_cancer_types = dataset.all_cancer_types[val_indices]
        current_val_sample_ids = dataset.sample_ids[val_indices]

        if cfg["use_cna_adjusted_zscore"]:
            yhat_val = yhat_val + dataset.cna_adjustment_intercepts[fold_idx].cpu().numpy() + dataset.cna_adjustment_coeffs[fold_idx].cpu().numpy() * X_val[:, :yhat_val.shape[1]]
            y_val = y_val + dataset.cna_adjustment_intercepts[fold_idx].cpu().numpy() + dataset.cna_adjustment_coeffs[fold_idx].cpu().numpy() * X_val[:, :y_val.shape[1]]

        val_ground_truths.append(y_val)
        val_predictions.append(yhat_val)
        val_sample_ids.append(current_val_sample_ids)

    val_ground_truths = np.vstack(val_ground_truths)
    val_predictions = np.vstack(val_predictions)
    val_sample_ids = np.hstack(val_sample_ids)

    # test
    model = get_model(cfg=cfg, logger=logger)
    train_indices, val_indices, test_indices = dataset.train_val_test_indices[0]
    X_train_val = dataset.X[train_indices + val_indices, :]
    X_test = dataset.X[test_indices, :]
    y_train_val = dataset.y[train_indices + val_indices, :]
    test_ground_truths = dataset.y[test_indices, :]
    model.fit(X_train_val, y_train_val)
    save_best_model(cfg=cfg, model=model, logger=logger)
    test_predictions = model.predict(X_test)
    test_sample_ids = dataset.sample_ids[test_indices]

    if cfg["use_cna_adjusted_zscore"]:
        test_predictions = test_predictions + dataset.cna_adjustment_intercepts[cfg["num_cv_folds"]].cpu().numpy() + dataset.cna_adjustment_coeffs[cfg["num_cv_folds"]].cpu().numpy() * X_test[:, :test_predictions.shape[1]]
        test_ground_truths = test_ground_truths + dataset.cna_adjustment_intercepts[cfg["num_cv_folds"]].cpu().numpy() + dataset.cna_adjustment_coeffs[cfg["num_cv_folds"]].cpu().numpy() * X_test[:, :test_predictions.shape[1]]

    save_results_split_helper(cfg=cfg, predictions=val_predictions, ground_truths=val_ground_truths, sample_ids=val_sample_ids, split_name="val", logger=logger)
    save_results_split_helper(cfg=cfg, predictions=test_predictions, ground_truths=test_ground_truths, sample_ids=test_sample_ids, split_name="test", logger=logger)
    save_features(cfg=cfg, dataset=dataset, logger=logger)
