import os
import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from utils import get_experiment_dir
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error


def plot_distributions_of_gene_based_pearson_corrs(cfg: Dict[str, Any], all_ground_truths: pd.DataFrame, all_predictions: pd.DataFrame, thresholded_cna_mask: pd.DataFrame, split_name: str) -> None:
    experiment_dir = get_experiment_dir(cfg=cfg)

    current_thresholded_cna_mask = thresholded_cna_mask[thresholded_cna_mask["sample_id"].isin(all_ground_truths["sample_id"].values)].drop(columns=["sample_id"]).values
    current_ground_truths = all_ground_truths.drop(columns=["sample_id"]).values
    current_predictions = all_predictions.drop(columns=["sample_id"]).values

    prediction_corrs = []
    ground_truth_corrs = []

    for j in range(current_ground_truths.shape[1]):
        current_df = pd.DataFrame.from_dict({"thresholded_cna_mask": current_thresholded_cna_mask[:, j].ravel(),
                                             "ground_truth": current_ground_truths[:, j].ravel(),
                                             "prediction": current_predictions[:, j].ravel()})
        current_grouped_df = current_df.groupby("thresholded_cna_mask").agg({"ground_truth": "median", "prediction": "median"}).reset_index(drop=False)
        ground_truth_corrs.append(pearsonr(x=current_grouped_df["thresholded_cna_mask"].values, y=current_grouped_df["ground_truth"].values)[0])
        prediction_corrs.append(pearsonr(x=current_grouped_df["thresholded_cna_mask"].values, y=current_grouped_df["prediction"].values)[0])

    os.makedirs(os.path.join(experiment_dir, f"{split_name}_results", "histograms"), exist_ok=True)

    plt.figure(figsize=(6, 6))
    plt.title(f"Ground Truths, Split: {split_name.capitalize()}, Median Pearson Corr: {np.round(np.median(ground_truth_corrs), 2)}")
    plt.savefig(os.path.join(experiment_dir, f"{split_name}_results", "histograms", "ground_truths.png"))

    plt.figure(figsize=(6, 6))
    plt.title(f"Predictions, Split: {split_name.capitalize()}, Median Pearson Corr: {np.round(np.median(prediction_corrs), 2)}")
    plt.savefig(os.path.join(experiment_dir, f"{split_name}_results", "histograms", "predictions.png"))


def get_mse_corr_p_value(ground_truths: np.ndarray, predictions: np.ndarray) -> Tuple[float]:
    mse = mean_squared_error(y_true=ground_truths, y_pred=predictions)
    corr, p_value = pearsonr(x=ground_truths, y=predictions)
    return mse, corr, p_value


def get_gene_based_evaluation_metrics(cfg: Dict[str, Any], all_ground_truths: pd.DataFrame, all_predictions: pd.DataFrame, dataset: Dataset):
    gene_based_evaluation_metrics_data = []
    for entrezgene_id in dataset.entrezgene_ids:
        gene_based_ground_truths = all_ground_truths[entrezgene_id]
        gene_based_predictions = all_predictions[entrezgene_id]
        gene_based_mse, gene_based_corr, _ = get_mse_corr_p_value(ground_truths=gene_based_ground_truths, predictions=gene_based_predictions)
        gene_based_normalized_mse = gene_based_mse / (mean_squared_error(y_true=gene_based_ground_truths, y_pred=np.zeros_like(gene_based_ground_truths)) + cfg["normalization_eps"])
        gene_based_evaluation_metrics_data.append((entrezgene_id, gene_based_normalized_mse, gene_based_corr))
    gene_based_evaluation_metrics_df = pd.DataFrame(data=gene_based_evaluation_metrics_data, columns=["entrezgene_id", "gene_normalized_mse", "gene_corr"])
    return gene_based_evaluation_metrics_df


def get_evaluation_metrics(cancer_type: str, all_ground_truths: pd.DataFrame, all_predictions: pd.DataFrame, thresholded_cna_mask: pd.DataFrame, current_sample_ids: List[str]) -> Dict[str, float]:
    current_thresholded_cna_mask = thresholded_cna_mask[thresholded_cna_mask["sample_id"].isin(current_sample_ids)].drop(columns=["sample_id"])
    current_thresholded_cna_mask = (current_thresholded_cna_mask.values.ravel() != 0)

    current_ground_truths = all_ground_truths[all_ground_truths["sample_id"].isin(current_sample_ids)].drop(columns=["sample_id"])
    current_ground_truths = current_ground_truths.values.ravel()

    current_predictions = all_predictions[all_predictions["sample_id"].isin(current_sample_ids)].drop(columns=["sample_id"])
    current_predictions = current_predictions.values.ravel()

    current_cna_ground_truths = current_ground_truths[np.argwhere(current_thresholded_cna_mask)]
    current_cna_predictions = current_predictions[np.argwhere(current_thresholded_cna_mask)]

    current_noncna_ground_truths = current_ground_truths[np.argwhere((1 - current_thresholded_cna_mask).astype(np.bool_))]
    current_noncna_predictions = current_predictions[np.argwhere((1 - current_thresholded_cna_mask).astype(np.bool_))]

    all_mse, all_corr, all_p_value = get_mse_corr_p_value(ground_truths=current_ground_truths, predictions=current_predictions)
    cna_mse, cna_corr, cna_p_value = get_mse_corr_p_value(ground_truths=current_cna_ground_truths, predictions=current_cna_predictions)
    noncna_mse, noncna_corr, noncna_p_value = get_mse_corr_p_value(ground_truths=current_noncna_ground_truths, predictions=current_noncna_predictions)

    return (cancer_type, all_mse, all_corr, all_p_value, cna_mse, cna_corr, cna_p_value, noncna_mse, noncna_corr, noncna_p_value)


def plot_scatter_plot(cfg: Dict[str, Any], split_name: str, current_cancer_type: str, current_sample_ids: List[str], all_ground_truths: pd.DataFrame, all_predictions: pd.DataFrame) -> None:
    experiment_dir = get_experiment_dir(cfg=cfg)

    current_ground_truths = all_ground_truths[all_ground_truths["sample_id"].isin(current_sample_ids)].drop(columns=["sample_id"]).values.ravel()
    current_predictions = all_predictions[all_predictions["sample_id"].isin(current_sample_ids)].drop(columns=["sample_id"]).values.ravel()

    plt.scatter(x=current_ground_truths, y=current_predictions, alpha=0.1)
    plt.xlabel(f"Split: {split_name.capitalize()}, Cancer type: {current_cancer_type.capitalize()}, Ground truth GEX values", fontsize=28)
    plt.ylabel(f"Split: {split_name.capitalize()}, Cancer type: {current_cancer_type.capitalize()}, Predicted GEX values", fontsize=28)
    bottom_left = np.maximum(np.min(current_ground_truths), np.min(current_predictions))
    top_right = np.minimum(np.max(current_ground_truths), np.max(current_predictions))
    plt.xlim(bottom_left, top_right)
    plt.ylim(bottom_left, top_right)
    plt.plot([bottom_left, top_right], [bottom_left, top_right], color='r')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    os.makedirs(os.path.join(experiment_dir, f"{split_name}_results", "scatter_plots"), exist_ok=True)
    plt.savefig(os.path.join(experiment_dir, f"{split_name}_results", "scatter_plots", f"{current_cancer_type}.png"))


def plot_box_plots(cfg: Dict[str, Any], split_name: str, current_cancer_type: str, current_sample_ids: List[str], all_ground_truths: pd.DataFrame, all_predictions: pd.DataFrame, thresholded_cna_mask: pd.DataFrame, dataset: Dataset) -> None:
    experiment_dir = get_experiment_dir(cfg=cfg)

    current_ground_truths = all_ground_truths[all_ground_truths["sample_id"].isin(current_sample_ids)].drop(columns=["sample_id"]).values
    current_ground_truths = (current_ground_truths - dataset.y_train_mean) / dataset.y_train_std
    current_ground_truths = current_ground_truths.ravel()

    current_predictions = all_predictions[all_predictions["sample_id"].isin(current_sample_ids)].drop(columns=["sample_id"]).values
    current_predictions = (current_predictions - dataset.y_train_mean) / dataset.y_train_std
    current_predictions = current_predictions.ravel()

    current_thresholded_cna_mask = thresholded_cna_mask[thresholded_cna_mask["sample_id"].isin(current_sample_ids)].drop(columns=["sample_id"]).values.ravel()
    current_df = pd.DataFrame.from_dict({"current_ground_truths": current_ground_truths,
                                         "current_predictions": current_predictions,
                                         "current_thresholded_cna_mask": current_thresholded_cna_mask})
    current_grouped_df = current_df.groupby("current_thresholded_cna_mask").agg({"current_ground_truths": "median", "current_predictions": "median"}).reset_index(drop=False)

    plt.figure(figsize=(6, 6))
    ground_truths_median_z_corr, ground_truths_median_z_p_value = pearsonr(current_grouped_df["current_thresholded_cna_mask"].values, current_grouped_df["current_ground_truths"].values)
    plt.title(f"Ground Truths, Split: {split_name.capitalize()}, Cancer type: {current_cancer_type.capitalize()}, Pearson Corr: {np.round(ground_truths_median_z_corr, 2)}, P-Value: {np.round(ground_truths_median_z_p_value, 2)}")
    current_df.boxplot(column="current_ground_truths", by="current_thresholded_cna_mask", figsize=(10, 10), fontsize=15)
    os.makedirs(os.path.join(experiment_dir, f"{split_name}_results", "box_plots", "ground_truths"), exist_ok=True)
    plt.savefig(os.path.join(experiment_dir, f"{split_name}_results", "box_plots", "ground_truths", f"{current_cancer_type}.png"))

    plt.figure(figsize=(6, 6))
    predictions_median_z_corr, predictions_median_z_p_value = pearsonr(current_grouped_df["current_thresholded_cna_mask"].values, current_grouped_df["current_ground_truths"].values)
    plt.title(f"Predictions, Split: {split_name.capitalize()}, Cancer type: {current_cancer_type.capitalize()}, Pearson Corr: {np.round(predictions_median_z_corr, 2)}, P-Value: {np.round(predictions_median_z_p_value, 2)}")
    current_df.boxplot(column="current_predictions", by="current_thresholded_cna_mask", figsize=(10, 10), fontsize=15)
    os.makedirs(os.path.join(experiment_dir, f"{split_name}_results", "box_plots", "predictions"), exist_ok=True)
    plt.savefig(os.path.join(experiment_dir, f"{split_name}_results", "box_plots", "predictions", f"{current_cancer_type}.png"))


def save_results_split(cfg: Dict[str, Any], data_loaders: Dict[str, DataLoader], split_name: str, model: nn.Module, loss_function, dataset: Dataset, thresholded_cna_mask: pd.DataFrame, logger: logging.Logger) -> None:
    model.eval()

    all_sample_ids = []
    all_ground_truths = []
    all_predictions = []

    total_loss = 0.0
    total_count = 0.0

    with torch.no_grad():
        for batch in data_loaders[split_name]:
            sample_id_indices = batch["sample_id_indices"]
            X = batch["X"]
            y = batch["y"]

            if cfg["normalize_input"]:
                X = (X - dataset.X_train_mean) / dataset.X_train_std

            yhat = model(X)

            if cfg["normalize_output"]:
                # then during training y was manually normalized, and yhat is produced as normalized as well.
                # we should unnormalize yhat so that it is comparable to y above, which was not normalized manually during evaluation.
                yhat = yhat * dataset.y_train_std + dataset.y_train_mean

            total_count += float(y.shape[0] * y.shape[1])
            total_loss += float(loss_function(yhat, y))

            all_sample_ids.append(np.array([dataset.sample_ids[int(sample_id_index)] for sample_id_index in sample_id_indices.numpy()]))
            all_ground_truths.append(y.cpu().numpy())
            all_predictions.append(yhat.cpu().numpy())

    all_sample_ids = np.hstack(all_sample_ids)
    all_ground_truths = np.vstack(all_ground_truths)
    all_predictions = np.vstack(all_predictions)
    all_loss = total_loss / total_count

    logger.log(level=logging.INFO, msg=f"{split_name.capitalize()} {cfg['loss_function']} loss is {np.round(all_loss, 2)}.")
    logger.log(level=logging.INFO, msg=f"Saving {split_name} results...")

    experiment_dir = get_experiment_dir(cfg=cfg)
    os.makedirs(os.path.join(experiment_dir, f"{split_name}_results"), exist_ok=True)

    all_ground_truths = pd.DataFrame(data=all_ground_truths, columns=dataset.entrezgene_ids, index=all_sample_ids).reset_index(drop=False).rename(columns={"index": "sample_id"})
    all_ground_truths = all_ground_truths.sort_values(by=["sample_id"])
    all_ground_truths = all_ground_truths[["sample_id"] + sorted([column for column in all_ground_truths.drop(columns=["sample_id"]).columns])]
    all_ground_truths.to_csv(os.path.join(experiment_dir, f"{split_name}_results", "ground_truths.tsv"), sep="\t", index=False)

    all_predictions = pd.DataFrame(data=all_predictions, columns=dataset.entrezgene_ids, index=all_sample_ids).reset_index(drop=False).rename(columns={"index": "sample_id"})
    all_predictions = all_predictions.sort_values(by=["sample_id"])
    all_predictions = all_predictions[["sample_id"] + sorted([column for column in all_predictions.drop(columns=["sample_id"]).columns])]
    all_predictions.to_csv(os.path.join(experiment_dir, f"{split_name}_results", "predictions.tsv"), sep="\t", index=False)

    all_cancer_types = pd.read_csv(os.path.join(cfg["processed_data_dir"], "cancer_type.tsv"), sep="\t")
    all_cancer_types = all_cancer_types[all_cancer_types["sample_id"].isin(all_sample_ids)]
    all_cancer_types = all_cancer_types.sort_values(by=["sample_id"])
    all_cancer_types = dict(all_cancer_types.groupby("cancer_type")["sample_id"].apply(list).reset_index(name="sample_ids").values)
    all_cancer_types["all"] = all_sample_ids.tolist()

    evaluation_metrics = []

    for current_cancer_type, current_sample_ids in all_cancer_types.items():
        evaluation_metrics.append(get_evaluation_metrics(cancer_type=current_cancer_type, all_ground_truths=all_ground_truths, all_predictions=all_predictions, thresholded_cna_mask=thresholded_cna_mask, current_sample_ids=current_sample_ids))
        plot_scatter_plot(cfg=cfg, split_name=split_name, current_cancer_type=current_cancer_type, current_sample_ids=current_sample_ids, all_ground_truths=all_ground_truths, all_predictions=all_predictions)
        plot_box_plots(cfg=cfg, split_name=split_name, current_cancer_type=current_cancer_type, current_sample_ids=current_sample_ids, all_ground_truths=all_ground_truths, all_predictions=all_predictions, thresholded_cna_mask=thresholded_cna_mask)

    evaluation_metrics = pd.DataFrame(data=evaluation_metrics, columns=["cancer_type", "all_mse", "all_corr", "all_p_value", "cna_mse", "cna_corr", "cna_p_value", "noncna_mse", "noncna_corr", "noncna_p_value"])
    evaluation_metrics.to_csv(os.path.join(experiment_dir, f"{split_name}_results", "evaluation_metrics.tsv"), sep="\t", index=False)

    gene_based_evaluation_metrics_df = get_gene_based_evaluation_metrics(all_ground_truths=all_ground_truths, all_predictions=all_predictions)
    entrezgene_id_to_aug_adg_ddg_dug_ratios_mapping_df = pd.read_csv(os.path.join(cfg["processed_data_dir"], "entrezgene_id_to_aug_adg_ddg_dug_ratios_mapping.tsv"), sep="\t")
    gene_based_evaluation_metrics_df = pd.merge(gene_based_evaluation_metrics_df, entrezgene_id_to_aug_adg_ddg_dug_ratios_mapping_df, how="inner", on="entrezgene_id")
    gene_based_evaluation_metrics_df["gene_predictability"] = gene_based_evaluation_metrics_df["aug"].values + gene_based_evaluation_metrics_df["ddg"].values
    gene_based_evaluation_metrics_df["gene_nonpredictability"] = gene_based_evaluation_metrics_df["adg"].values + gene_based_evaluation_metrics_df["dug"].values

    gene_predictability_normalized_mse_corr, gene_predictability_normalized_mse_p_value = pearsonr(gene_based_evaluation_metrics_df["gene_predictability"].values, gene_based_evaluation_metrics_df["gene_normalized_mse"].values)
    gene_predictability_corr_corr, gene_predictability_corr_p_value = pearsonr(gene_based_evaluation_metrics_df["gene_predictability"], gene_based_evaluation_metrics_df["gene_corr"])
    gene_nonpredictability_normalized_mse_corr, gene_nonpredictability_normalized_mse_p_value = pearsonr(gene_based_evaluation_metrics_df["gene_nonpredictability"], gene_based_evaluation_metrics_df["gene_normalized_mse"])
    gene_nonpredictability_corr_corr, gene_nonpredictability_corr_p_value = pearsonr(gene_based_evaluation_metrics_df["gene_nonpredictability"], gene_based_evaluation_metrics_df["gene_corr"])

    gene_predictability_evaluation_metrics = pd.DataFrame.from_dict({"evaluation_metric_name": ["gene_predictability_normalized_mse_corr", "gene_predictability_normalized_mse_p_value", "gene_predictability_corr_corr", "gene_predictability_corr_p_value", "gene_nonpredictability_normalized_mse_corr", "gene_nonpredictability_normalized_mse_p_value", "gene_nonpredictability_corr_corr", "gene_nonpredictability_corr_p_value"],
                                                                     "evaluation_metric_value": [gene_predictability_normalized_mse_corr, gene_predictability_normalized_mse_p_value, gene_predictability_corr_corr, gene_predictability_corr_p_value, gene_nonpredictability_normalized_mse_corr, gene_nonpredictability_normalized_mse_p_value, gene_nonpredictability_corr_corr, gene_nonpredictability_corr_p_value]})
    gene_predictability_evaluation_metrics.to_csv(os.path.join(experiment_dir, f"{split_name}_results", "gene_predictability_evaluation_metrics.tsv"), sep="\t", index=False)

    plot_distributions_of_gene_based_pearson_corrs(cfg=cfg, all_ground_truths=all_ground_truths, all_predictions=all_predictions, thresholded_cna_mask=thresholded_cna_mask, split_name=split_name)

    logger.log(level=logging.INFO, msg=f"Saved results for {split_name} split.")


def save_results(cfg: Dict[str, Any], data_loaders: Dict[str, DataLoader], model: nn.Module, loss_function, dataset: Dataset, logger: logging.Logger) -> None:
    thresholded_cna_mask = pd.read_csv(os.path.join(cfg["processed_data_dir"], "thresholded_cna.tsv"), sep="\t")
    thresholded_cna_mask = thresholded_cna_mask.sort_values(by="sample_id")
    thresholded_cna_mask = thresholded_cna_mask[["sample_id"] + sorted([column for column in thresholded_cna_mask.drop(columns=["sample_id"]).columns])]

    for split_name in ["val", "test"]:
        save_results_split(cfg=cfg, data_loaders=data_loaders, split_name=split_name, model=model, loss_function=loss_function, dataset=dataset, thresholded_cna_mask=thresholded_cna_mask, logger=logger)
